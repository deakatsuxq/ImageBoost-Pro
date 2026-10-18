"""Native Real-ESRGAN inference, optional torch and genuine learned upscaling."""

from __future__ import annotations

import importlib.util
import math
import re
from pathlib import Path
from typing import Callable

import numpy as np

from ..core.pipeline import ProcessingCancelled as PipelineCancelled


class ProcessingCancelled(PipelineCancelled, RuntimeError):
    pass


def runtime_status() -> dict:
    if importlib.util.find_spec("torch") is None:
        return {"available": False, "device": "cpu", "cuda": False,
                "error": "PyTorch не установлен. Установите requirements-ai.txt; AI-дистрибутив включает PyTorch."}
    try:
        import torch

        if tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2]) < (2, 6):
            raise RuntimeError("Требуется PyTorch 2.6 или новее для безопасной загрузки весов.")
        cuda = torch.cuda.is_available()
        return {"available": True, "device": "cuda" if cuda else "cpu", "cuda": cuda, "error": ""}
    except Exception as error:
        return {"available": False, "device": "cpu", "cuda": False, "error": str(error)}


def _load_network(model_path: str | Path):
    status = runtime_status()
    if not status["available"]:
        raise RuntimeError(status["error"])
    import torch

    from .rrdbnet import RRDBNet

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Файл модели не найден: {path}")
    if path.stat().st_size > 100 * 1024 * 1024:
        raise ValueError("Ожидались официальные RRDBNet-веса размером не более 100 МиБ.")
    # Never fall back to pickle's unsafe weights_only=False, even for older files.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Файл не содержит словарь весов Real-ESRGAN.")
    state = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Файл не содержит допустимые тензоры RRDBNet.")
    first = state.get("conv_first.weight")
    if first is None or tuple(first.shape) not in ((64, 3, 3, 3), (64, 12, 3, 3)):
        raise ValueError("Поддерживаются только официальные Real-ESRGAN RRDBNet x2plus/x4plus/anime 6B.")
    blocks = {int(m.group(1)) for key in state if (m := re.match(r"^body\.(\d+)\.", key))}
    scale = 2 if first.shape[1] == 12 else 4
    count = len(blocks)
    if blocks != set(range(count)) or (scale, count) not in ((2, 23), (4, 23), (4, 6)):
        raise ValueError("Неподдерживаемая архитектура Real-ESRGAN: ожидаются 23 либо 6 RRDB-блоков.")
    network = RRDBNet(scale=scale, num_block=count)
    network.load_state_dict(state, strict=True)
    # Reject NaN/Inf checkpoints before they can contaminate results.
    if not all(torch.isfinite(parameter).all().item() for parameter in network.parameters()):
        raise ValueError("Модель содержит некорректные NaN/Inf-веса.")
    network.eval()
    return network


def validate_model(model_path: str | Path) -> dict:
    network = _load_network(model_path)
    return {"scale": network.scale, "num_blocks": network.num_block,
            "architecture": "RRDBNet", "pretrained_quality_verified": False}


def _check_cancelled(cancelled):
    if cancelled and cancelled():
        raise ProcessingCancelled("AI-обработка отменена.")


def _run_tiled(network, image, tile_size, progress=None, cancelled=None):
    import torch

    _, _, height, width = image.shape
    scale = network.scale
    # Pixel-unshuffle phase must be the same for every tile of x2 models.
    alignment = 2 if scale == 2 else 1
    tile_size = max(alignment, tile_size // alignment * alignment)
    halo = math.ceil(network.halo / alignment) * alignment
    output = torch.empty((1, 3, height * scale, width * scale), dtype=image.dtype, device="cpu")
    columns = math.ceil(width / tile_size)
    rows = math.ceil(height / tile_size)
    done = 0
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            _check_cancelled(cancelled)
            right, bottom = min(x + tile_size, width), min(y + tile_size, height)
            left_in, top_in = max(0, x - halo), max(0, y - halo)
            right_in, bottom_in = min(width, right + halo), min(height, bottom + halo)
            tile = image[:, :, top_in:bottom_in, left_in:right_in].to(next(network.parameters()).device)
            prediction = network(tile)
            source_x, source_y = (x - left_in) * scale, (y - top_in) * scale
            output[:, :, y * scale:bottom * scale, x * scale:right * scale] = prediction[
                :, :, source_y:source_y + (bottom - y) * scale,
                source_x:source_x + (right - x) * scale].cpu()
            done += 1
            if progress:
                progress(done / (columns * rows) * 100.0)
    return output


def upscale(rgb: np.ndarray, model_path: str | Path, scale: float,
            tile_size: int = 256, device: str = "auto",
            progress: Callable[[float], None] | None = None,
            cancelled: Callable[[], bool] | None = None) -> np.ndarray:
    """Real learned upscale with native inference before optional final resampling.

    Repeated native inference reaches the requested scale; Lanczos resamples only
    afterwards when the requested target is not a native power of the model.
    Full receptive-field halos avoid visible tile seams, at additional compute cost.
    Output/intermediates are limited to 64 million RGB pixels to bound memory use.
    """
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Ожидается непустое RGB-изображение uint8 (H, W, 3).")
    if not rgb.shape[0] or not rgb.shape[1]:
        raise ValueError("Изображение пустое.")
    if (isinstance(scale, bool) or not isinstance(scale, (int, float))
            or not math.isfinite(scale) or not 0.01 <= scale <= 16):
        raise ValueError("AI-масштаб должен быть конечным числом от 0.01 до 16.")
    if not isinstance(tile_size, int) or isinstance(tile_size, bool) or not 16 <= tile_size <= 2048:
        raise ValueError("Размер AI-тайла должен быть целым числом от 16 до 2048.")
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("Устройство AI должно быть auto, cpu или cuda.")
    wanted_height = max(1, round(rgb.shape[0] * scale))
    wanted_width = max(1, round(rgb.shape[1] * scale))
    if wanted_height * wanted_width > 64_000_000:
        raise ValueError("Результат AI превышает 64 мегапикселя. Уменьшите масштаб или исходное фото.")
    _check_cancelled(cancelled)
    network = _load_network(model_path)
    import torch
    from torch.nn import functional as F

    passes = max(1, math.ceil(math.log(scale, network.scale)))
    native_scale = network.scale ** passes
    if rgb.shape[0] * rgb.shape[1] * native_scale ** 2 > 64_000_000:
        raise ValueError("Промежуточный результат нейросети превышает 64 Мп. Выберите модель x2, "
                         "уменьшите масштаб или исходное фото.")
    use_cuda = torch.cuda.is_available() if device == "auto" else device == "cuda"
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна. Выберите CPU или установите совместимый PyTorch/CUDA.")
    network.to("cuda" if use_cuda else "cpu")
    image = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).float() / 255.0
    if progress:
        progress(0.0)
    try:
        with torch.inference_mode():
            for current in range(passes):
                _check_cancelled(cancelled)
                height, width = image.shape[2:]
                # Mirror the official pre-padding; replicate supports tiny photographs too.
                pad_right = 10 + ((width + 10) % 2 if network.scale == 2 else 0)
                pad_bottom = 10 + ((height + 10) % 2 if network.scale == 2 else 0)
                if (width + pad_right) * (height + pad_bottom) * network.scale ** 2 > 64_000_000:
                    raise ValueError("Промежуточный результат с полями нейросети превышает 64 Мп. "
                                     "Уменьшите исходное фото или масштаб.")
                mode = "reflect" if height > pad_bottom and width > pad_right else "replicate"
                padded = F.pad(image, (0, pad_right, 0, pad_bottom), mode=mode)
                def update(percent, pass_number=current):
                    if progress:
                        progress((pass_number + percent / 100.0) / passes * 100.0)
                image = _run_tiled(network, padded, tile_size, update, cancelled)
                image = image[:, :, :height * network.scale, :width * network.scale].clamp_(0, 1)
                # Remain in float between passes; only quantize the final result.
        _check_cancelled(cancelled)
        result = image[0].permute(1, 2, 0).numpy()
        result = np.rint(result * 255.0).clip(0, 255).astype(np.uint8)
        if result.shape[:2] != (wanted_height, wanted_width):
            import cv2

            result = cv2.resize(result, (wanted_width, wanted_height), interpolation=cv2.INTER_LANCZOS4)
        return result
    except torch.OutOfMemoryError as error:
        raise RuntimeError("Недостаточно памяти для AI. Уменьшите фото/тайл или выберите CPU.") from error
    finally:
        if use_cuda:
            del network
            torch.cuda.empty_cache()
