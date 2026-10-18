"""Immutable, cancellable image processing pipeline shared by previews and export."""

from collections.abc import Callable
import math
from ._image import validate_rgb
from .settings import ProcessingSettings
from .upscaling import resize_image
from .denoising import denoise_image
from .sharpening import sharpen_image
from .color_correction import correct_color
from .night_recovery import recover_night
from .artifact_removal import remove_jpeg_artifacts, reduce_banding, deconvolve_image
from ._resources import available_memory_bytes


class ProcessingCancelled(Exception):
    """A cooperative cancellation; original and settings stay untouched."""


def output_size(width: int, height: int, settings: ProcessingSettings) -> tuple[int, int]:
    if type(width) is not int or type(height) is not int or min(width, height) < 1:
        raise ValueError("Некорректное исходное разрешение")
    if not isinstance(settings, ProcessingSettings):
        raise ValueError("Требуется ProcessingSettings")
    wanted_w, wanted_h = settings.target_width, settings.target_height
    if wanted_w or wanted_h:
        if settings.keep_aspect:
            ratio = min(wanted_w / width if wanted_w else math.inf,
                        wanted_h / height if wanted_h else math.inf)
            out_width, out_height = max(1, round(width * ratio)), max(1, round(height * ratio))
        else:
            out_width, out_height = wanted_w or width, wanted_h or height
    else:
        out_width = max(1, round(width * settings.scale))
        out_height = max(1, round(height * settings.scale))
    if max(out_width, out_height) > 100_000 or out_width * out_height > settings.max_output_pixels:
        raise ValueError(f"Итоговое разрешение {out_width}×{out_height} превышает лимит "
                         f"{settings.max_output_pixels:,} пикселей. Уменьшите масштаб.")
    return out_width, out_height


def process_image(rgb, settings: ProcessingSettings,
                  progress: Callable[[int, str], None] | None = None,
                  cancelled: Callable[[], bool] | None = None):
    rgb = validate_rgb(rgb)
    if not isinstance(settings, ProcessingSettings):
        raise ValueError("Требуется ProcessingSettings")
    size = output_size(rgb.shape[1], rgb.shape[0], settings)
    if rgb.shape[0] * rgb.shape[1] > 128_000_000:
        raise ValueError("Исходник превышает безопасный лимит 128 Мп. Уменьшите изображение.")

    def check_cancel():
        if cancelled is not None and cancelled():
            raise ProcessingCancelled("Обработка отменена")

    check_cancel()
    stages = []
    s = settings
    if s.jpeg_artifacts:
        stages.append(("Удаление JPEG-артефактов", lambda image: remove_jpeg_artifacts(image, s.jpeg_artifacts)))
    if s.banding_reduction:
        stages.append(("Сглаживание цветовых полос", lambda image: reduce_banding(image, s.banding_reduction)))
    if s.denoise_strength:
        stages.append(("Шумоподавление", lambda image: denoise_image(
            image, s.denoise_method, s.denoise_strength, s.denoise_luma, s.denoise_chroma,
            s.detail_preservation, s.edge_protection, cancelled)))
    if s.night_mode != "off" or s.gamma != 1 or s.clahe or s.retinex:
        stages.append(("Восстановление освещения", lambda image: recover_night(
            image, s.night_mode, s.night_strength, s.gamma, s.clahe, s.retinex,
            s.highlight_protection, cancelled)))
    color_names = ("exposure", "brightness", "contrast", "saturation", "vibrance", "temperature",
                   "tint", "white_balance", "shadows", "highlights", "black_point", "white_point")
    if any(getattr(s, name) for name in color_names):
        stages.append(("Цвет и освещение", lambda image: correct_color(
            image, **{name: getattr(s, name) for name in color_names})))
    if s.deblur:
        stages.append(("Деконволюция нерезкости", lambda image: deconvolve_image(image, s.deblur)))
    if s.upscaler == "realesrgan":
        if not s.ai_model_path:
            raise ValueError("Подключите локальные веса Real-ESRGAN в настройках AI")

        def neural(image):
            from ..ai.realesrgan_engine import upscale
            factor = max(size[0] / image.shape[1], size[1] / image.shape[0])
            # The engine propagates actual tile progress. Boundaries remain cooperative.
            stage_index = next(index for index, (name, _) in enumerate(stages) if name == "Real-ESRGAN")
            def ai_progress(percent, text="Real-ESRGAN"):
                check_cancel()
                if progress:
                    progress(round((stage_index + percent / 100) / len(stages) * 100), text)
            result = upscale(image, s.ai_model_path, factor, tile_size=s.ai_tile_size,
                             device=s.ai_device, progress=ai_progress, cancelled=cancelled)
            return resize_image(result, *size, interpolation=s.interpolation)

        stages.append(("Real-ESRGAN", neural))
    elif size != (rgb.shape[1], rgb.shape[0]):
        stages.append(("Изменение разрешения", lambda image: resize_image(image, *size, s.interpolation)))
    if s.sharpen_amount or s.detail_amount or s.microcontrast:
        stages.append(("Резкость и детали", lambda image: sharpen_image(
            image, s.sharpen_method, s.sharpen_amount, s.sharpen_radius, s.sharpen_threshold,
            s.detail_amount, s.microcontrast, s.halo_protection)))
    # Float colour transforms can keep several full-size arrays alive. This is a
    # conservative estimate, not a promise of peak usage or a substitute for caps.
    peak_pixels = max(rgb.shape[0] * rgb.shape[1], size[0] * size[1])
    required_bytes = peak_pixels * (72 if stages else 3) + rgb.nbytes
    available_bytes = available_memory_bytes()
    if available_bytes is not None and required_bytes > available_bytes * 0.75:
        raise MemoryError("Недостаточно свободной памяти для безопасной обработки. "
                          "Уменьшите разрешение или закройте другие приложения.")
    image = rgb.copy()
    if progress:
        progress(0, "Подготовка")
    for index, (label, stage) in enumerate(stages):
        check_cancel()
        if progress:
            progress(round(index / len(stages) * 100), label)
        image = validate_rgb(stage(image))
        check_cancel()
        if progress:
            progress(round((index + 1) / len(stages) * 100), label)
    check_cancel()
    if progress:
        progress(100, "Готово")
    return image
