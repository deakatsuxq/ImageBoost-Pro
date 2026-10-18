"""Validated, immutable and JSON-serializable processing parameters."""

from dataclasses import dataclass, fields
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class ProcessingSettings:
    scale: float = 1.0
    target_width: int = 0
    target_height: int = 0
    keep_aspect: bool = True
    interpolation: str = "lanczos"
    upscaler: str = "classical"
    ai_model_path: str = ""
    ai_tile_size: int = 256
    ai_device: str = "auto"
    denoise_method: str = "nlm"
    denoise_strength: float = 0
    denoise_luma: float = 100
    denoise_chroma: float = 100
    detail_preservation: float = 50
    edge_protection: float = 50
    sharpen_method: str = "unsharp"
    sharpen_amount: float = 0
    sharpen_radius: float = 1.2
    sharpen_threshold: float = 3
    detail_amount: float = 0
    microcontrast: float = 0
    halo_protection: float = 75
    night_mode: str = "off"
    night_strength: float = 65
    gamma: float = 1
    clahe: float = 0
    retinex: float = 0
    highlight_protection: float = 80
    exposure: float = 0
    brightness: float = 0
    contrast: float = 0
    saturation: float = 0
    vibrance: float = 0
    temperature: float = 0
    tint: float = 0
    white_balance: bool = False
    shadows: float = 0
    highlights: float = 0
    black_point: float = 0
    white_point: float = 0
    jpeg_artifacts: float = 0
    banding_reduction: float = 0
    deblur: float = 0
    max_output_pixels: int = 64_000_000

    def validate(self) -> None:
        """Validate saved or externally supplied parameters without coercion."""
        self.__post_init__()

    def __post_init__(self) -> None:
        enums = {
            "interpolation": {"nearest", "bilinear", "bicubic", "lanczos", "area"},
            "upscaler": {"classical", "realesrgan"},
            "ai_device": {"auto", "cpu", "cuda"},
            "denoise_method": {"gaussian", "median", "bilateral", "nlm", "wavelet"},
            "sharpen_method": {"unsharp", "highpass", "laplacian", "adaptive"},
            "night_mode": {"off", "natural", "balanced", "maximum"},
        }
        for name, choices in enums.items():
            if not isinstance(getattr(self, name), str) or getattr(self, name) not in choices:
                raise ValueError(f"Недопустимое значение {name}: {getattr(self, name)!r}")
        for name in ("keep_aspect", "white_balance"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name}: требуется логическое значение")
        for name, low, high in (
            ("target_width", 0, 100_000), ("target_height", 0, 100_000),
            ("ai_tile_size", 16, 2048), ("max_output_pixels", 1, 256_000_000),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name}: требуется целое число {low}–{high}")
        numeric_ranges = {
            "scale": (0.01, 16), "sharpen_radius": (0.1, 20),
            "sharpen_threshold": (0, 100), "gamma": (0.1, 5), "exposure": (-5, 5),
        }
        signed = {"brightness", "contrast", "saturation", "vibrance", "temperature", "tint",
                  "shadows", "highlights", "black_point", "white_point"}
        excluded = set(enums) | {"keep_aspect", "white_balance", "target_width", "target_height",
                                "ai_tile_size", "max_output_pixels", "ai_model_path"}
        for field in fields(self):
            if field.name in excluded:
                continue
            low, high = numeric_ranges.get(field.name, (-100, 100) if field.name in signed else (0, 100))
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{field.name}: требуется конечное число")
            if not low <= value <= high:
                raise ValueError(f"{field.name}: допустимый диапазон {low}–{high}")
        if not isinstance(self.ai_model_path, str) or len(self.ai_model_path) > 4096:
            raise ValueError("Некорректный путь AI-модели")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ProcessingSettings":
        if not isinstance(values, Mapping):
            raise ValueError("Настройки должны быть объектом JSON")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Неизвестные параметры: {', '.join(sorted(map(str, unknown)))}")
        try:
            return cls(**dict(values))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Некорректные настройки: {error}") from error
