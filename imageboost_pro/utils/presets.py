"""Validated, portable JSON presets; never pickle Python objects."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from imageboost_pro.core.settings import ProcessingSettings

BUILTIN_PRESETS = {
    "Автоматическое улучшение": ProcessingSettings(),  # UI evaluates the current analysis.
    "Мягкое улучшение": ProcessingSettings(contrast=5, sharpen_amount=12),
    "Максимальная детализация": ProcessingSettings(sharpen_method="adaptive", sharpen_amount=50,
                                                  detail_amount=20, microcontrast=15),
    "Удаление шума": ProcessingSettings(denoise_strength=45),
    "Ночное фото": ProcessingSettings(night_mode="balanced"),
    "Восстановление старого фото": ProcessingSettings(denoise_strength=20, contrast=12,
                                                    white_balance=True, sharpen_amount=20),
    "Чёткий портрет": ProcessingSettings(denoise_method="bilateral", denoise_strength=15,
                                        sharpen_amount=20, halo_protection=90),
    "Увеличение x2": ProcessingSettings(scale=2),
    "Увеличение x4": ProcessingSettings(scale=4),
    "Исправление освещения": ProcessingSettings(shadows=35, highlights=-15, clahe=15),
    "Удаление JPEG-артефактов": ProcessingSettings(jpeg_artifacts=50),
    "Естественные цвета": ProcessingSettings(white_balance=True, vibrance=12),
}


def save_preset(path: str | Path, settings: ProcessingSettings) -> None:
    settings.validate()
    Path(path).write_text(json.dumps({"version": 1, "settings": asdict(settings)},
                                    ensure_ascii=False, indent=2), encoding="utf-8")


def load_preset(path: str | Path) -> ProcessingSettings:
    path = Path(path)
    if path.stat().st_size > 128_000:
        raise ValueError("Файл пресета слишком большой.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Не удалось прочитать JSON-пресет.") from exc
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("settings"), dict):
        raise ValueError("Неподдерживаемый формат пресета (ожидается версия 1).")
    return ProcessingSettings.from_dict(data["settings"])
