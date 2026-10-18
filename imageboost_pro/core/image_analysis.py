"""Deterministic, local quality measurements for an RGB photograph.

Measurements are useful heuristics rather than a semantic judgment: deliberately
dark scenes, shallow depth of field, and strongly coloured subjects can naturally
trigger recommendations. No model, network service, or training data is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from ._image import validate_rgb

if TYPE_CHECKING:
    from .settings import ProcessingSettings


@dataclass(frozen=True)
class AnalysisReport:
    """Measurements with explanations suitable for showing directly in the UI.

    Clipping values are pixel fractions (0..1); colour cast is a percentage;
    brightness, contrast and noise are in 8-bit luminance levels. Sharpness is
    Laplacian variance, entropy is bits/pixel, and JPEG grid strength is the
    excess luminance discontinuity at eight-pixel boundaries. Width and height
    describe the original, not the analysis preview. Noise and sharpness use an
    unscaled central crop; global statistics use the bounded analysis preview.
    """

    metrics: dict[str, float]
    issues: tuple[str, ...]
    recommendations: tuple[str, ...]

    def to_text(self) -> str:
        m = self.metrics
        lines = [
            "Анализ качества фотографии",
            f"Размер: {int(m['width'])} × {int(m['height'])} пикселей",
            f"Средняя яркость: {m['brightness']:.1f} / 255",
            f"Контраст (стандартное отклонение): {m['contrast']:.1f}",
            f"Оценка шума: {m['noise_sigma']:.2f}",
            f"Резкость (дисперсия лапласиана): {m['sharpness']:.1f}",
            f"Потери в тенях: {m['clipped_shadows']:.1%}",
            f"Потери в светах: {m['clipped_highlights']:.1%}",
            f"Возможный цветовой сдвиг: {m['color_cast']:.1f}%",
            f"Выраженность сетки JPEG: {m['jpeg_grid_strength']:.2f}",
            f"Энтропия яркости: {m['entropy']:.2f} бит/пиксель",
            f"Масштаб общего анализа: {m['analysis_scale']:.1%}",
            "Шум и резкость оценены по центральному участку в исходном масштабе (до 1600 × 1600).",
            "",
            "Наблюдения:",
        ]
        lines.extend(f"• {item}" for item in self.issues or ("Выраженные проблемы не обнаружены.",))
        lines.extend(("", "Рекомендации:"))
        lines.extend(f"• {item}" for item in self.recommendations or ("Сохраните естественный вид изображения.",))
        lines.extend(("", "Это эвристическая оценка: учитывайте сюжет и замысел фотографии."))
        return "\n".join(lines)

    def suggest_settings(self, mode: str = "balanced") -> ProcessingSettings:
        """Return a conservative starting point, never modifying a photograph.

        The selected mode controls strength; the result remains editable by the
        user. A single photograph cannot establish whether blur or a colour cast
        was intended, so correction amounts are deliberately bounded.
        """
        from .settings import ProcessingSettings

        factors = {"natural": 0.65, "balanced": 1.0, "maximum": 1.35}
        if mode not in factors:
            raise ValueError("Режим анализа: natural, balanced или maximum.")
        factor = factors[mode]
        m = self.metrics
        noise = m["noise_sigma"]
        dark = m["brightness"] < 70
        low_contrast = m["contrast"] < 35
        soft = m["sharpness"] < 140 and m["contrast"] > 6
        return ProcessingSettings(
            denoise_strength=min(45.0, noise * 1.7 * factor) if noise > 4 else 0.0,
            sharpen_amount=28.0 * factor if soft else 0.0,
            detail_amount=8.0 * factor if soft and noise < 8 else 0.0,
            night_mode=mode if dark else "off",
            night_strength=min(90.0, 55.0 * factor),
            contrast=10.0 * factor if low_contrast else 0.0,
            shadows=12.0 * factor if m["clipped_shadows"] > 0.03 and not dark else 0.0,
            highlights=-15.0 * factor if m["clipped_highlights"] > 0.03 else 0.0,
            white_balance=m["color_cast"] > 20,
            jpeg_artifacts=min(40.0, m["jpeg_grid_strength"] * 4 * factor)
            if m["jpeg_grid_strength"] > 2
            else 0.0,
        )


def _noise_sigma(gray: np.ndarray) -> float:
    """Robust high-pass MAD in relatively flat areas (8-bit sigma units).

    The separable second-difference kernel has an L2 norm of six. MAD's Gaussian
    calibration therefore converts response magnitude to a noise standard
    deviation without mistaking isolated strong edges for pervasive noise.
    """
    if min(gray.shape) < 3:
        return 0.0
    smooth = cv2.GaussianBlur(gray, (0, 0), 1.2)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)[1:-1, 1:-1]
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    highpass = cv2.filter2D(gray, cv2.CV_32F, kernel)[1:-1, 1:-1]
    flat = gradient <= max(2.0, float(np.percentile(gradient, 60)))
    values = highpass[flat]
    if values.size == 0:
        return 0.0
    median = np.median(values)
    return float(np.median(np.abs(values - median)) / (0.67448975 * 6.0))


def _jpeg_grid_strength(gray: np.ndarray) -> float:
    """Estimate block boundaries at the original JPEG grid spacing."""
    strengths = []
    for axis in (0, 1):
        diff = np.abs(np.diff(gray, axis=axis))
        length = diff.shape[axis]
        if length < 16:
            continue
        positions = np.arange(1, length + 1)
        boundary = positions % 8 == 0
        boundary_values = np.take(diff, np.flatnonzero(boundary), axis=axis)
        other_values = np.take(diff, np.flatnonzero(~boundary), axis=axis)
        strengths.append(max(0.0, float(boundary_values.mean() - other_values.mean())))
    return float(np.mean(strengths)) if strengths else 0.0


def analyze_image(rgb: np.ndarray) -> AnalysisReport:
    """Analyze an RGB uint8 array without modifying it.

    Global statistics use a preview area-resampled to at most 1600 px per side.
    Noise and sharpness use an unscaled central crop of at most 1600 × 1600 px,
    preserving the original noise magnitude and spatial frequency. A crop can
    miss defects elsewhere in a photograph, so these are representative local
    estimates. JPEG boundaries use an unscaled top-left crop, preserving the
    original eight-pixel grid. ``analysis_scale`` describes the global preview.
    """
    rgb = validate_rgb(rgb)
    height, width = rgb.shape[:2]

    scale = min(1.0, 1600.0 / max(height, width))
    preview = rgb
    if scale < 1:
        preview = cv2.resize(
            rgb,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    gray8 = cv2.cvtColor(preview, cv2.COLOR_RGB2GRAY)
    gray = gray8.astype(np.float32)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    sample_y = max(0, (height - 1600) // 2)
    sample_x = max(0, (width - 1600) // 2)
    sample = rgb[sample_y:sample_y + 1600, sample_x:sample_x + 1600]
    sample_gray = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY).astype(np.float32)
    noise = _noise_sigma(sample_gray)
    sharpness = float(cv2.Laplacian(sample_gray, cv2.CV_32F).var())
    shadows = float(np.mean(gray8 <= 3))
    highlights = float(np.mean(gray8 >= 252))
    channel_means = preview.mean(axis=(0, 1), dtype=np.float64)
    cast = float(100 * (channel_means.max() - channel_means.min()) / max(32, channel_means.max()))
    histogram = np.bincount(gray8.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram[histogram > 0] / gray8.size
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    crop = rgb[:1600, :1600]
    grid = _jpeg_grid_strength(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32))
    metrics = {
        "width": float(width),
        "height": float(height),
        "brightness": brightness,
        "contrast": contrast,
        "noise_sigma": noise,
        "sharpness": sharpness,
        "clipped_shadows": shadows,
        "clipped_highlights": highlights,
        "color_cast": cast,
        "jpeg_grid_strength": grid,
        "entropy": entropy,
        "low_resolution": float(width * height < 1_000_000),
        "analysis_scale": scale,
    }
    issues: list[str] = []
    recommendations: list[str] = []
    if brightness < 70:
        issues.append("Низкая средняя яркость: возможна недоэкспозиция или ночной сюжет.")
        recommendations.append("Попробуйте ночное восстановление с защитой светов и умеренным шумоподавлением.")
    elif brightness > 200:
        issues.append("Высокая средняя яркость: проверьте экспозицию и детали в светах.")
    if contrast < 25:
        issues.append("Низкий общий контраст.")
        recommendations.append("Умеренно повысьте контраст; для локальных деталей используйте CLAHE.")
    if noise > 6:
        issues.append("Обнаружена заметная высокочастотная составляющая, похожая на шум.")
        recommendations.append("Примените шумоподавление перед повышением резкости.")
    if sharpness < 100 and contrast > 6:
        issues.append("Невысокая резкость центрального участка в исходном масштабе; возможна размытость.")
        recommendations.append("Попробуйте умеренную нерезкую маску; утраченную информацию нельзя восстановить точно.")
    if shadows > 0.03:
        issues.append(f"Почти чёрные пиксели: {shadows:.1%}; часть деталей в тенях могла быть потеряна.")
        recommendations.append("Поднимайте тени осторожно: полностью обрезанные детали не восстанавливаются.")
    if highlights > 0.03:
        issues.append(f"Почти белые пиксели: {highlights:.1%}; возможна потеря деталей в светах.")
        recommendations.append("Уменьшите света и включите защиту светлых областей.")
    if cast > 20:
        issues.append("Средние значения RGB различаются: возможен цветовой сдвиг или преобладание одного цвета.")
        recommendations.append("Проверьте баланс белого визуально; цвет предметов тоже влияет на эту оценку.")
    if grid > 2:
        issues.append("Обнаружены усиленные границы блоков размером 8 × 8 пикселей.")
        recommendations.append("Попробуйте подавление артефактов JPEG перед повышением резкости.")
    if metrics["low_resolution"]:
        issues.append("Разрешение менее одного мегапикселя.")
        recommendations.append("Для увеличения используйте Lanczos; новые реальные детали он не создаёт.")
    return AnalysisReport(metrics, tuple(issues), tuple(recommendations))
