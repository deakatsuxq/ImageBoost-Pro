"""Noise removal with independently controlled LAB luminance and chroma."""

import cv2
import numpy as np
from ._image import validate_rgb, uint8


def _colored_nlm(rgb, luma_strength, chroma_strength, cancelled=None):
    """Bound NLM working memory and allow cancellation between overlapping tiles."""
    height, width = rgb.shape[:2]
    result = np.empty_like(rgb)
    tile, overlap = 512, 16  # covers the 21-pixel search and 7-pixel template support
    for top in range(0, height, tile):
        for left in range(0, width, tile):
            if cancelled is not None and cancelled():
                from .pipeline import ProcessingCancelled
                raise ProcessingCancelled("Шумоподавление отменено")
            bottom, right = min(top + tile, height), min(left + tile, width)
            y0, x0 = max(0, top - overlap), max(0, left - overlap)
            y1, x1 = min(height, bottom + overlap), min(width, right + overlap)
            # OpenCV's coloured NLM API specifically expects BGR, unlike our public RGB contract.
            bgr = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2BGR)
            filtered = cv2.fastNlMeansDenoisingColored(bgr, None, luma_strength, chroma_strength, 7, 21)
            filtered = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
            result[top:bottom, left:right] = filtered[top - y0:bottom - y0, left - x0:right - x0]
    return result


def denoise_image(rgb, method="nlm", strength=50, luma=100, chroma=100,
                  detail_preservation=50, edge_protection=50, cancelled=None):
    rgb = validate_rgb(rgb)
    if strength <= 0 or (luma <= 0 and chroma <= 0):
        return rgb.copy()
    amount = float(strength) / 100
    if method == "gaussian":
        result = cv2.GaussianBlur(rgb, (0, 0), 0.35 + amount * 2)
    elif method == "median":
        result = cv2.medianBlur(rgb, 3 if strength < 45 else (5 if strength < 80 else 7))
    elif method == "bilateral":
        result = cv2.bilateralFilter(rgb, 7, 8 + strength * 0.8, 2 + strength / 15)
    elif method == "nlm":
        result = _colored_nlm(rgb, 1 + strength * 0.18, 1 + strength * 0.23, cancelled)
    elif method == "wavelet":
        from skimage.restoration import denoise_wavelet
        result = uint8(denoise_wavelet(rgb.astype(np.float32) / 255,
                                     sigma=(1 + strength * 0.3) / 255,
                                     method="BayesShrink", mode="soft", wavelet="db2",
                                     channel_axis=-1, convert2ycbcr=True,
                                     rescale_sigma=True) * 255)
    else:
        raise ValueError(f"Неизвестный метод шумоподавления: {method}")
    source_lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    result_lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = source_lab[..., 0]
    gradient = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
                             cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    edges = np.clip(gradient / 160, 0, 1)
    texture = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 1))
    protection = 1 - (edge_protection / 100 * 0.75) * edges
    protection *= 1 - detail_preservation / 100 * 0.35 * np.clip(texture / 24, 0, 1)
    mix = np.array([luma, chroma, chroma], dtype=np.float32) / 100
    mixed = source_lab + (result_lab - source_lab) * protection[..., None] * mix * amount
    return cv2.cvtColor(uint8(mixed), cv2.COLOR_LAB2RGB)
