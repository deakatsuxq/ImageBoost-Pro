"""Low-light enhancement: edge-preserving denoise, gamma, CLAHE and Retinex.

These operations improve visibility; they cannot recover information absent in a
black or clipped source. Strong modes intentionally trade natural light for detail.
"""

import cv2
import numpy as np
from ._image import validate_rgb, uint8
from .denoising import _colored_nlm

NIGHT_PRESETS = {
    "natural": {"gamma": 1.5, "clahe": 20, "retinex": 8},
    "balanced": {"gamma": 1.9, "clahe": 36, "retinex": 20},
    "maximum": {"gamma": 2.5, "clahe": 55, "retinex": 36},
}


def multiscale_retinex(light: np.ndarray) -> np.ndarray:
    """Illumination-normalized luminance in [0,255], with robust percentile range."""
    source = light.astype(np.float32) + 1
    response = np.zeros_like(source)
    shorter = min(source.shape)
    for sigma in (max(0.8, shorter * 0.015), max(1.6, shorter * 0.08), max(3.2, shorter * 0.25)):
        illumination = cv2.GaussianBlur(source, (0, 0), sigma)
        response += (np.log(source) - np.log(np.maximum(illumination, 1))) / 3
    lower, upper = np.percentile(response, [1, 99])
    if upper - lower < 1e-5:
        return light.astype(np.float32).copy()
    return np.clip((response - lower) / (upper - lower), 0, 1) * 255


def recover_night(rgb, mode="balanced", strength=65, gamma=1, clahe=0, retinex=0,
                  highlight_protection=80, cancelled=None):
    rgb = validate_rgb(rgb)
    if mode not in {"off", *NIGHT_PRESETS}:
        raise ValueError("Неизвестный режим ночного восстановления")
    if mode == "off" and gamma == 1 and clahe == 0 and retinex == 0:
        return rgb.copy()
    if mode != "off" and strength <= 0:
        return rgb.copy()
    base = rgb
    amount = strength / 100 if mode != "off" else 1
    if mode != "off":
        # Filtering before lifting shadows avoids magnifying existing sensor noise.
        base = _colored_nlm(rgb, 2 + amount * 3, 3 + amount * 5, cancelled)
        preset = NIGHT_PRESETS[mode]
        gamma *= preset["gamma"]
        clahe = max(clahe, preset["clahe"])
        retinex = max(retinex, preset["retinex"])
    lab = cv2.cvtColor(base, cv2.COLOR_RGB2LAB)
    light = lab[..., 0].astype(np.float32)
    original = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    corrected = np.power(light / 255, 1 / gamma) * 255
    if clahe:
        grid = (min(8, rgb.shape[1]), min(8, rgb.shape[0]))
        local = cv2.createCLAHE(clipLimit=1 + clahe / 30, tileGridSize=grid).apply(uint8(corrected))
        corrected += (local.astype(np.float32) - corrected) * clahe / 100
    if retinex:
        normalized = multiscale_retinex(corrected)
        corrected += (normalized - corrected) * retinex / 100
    # Bright lights keep their original luminance. The rolloff is continuous.
    protect = 1 - highlight_protection / 100 * np.clip((original / 255 - 0.55) / 0.4, 0, 1)
    # Truly black pixels have no recoverable data and must not become artificial gray.
    visible = np.clip(original / 8, 0, 1)
    lab[..., 0] = uint8(original + (corrected - original) * protect * visible * amount)
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if mode != "off":
        # Modest preset-dependent saturation restores colour lost under low luminance.
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
        hsv[..., 1] = uint8(hsv[..., 1].astype(np.float32) * (1 + 0.08 * amount))
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return enhanced
