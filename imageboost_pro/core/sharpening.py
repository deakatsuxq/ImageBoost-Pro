"""Luminance sharpening with noise threshold and local halo limiting."""

import cv2
import numpy as np
from ._image import validate_rgb, uint8


def sharpen_image(rgb, method="unsharp", amount=50, radius=1.2, threshold=3,
                  detail=0, microcontrast=0, halo_protection=75):
    rgb = validate_rgb(rgb)
    if amount <= 0 and detail <= 0 and microcontrast <= 0:
        return rgb.copy()
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    light = lab[..., 0].astype(np.float32)
    blur = cv2.GaussianBlur(light, (0, 0), radius)
    high = light - blur
    if method == "unsharp":
        residual = high
    elif method == "highpass":
        residual = 24 * np.tanh(high / 24)
    elif method == "laplacian":
        smooth = cv2.GaussianBlur(light, (0, 0), max(0.4, radius / 2))
        residual = -cv2.Laplacian(smooth, cv2.CV_32F, ksize=1)
    elif method == "adaptive":
        variance = cv2.GaussianBlur(light * light, (0, 0), radius) - blur * blur
        residual = high * np.clip(variance / (variance + 50), 0, 1)
    else:
        raise ValueError(f"Неизвестный метод резкости: {method}")
    residual *= np.clip((np.abs(high) - threshold) / max(1, threshold + 1), 0, 1)
    corrected = light + residual * amount / 50
    if detail:
        fine = light - cv2.GaussianBlur(light, (0, 0), 0.7)
        corrected += fine * (np.abs(fine) > max(2, threshold)) * detail / 100
    if microcontrast:
        corrected += (light - cv2.GaussianBlur(light, (0, 0), 10)) * microcontrast / 200
    kernel = np.ones((3, 3), np.uint8)
    margin = 32 * (1 - halo_protection / 100)
    low = cv2.erode(light, kernel) - margin
    high_limit = cv2.dilate(light, kernel) + margin
    corrected = np.clip(corrected, low, high_limit)
    output = lab.copy()
    output[..., 0] = uint8(corrected)
    return cv2.cvtColor(output, cv2.COLOR_LAB2RGB)
