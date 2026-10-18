"""Exposure, gray-world balance, tonal and saturation corrections in RGB."""

import numpy as np
from ._image import validate_rgb, uint8


def correct_color(rgb, *, exposure=0, brightness=0, contrast=0, saturation=0,
                  vibrance=0, temperature=0, tint=0, white_balance=False,
                  shadows=0, highlights=0, black_point=0, white_point=0):
    rgb = validate_rgb(rgb)
    if not any((exposure, brightness, contrast, saturation, vibrance, temperature, tint,
                white_balance, shadows, highlights, black_point, white_point)):
        return rgb.copy()
    image = rgb.astype(np.float32) / 255
    if white_balance:
        # Neutral gray-world assumes that mean scene reflectance is approximately neutral.
        # Clip gains to avoid destroying intentionally monochromatic scenes.
        means = np.mean(image.reshape(-1, 3), axis=0)
        gain = np.clip(np.mean(means) / np.maximum(means, 1 / 255), 0.65, 1.55)
        image *= gain
    if temperature or tint:
        image *= np.array([1 + temperature / 400 + tint / 800,
                           1 - tint / 400, 1 - temperature / 400 + tint / 800], np.float32)
    image *= 2 ** exposure
    image += brightness / 200
    image = (image - 0.5) * (1 + contrast / 100) + 0.5
    lum = image @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    tones = np.clip(lum, 0, 1)
    delta = shadows / 100 * 0.4 * (1 - tones) ** 3
    delta += highlights / 100 * 0.4 * tones ** 3
    delta += black_point / 100 * 0.25 * (1 - tones) ** 6
    delta += white_point / 100 * 0.25 * tones ** 6
    image += delta[..., None]
    lum = image @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    chroma = image - lum[..., None]
    current_sat = np.max(image, axis=2) - np.min(image, axis=2)
    factor = 1 + saturation / 100 + vibrance / 100 * (1 - np.clip(current_sat, 0, 1))
    image = lum[..., None] + chroma * factor[..., None]
    return uint8(image * 255)
