"""Explicit classical interpolation, without claiming neural reconstruction."""

import cv2
from ._image import validate_rgb

INTERPOLATIONS = {
    "nearest": cv2.INTER_NEAREST, "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4, "area": cv2.INTER_AREA,
}


def resize_image(rgb, width: int, height: int, interpolation: str = "lanczos"):
    rgb = validate_rgb(rgb)
    if type(width) is not int or type(height) is not int or width < 1 or height < 1:
        raise ValueError("Ширина и высота должны быть положительными целыми числами")
    if interpolation not in INTERPOLATIONS:
        raise ValueError("Неизвестный метод интерполяции")
    if (width, height) == (rgb.shape[1], rgb.shape[0]):
        return rgb.copy()
    return cv2.resize(rgb, (width, height), interpolation=INTERPOLATIONS[interpolation])
