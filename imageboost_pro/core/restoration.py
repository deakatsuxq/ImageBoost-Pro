"""Mask-based damaged-pixel repair using real OpenCV inpainting."""

import cv2
import numpy as np
from ._image import validate_rgb


def inpaint_mask(rgb, mask, radius=3, method="telea"):
    rgb = validate_rgb(rgb)
    if not isinstance(mask, np.ndarray) or mask.shape != rgb.shape[:2]:
        raise ValueError("Маска повреждений должна совпадать с размером фотографии")
    if not np.issubdtype(mask.dtype, np.number) and mask.dtype != np.bool_:
        raise ValueError("Маска должна содержать числовые значения")
    if not np.isfinite(mask).all():
        raise ValueError("Маска содержит некорректные значения")
    if not 0.1 <= radius <= 20:
        raise ValueError("Радиус восстановления: 0.1–20 пикселей")
    algorithms = {"telea": cv2.INPAINT_TELEA, "navier_stokes": cv2.INPAINT_NS}
    if method not in algorithms:
        raise ValueError("Неизвестный метод восстановления")
    damaged = np.asarray(mask > 0, np.uint8) * 255
    if not np.any(damaged):
        return rgb.copy()
    return cv2.inpaint(rgb, damaged, float(radius), algorithms[method])
