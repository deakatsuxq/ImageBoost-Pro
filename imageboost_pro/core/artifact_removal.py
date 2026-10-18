"""Conservative JPEG grid cleanup, flat-area debanding, Gaussian deconvolution."""

import cv2
import numpy as np
from ._image import validate_rgb, uint8


def remove_jpeg_artifacts(rgb, strength=50):
    rgb = validate_rgb(rgb)
    if strength <= 0:
        return rgb.copy()
    smooth = cv2.bilateralFilter(rgb, 5, 5 + strength * 0.45, 2)
    # JPEG block boundaries are every 8 pixels. Smooth a 2-pixel band while protecting
    # true edges with a bilateral filter; all other pixels remain untouched.
    height, width = rgb.shape[:2]
    yy = np.arange(height)[:, None]
    xx = np.arange(width)[None, :]
    grid = ((xx % 8 <= 1) | (xx % 8 == 7) | (yy % 8 <= 1) | (yy % 8 == 7))
    mix = grid[..., None].astype(np.float32) * strength / 100
    return uint8(rgb.astype(np.float32) + (smooth.astype(np.float32) - rgb) * mix)


def reduce_banding(rgb, strength=50):
    rgb = validate_rgb(rgb)
    if strength <= 0:
        return rgb.copy()
    image = rgb.astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    edges = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    flat = np.clip(1 - edges / 28, 0, 1)[..., None]
    smooth = cv2.GaussianBlur(image, (0, 0), 0.5 + strength / 70)
    return uint8(image + (smooth - image) * flat * strength / 100)


def deconvolve_image(rgb, strength=50):
    """Richardson–Lucy for an assumed mild Gaussian PSF; no unknown motion estimation."""
    rgb = validate_rgb(rgb)
    if strength <= 0:
        return rgb.copy()
    from skimage.restoration import richardson_lucy
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    light = lab[..., 0].astype(np.float32) / 255
    kernel = cv2.getGaussianKernel(5, 0.8 + strength / 150)
    psf = (kernel @ kernel.T).astype(np.float32)
    iterations = 5 + int(strength / 10)
    # Richardson–Lucy convolutions assume zero outside the input; reflect padding
    # keeps that artificial black border outside the actual image for every pass.
    padding = 2 * iterations + 2
    extended = np.pad(light, padding, mode="reflect")
    restored = richardson_lucy(extended, psf, num_iter=iterations, clip=True)
    restored = restored[padding:-padding, padding:-padding]
    # Suppress extreme ringing with a local source-derived envelope.
    lower = cv2.erode(light, np.ones((3, 3), np.uint8)) - 0.04
    upper = cv2.dilate(light, np.ones((3, 3), np.uint8)) + 0.04
    restored = np.clip(restored, lower, upper)
    lab[..., 0] = uint8((light + (restored - light) * strength / 100) * 255)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
