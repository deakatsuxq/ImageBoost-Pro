"""Colour-correct, orientation-aware input and safe, metadata-aware output.

The processing convention is always RGB uint8. Alpha never passes through the
colour algorithms, and is resized independently when the result is enlarged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import errno
import io
import os
from pathlib import Path
import shutil
import tempfile
import warnings

import numpy as np
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError

MAX_INPUT_PIXELS = 64_000_000
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
FORMATS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP",
           ".bmp": "BMP", ".tif": "TIFF", ".tiff": "TIFF"}


class ImageIOError(ValueError):
    """An image cannot be read or safely exported (message is user-facing)."""


@dataclass
class ImageDocument:
    rgb: np.ndarray
    alpha: np.ndarray | None = None
    path: Path = field(default_factory=Path)
    format: str = "PNG"
    icc_profile: bytes | None = None
    exif: bytes | None = None
    warnings: tuple[str, ...] = ()

    @property
    def width(self) -> int:
        return self.rgb.shape[1]

    @property
    def height(self) -> int:
        return self.rgb.shape[0]

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height


def _rgb_array(value: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8:
        raise ImageIOError("Ожидается изображение RGB с глубиной 8 бит на канал.")
    if value.ndim != 3 or value.shape[2] != 3 or min(value.shape[:2]) < 1:
        raise ImageIOError("Некорректные размеры изображения RGB.")
    if value.shape[0] * value.shape[1] > MAX_INPUT_PIXELS:
        raise ImageIOError("Изображение превышает безопасный предел 64 мегапикселя.")
    return np.ascontiguousarray(value)


def load_image(path: str | Path) -> ImageDocument:
    path = Path(path)
    messages: list[str] = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as source:
                if source.width * source.height > MAX_INPUT_PIXELS:
                    raise ImageIOError("Файл превышает безопасный предел 64 мегапикселя.")
                image_format = source.format or path.suffix.lstrip(".").upper()
                original_profile = source.info.get("icc_profile")
                image = ImageOps.exif_transpose(source)
                image.load()
                exif = image.getexif()
                exif.pop(274, None)  # Orientation is baked into pixels exactly once.
                exif_bytes = exif.tobytes() if exif else None
                # Pillow's convert('L') clips 16-bit grayscale. Scale its full range instead.
                if image.mode.startswith("I;16") or image.mode == "I":
                    high = np.asarray(image, dtype=np.float64)
                    image = Image.fromarray(np.clip(high / 257.0, 0, 255).round().astype(np.uint8))
                    messages.append("16-битное изображение преобразовано в 8 бит на канал.")
                if image.mode == "F":
                    raise ImageIOError("HDR с плавающей точкой не поддерживается; преобразуйте файл в TIFF/PNG 8 бит.")
                rgba = image.convert("RGBA")
                alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8).copy()
                alpha = alpha if np.any(alpha != 255) else None
                rgb_image = image if image.mode == "CMYK" else rgba.convert("RGB")
                profile = None
                if original_profile:
                    try:
                        srgb = ImageCms.createProfile("sRGB")
                        rgb_image = ImageCms.profileToProfile(
                            rgb_image, ImageCms.ImageCmsProfile(io.BytesIO(original_profile)),
                            srgb, outputMode="RGB",
                        )
                        profile = ImageCms.ImageCmsProfile(srgb).tobytes()
                    except (ImageCms.PyCMSError, OSError, ValueError):
                        messages.append("Цветовой профиль повреждён или не поддерживается; использовано преобразование RGB.")
                rgb = np.array(rgb_image.convert("RGB"), dtype=np.uint8)
                return ImageDocument(_rgb_array(rgb), alpha, path.resolve(), image_format,
                                     profile, exif_bytes, tuple(messages))
    except ImageIOError:
        raise
    except (OSError, UnidentifiedImageError, SyntaxError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ImageIOError(f"Не удалось открыть «{path.name}»: {exc}") from exc


def _commit_without_overwrite(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
    except OSError as exc:
        if exc.errno not in {errno.EPERM, errno.EOPNOTSUPP, errno.EXDEV, errno.ENOSYS}:
            raise
        # FAT/network filesystems may not offer hardlinks. Exclusive open still
        # protects a file created after the initial destination existence check.
        created = False
        try:
            with destination.open("xb") as output, temporary.open("rb") as source:
                created = True
                shutil.copyfileobj(source, output)
        except BaseException:
            if created:
                destination.unlink(missing_ok=True)
            raise


def save_image(document: ImageDocument, rgb: np.ndarray, path: str | Path,
               quality: int = 95, keep_metadata: bool = True,
               overwrite: bool = False) -> Path:
    path = Path(path).resolve()
    rgb = _rgb_array(rgb)
    image_format = FORMATS.get(path.suffix.lower())
    if image_format is None:
        raise ImageIOError("Выберите JPG, PNG, WebP, BMP или TIFF.")
    if isinstance(quality, bool) or not isinstance(quality, int) or not 1 <= quality <= 100:
        raise ImageIOError("Качество должно быть целым числом от 1 до 100.")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Файл «{path.name}» уже существует. Подтвердите перезапись.")
    image = Image.fromarray(rgb)
    if document.alpha is not None:
        alpha = document.alpha
        if alpha.dtype != np.uint8 or alpha.shape != document.rgb.shape[:2]:
            raise ImageIOError("Некорректный альфа-канал исходного изображения.")
        alpha_image = Image.fromarray(alpha).resize(image.size, Image.Resampling.LANCZOS)
        if image_format in {"PNG", "WEBP", "TIFF"}:
            image.putalpha(alpha_image)
        else:
            background = Image.new("RGB", image.size, "white")
            background.paste(image, mask=alpha_image)
            image = background
    options: dict = {}
    if image_format in {"JPEG", "WEBP"}:
        options["quality"] = quality
    if image_format == "JPEG":
        options.update(subsampling=0, optimize=True)
    if image_format == "TIFF":
        options["compression"] = "tiff_deflate"
    if keep_metadata and image_format in {"JPEG", "PNG", "TIFF", "WEBP"}:
        if document.icc_profile:
            options["icc_profile"] = document.icc_profile
        if document.exif:
            exif = Image.Exif()
            exif.load(document.exif)
            exif.pop(274, None)
            # A processed file must never advertise the old pixel dimensions.
            exif[40962], exif[40963] = image.size
            options["exif"] = exif.tobytes()
    temporary = None
    try:
        fd, temporary_name = tempfile.mkstemp(prefix=".imageboost-", suffix=path.suffix, dir=path.parent)
        os.close(fd)
        temporary = Path(temporary_name)
        image.save(temporary, format=image_format, **options)
        if overwrite:
            os.replace(temporary, path)
        else:
            _commit_without_overwrite(temporary, path)
        return path
    except FileExistsError:
        raise
    except OSError as exc:
        raise ImageIOError(f"Не удалось сохранить «{path.name}»: {exc}") from exc
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def unique_output_path(folder: str | Path, source: str | Path, extension: str = ".png") -> Path:
    """Choose a readable candidate; save_image also checks races with other writers."""
    folder, source = Path(folder), Path(source)
    extension = extension if extension.startswith(".") else "." + extension
    candidate = folder / f"{source.stem}_улучшено{extension}"
    index = 2
    while candidate.exists():
        candidate = folder / f"{source.stem}_улучшено_{index}{extension}"
        index += 1
    return candidate
