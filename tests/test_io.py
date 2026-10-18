from pathlib import Path

import numpy as np
from PIL import Image, ImageCms
import pytest

from imageboost_pro.utils.image_io import (
    ImageIOError, load_image, save_image, unique_output_path,
)


@pytest.fixture
def rgba_document(tmp_path):
    y, x = np.mgrid[:24, :32]
    rgba = np.stack((x * 7, y * 9, np.full_like(x, 25), x * 8), axis=2).astype(np.uint8)
    path = tmp_path / "Снимок с прозрачностью.png"
    Image.fromarray(rgba).save(path)
    return load_image(path)


@pytest.mark.parametrize("extension", ["png", "tiff", "webp", "jpg", "jpeg", "bmp"])
def test_export_supported_formats_preserves_size_and_rgb_order(tmp_path, rgba_document, extension):
    # The source has much more red/green than blue; catches BGR/RGB swaps.
    rgb = np.repeat(np.repeat(rgba_document.rgb, 2, axis=0), 2, axis=1)
    path = tmp_path / f"Экспорт.{extension}"
    save_image(rgba_document, rgb, path, quality=100)
    reloaded = load_image(path)
    assert reloaded.size == (64, 48)
    if extension in {"png", "tiff"}:
        np.testing.assert_array_equal(reloaded.rgb, rgb)
        assert reloaded.alpha.shape == (48, 64)
        assert reloaded.alpha[:, 0].max() < 5
        assert reloaded.rgb[:, -1, 0].mean() > reloaded.rgb[:, -1, 2].mean()
    elif extension in {"jpg", "jpeg", "bmp"}:
        assert reloaded.alpha is None
        assert reloaded.rgb[:, 0].mean() > 245  # explicit white matte


def test_no_overwrite_even_for_original_and_cleanup(tmp_path, rgba_document):
    before = rgba_document.path.read_bytes()
    with pytest.raises(FileExistsError):
        save_image(rgba_document, np.zeros_like(rgba_document.rgb), rgba_document.path)
    assert rgba_document.path.read_bytes() == before
    dest = tmp_path / "out.png"
    save_image(rgba_document, rgba_document.rgb, dest)
    with pytest.raises(FileExistsError):
        save_image(rgba_document, rgba_document.rgb, dest)
    save_image(rgba_document, np.zeros_like(rgba_document.rgb), dest, overwrite=True)
    assert load_image(dest).rgb.max() == 0
    assert list(tmp_path.glob(".imageboost-*")) == []


def test_exif_orientation_is_applied_once_and_export_dimensions_updated(tmp_path):
    rgb = np.zeros((10, 20, 3), np.uint8)
    rgb[:, :10, 0] = 255
    exif = Image.Exif()
    exif[274] = 6
    exif[315] = "Test author"
    source = tmp_path / "rotated.jpg"
    Image.fromarray(rgb).save(source, exif=exif)
    doc = load_image(source)
    assert doc.size == (10, 20)
    assert doc.rgb[:10, :, 0].mean() > doc.rgb[10:, :, 0].mean()
    output = tmp_path / "oriented.png"
    save_image(doc, np.repeat(doc.rgb, 2, axis=0), output)
    with Image.open(output) as result:
        assert result.size == (10, 40)
        assert result.getexif().get(274, 1) == 1
        assert result.getexif()[40962] == 10
        assert result.getexif()[40963] == 40
        assert result.getexif()[315] == "Test author"
    save_image(doc, doc.rgb, tmp_path / "private.png", keep_metadata=False)
    with Image.open(tmp_path / "private.png") as image:
        assert not image.getexif()


def test_icc_srgb_conversion_and_profile_export(tmp_path):
    source = tmp_path / "profile.png"
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    Image.new("RGB", (12, 8), (220, 80, 15)).save(source, icc_profile=profile)
    document = load_image(source)
    assert document.icc_profile
    save_image(document, document.rgb, tmp_path / "result.png")
    with Image.open(tmp_path / "result.png") as result:
        assert result.info["icc_profile"]
        assert result.getpixel((0, 0)) == (220, 80, 15)


def test_grayscale_and_16bit_are_not_clipped_white(tmp_path):
    sixteen = np.array([[0, 32768, 65535]], np.uint16)
    Image.fromarray(sixteen).save(tmp_path / "16.png")
    document = load_image(tmp_path / "16.png")
    np.testing.assert_array_equal(document.rgb[0, :, 0], [0, 128, 255])
    assert document.warnings
    Image.new("L", (1, 1), 123).save(tmp_path / "gray.png")
    np.testing.assert_array_equal(load_image(tmp_path / "gray.png").rgb, [[[123, 123, 123]]])


def test_invalid_input_and_export_errors_are_explained(tmp_path, rgba_document):
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    with pytest.raises(ImageIOError, match="Не удалось открыть"):
        load_image(broken)
    with pytest.raises(ImageIOError, match="Выберите"):
        save_image(rgba_document, rgba_document.rgb, tmp_path / "result.exe")
    with pytest.raises(ImageIOError, match="Качество"):
        save_image(rgba_document, rgba_document.rgb, tmp_path / "result.png", quality=101)
    with pytest.raises(ImageIOError, match="8 бит"):
        save_image(rgba_document, rgba_document.rgb.astype(float), tmp_path / "result.png")


def test_queue_output_names_do_not_collide(tmp_path):
    first = unique_output_path(tmp_path, Path("C:/фото.png"))
    first.touch()
    second = unique_output_path(tmp_path, Path("C:/фото.jpg"))
    assert first.name == "фото_улучшено.png"
    assert second.name == "фото_улучшено_2.png"
