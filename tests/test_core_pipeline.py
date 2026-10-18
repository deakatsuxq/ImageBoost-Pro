"""Algorithm quality invariants using known synthetic images, not snapshots."""

from dataclasses import asdict

import cv2
import numpy as np
import pytest

from imageboost_pro.core.settings import ProcessingSettings
from imageboost_pro.core.pipeline import output_size, process_image, ProcessingCancelled
from imageboost_pro.core.upscaling import resize_image
from imageboost_pro.core.denoising import denoise_image
from imageboost_pro.core.sharpening import sharpen_image
from imageboost_pro.core.color_correction import correct_color
from imageboost_pro.core.night_recovery import recover_night
from imageboost_pro.core.restoration import inpaint_mask
from imageboost_pro.core.artifact_removal import reduce_banding, remove_jpeg_artifacts, deconvolve_image


@pytest.fixture
def image():
    yy, xx = np.indices((64, 96))
    return np.stack((xx * 2 + 20, yy * 3 + 10, (xx + yy) + 30), axis=-1).astype(np.uint8)


def test_settings_roundtrip_and_frozen():
    value = ProcessingSettings(scale=1.5, night_mode="natural", denoise_strength=20)
    assert ProcessingSettings.from_dict(asdict(value)) == value
    assert value.validate() is None
    with pytest.raises(AttributeError):
        value.scale = 4


@pytest.mark.parametrize("bad", [
    {"unknown": 1}, {"gamma": 0}, {"exposure": 6}, {"scale": float("nan")},
    {"scale": float("inf")}, {"scale": True}, {"target_width": 2.5},
    {"denoise_strength": -1}, {"upscaler": "fake_ai"}, {"night_mode": []},
    {"ai_model_path": None}, {"keep_aspect": "false"},
])
def test_bad_settings_rejected(bad):
    with pytest.raises(ValueError):
        ProcessingSettings.from_dict(bad)


@pytest.mark.parametrize("scale", [1.5, 2, 3, 4, 8])
def test_scale_dimensions(image, scale):
    settings = ProcessingSettings(scale=scale)
    width, height = output_size(96, 64, settings)
    output = process_image(image, settings)
    assert (width, height) == (round(96 * scale), round(64 * scale))
    assert output.shape == (height, width, 3)


def test_custom_dimensions_and_guard():
    assert output_size(800, 600, ProcessingSettings(target_width=1200)) == (1200, 900)
    assert output_size(800, 600, ProcessingSettings(target_height=150)) == (200, 150)
    assert output_size(800, 600, ProcessingSettings(target_width=400, target_height=200)) == (267, 200)
    assert output_size(800, 600, ProcessingSettings(target_width=400, target_height=200,
                                                 keep_aspect=False)) == (400, 200)
    with pytest.raises(ValueError, match="лимит"):
        output_size(6000, 4000, ProcessingSettings(scale=8))


@pytest.mark.parametrize("method", ["nearest", "bilinear", "bicubic", "lanczos", "area"])
def test_interpolation_constants_and_rgb(method):
    color = np.full((5, 7, 3), [252, 48, 17], dtype=np.uint8)
    result = resize_image(color, 21, 15, method)
    assert result.shape == (15, 21, 3)
    assert np.max(np.abs(result.astype(int) - color[0, 0])) <= 1


def test_neutral_pipeline_identity_and_no_alias(image):
    original = image.copy()
    result = process_image(image, ProcessingSettings())
    np.testing.assert_array_equal(result, original)
    result[0, 0] = 0
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("bad", [np.zeros((3, 3), np.uint8), np.zeros((3, 3, 4), np.uint8),
                                  np.zeros((3, 3, 3), float), np.zeros((0, 3, 3), np.uint8)])
def test_input_rejected(bad):
    with pytest.raises(ValueError):
        process_image(bad, ProcessingSettings())


def test_cancellation_and_real_stage_progress(image):
    progress = []
    with pytest.raises(ProcessingCancelled):
        process_image(image, ProcessingSettings(exposure=1, scale=2),
                      lambda percent, stage: progress.append((percent, stage)),
                      lambda: any(percent >= 50 for percent, _ in progress))
    assert progress[-1][0] == 50
    with pytest.raises(ProcessingCancelled):
        process_image(image, ProcessingSettings(), cancelled=lambda: True)


@pytest.mark.parametrize("method", ["gaussian", "median", "bilateral", "nlm", "wavelet"])
def test_denoising_reduces_known_noise(method):
    rng = np.random.default_rng(123)
    reference = np.full((96, 96, 3), 115, dtype=np.uint8)
    noisy = np.clip(reference.astype(float) + rng.normal(0, 15, reference.shape), 0, 255).astype(np.uint8)
    original = noisy.copy()
    result = denoise_image(noisy, method=method, strength=80,
                           detail_preservation=0, edge_protection=0)
    before = np.mean((noisy.astype(float) - reference) ** 2)
    after = np.mean((result.astype(float) - reference) ** 2)
    assert after < before * 0.6, (method, before, after)
    np.testing.assert_array_equal(noisy, original)


@pytest.mark.parametrize("method", ["unsharp", "highpass", "laplacian", "adaptive"])
def test_sharpening_improves_known_soft_edge(method):
    image = np.full((64, 64, 3), 60, np.uint8)
    image[:, 32:] = 180
    soft = cv2.GaussianBlur(image, (0, 0), 1.4)
    result = sharpen_image(soft, method=method, amount=60, threshold=0,
                           halo_protection=100)
    gradient_before = np.diff(soft[32, :, 0].astype(float)).max()
    gradient_after = np.diff(result[32, :, 0].astype(float)).max()
    assert gradient_after > gradient_before
    assert result.min() >= 58 and result.max() <= 182


@pytest.mark.parametrize("mode", ["natural", "balanced", "maximum"])
def test_night_lifts_dark_scene_protects_highlights(mode):
    night = np.full((80, 80, 3), [22, 25, 30], np.uint8)
    night[25:40, 25:40] = 250
    original = night.copy()
    result = recover_night(night, mode=mode, strength=75, highlight_protection=100)
    assert result[:20].mean() > night[:20].mean() + 8
    assert np.abs(result[27:38, 27:38].astype(float) - 250).mean() < 5
    np.testing.assert_array_equal(night, original)


def test_night_black_information_and_neutral_gamma(image):
    black = np.zeros((16, 16, 3), dtype=np.uint8)
    np.testing.assert_array_equal(recover_night(black), black)
    np.testing.assert_array_equal(recover_night(image, mode="off"), image)
    np.testing.assert_array_equal(recover_night(image, strength=0), image)


def test_exposure_gamma_color_and_gray_balance():
    image = np.full((32, 32, 3), [60, 90, 120], np.uint8)
    exposed = correct_color(image, exposure=1)
    np.testing.assert_array_equal(exposed[0, 0], [120, 180, 240])
    gray = correct_color(image, white_balance=True)
    assert np.ptp(gray[0, 0].astype(int)) <= 2
    neutral = correct_color(image, saturation=-100)
    assert np.ptp(neutral[0, 0].astype(int)) == 0
    warm = correct_color(image, temperature=80)
    assert warm[..., 0].mean() > 60 and warm[..., 2].mean() < 120
    gamma = recover_night(image, mode="off", gamma=2)
    assert gamma.mean() > image.mean()


def test_damage_mask_repair_and_input_preserved():
    photo = np.full((40, 40, 3), [90, 120, 150], dtype=np.uint8)
    damaged = photo.copy()
    damaged[18:21, 5:35] = 255
    mask = np.zeros((40, 40), np.uint8)
    mask[18:21, 5:35] = 255
    repaired = inpaint_mask(damaged, mask)
    assert np.mean(np.abs(repaired.astype(float) - photo)) < np.mean(np.abs(damaged.astype(float) - photo))
    np.testing.assert_array_equal(repaired[mask == 0], damaged[mask == 0])
    assert np.all(damaged[18:21, 5:35] == 255)


@pytest.mark.parametrize("algorithm", [remove_jpeg_artifacts, reduce_banding, deconvolve_image])
def test_artifact_algorithms_valid_and_neutral(image, algorithm):
    np.testing.assert_array_equal(algorithm(image, strength=0), image)
    output = algorithm(image, strength=50)
    assert output.shape == image.shape and output.dtype == np.uint8


@pytest.mark.parametrize("shape", [(1, 1, 3), (1, 4, 3), (3, 2, 3)])
def test_tiny_image_all_filters(shape):
    image = np.full(shape, [60, 90, 120], np.uint8)
    for method in ("gaussian", "median", "bilateral", "nlm", "wavelet"):
        assert denoise_image(image, method=method).shape == shape
    assert process_image(image, ProcessingSettings(
        night_mode="balanced", denoise_strength=20, sharpen_amount=20,
        jpeg_artifacts=20, banding_reduction=20, deblur=20)).shape == shape


def test_ai_missing_model_explicit_error(image):
    with pytest.raises(ValueError, match="веса"):
        process_image(image, ProcessingSettings(upscaler="realesrgan"))


def test_all_stages_pipeline_rgb_shape_and_monotonic_progress(image):
    original = image.copy()
    progress = []
    output = process_image(image, ProcessingSettings(
        denoise_strength=25, gamma=1.3, exposure=0.2, sharpen_amount=35,
        scale=1.5, jpeg_artifacts=30, banding_reduction=20, deblur=10),
        lambda percent, stage: progress.append(percent))
    assert output.shape == (96, 144, 3) and output.dtype == np.uint8
    assert progress == sorted(progress) and progress[0] == 0 and progress[-1] == 100
    np.testing.assert_array_equal(image, original)
