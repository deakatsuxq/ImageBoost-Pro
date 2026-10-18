"""Quality measurements use deterministic synthetic inputs with known defects."""

import cv2
import numpy as np
import pytest

from imageboost_pro.core.image_analysis import analyze_image


def _rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(gray[..., None], 3, axis=2)


def test_noise_estimate_distinguishes_noisy_flat_image():
    clean = np.full((256, 256, 3), 128, dtype=np.uint8)
    rng = np.random.default_rng(251)
    # Shared noise in RGB gives a known luminance noise standard deviation.
    noisy = _rgb(np.clip(128 + rng.normal(0, 12, (256, 256)), 0, 255).astype(np.uint8))
    assert analyze_image(clean).metrics["noise_sigma"] == 0
    sigma = analyze_image(noisy).metrics["noise_sigma"]
    assert 9 < sigma < 15


def test_noise_estimate_does_not_mistake_isolated_edges_for_noise():
    step = np.zeros((256, 256), dtype=np.uint8)
    step[:, 128:] = 255
    assert analyze_image(_rgb(step)).metrics["noise_sigma"] < 0.1


def test_sharpness_decreases_with_blur():
    y, x = np.indices((256, 256))
    pattern = _rgb((((x // 16 + y // 16) % 2) * 190 + 30).astype(np.uint8))
    blurred = cv2.GaussianBlur(pattern, (0, 0), 3)
    original_score = analyze_image(pattern).metrics["sharpness"]
    assert analyze_image(blurred).metrics["sharpness"] < original_score / 10


def test_dark_image_recommends_night_processing_without_modifying_input():
    image = np.full((64, 96, 3), 20, dtype=np.uint8)
    before = image.copy()
    report = analyze_image(image)
    assert report.metrics["brightness"] == 20
    assert "ночн" in " ".join(report.recommendations).lower()
    assert report.suggest_settings().night_mode == "balanced"
    assert report.suggest_settings("natural").night_strength < report.suggest_settings("maximum").night_strength
    np.testing.assert_array_equal(image, before)


def test_clipping_fractions_and_russian_report():
    image = np.zeros((32, 64, 3), dtype=np.uint8)
    image[:, 32:] = 255
    report = analyze_image(image)
    assert report.metrics["clipped_shadows"] == 0.5
    assert report.metrics["clipped_highlights"] == 0.5
    assert report.metrics["entropy"] == 1
    assert "Анализ качества фотографии" in report.to_text()
    assert "50.0%" in report.to_text()


def test_block_grid_is_measured_at_eight_pixel_boundaries():
    y, x = np.indices((128, 128))
    blocks = _rgb((((x // 8 + y // 8) % 2) * 30 + 110).astype(np.uint8))
    assert analyze_image(blocks).metrics["jpeg_grid_strength"] > 20
    assert analyze_image(np.full_like(blocks, 125)).metrics["jpeg_grid_strength"] == 0


def test_large_image_reports_original_resolution_and_bounded_analysis_scale():
    report = analyze_image(np.full((100, 3200, 3), 100, dtype=np.uint8))
    assert report.metrics["width"] == 3200
    assert report.metrics["height"] == 100
    assert report.metrics["analysis_scale"] == 0.5


def test_single_pixel_has_finite_measurements():
    report = analyze_image(np.array([[[0, 0, 0]]], dtype=np.uint8))
    assert all(np.isfinite(value) for value in report.metrics.values())
    assert report.metrics["noise_sigma"] == 0


@pytest.mark.parametrize(
    "invalid",
    [np.zeros((2, 2)), np.zeros((2, 2, 4), dtype=np.uint8), np.zeros((2, 2, 3)), np.zeros((0, 2, 3), dtype=np.uint8)],
)
def test_rejects_invalid_arrays(invalid):
    with pytest.raises(ValueError):
        analyze_image(invalid)


def test_rejects_unknown_suggestion_mode():
    report = analyze_image(np.full((20, 20, 3), 100, dtype=np.uint8))
    with pytest.raises(ValueError):
        report.suggest_settings("unknown")
