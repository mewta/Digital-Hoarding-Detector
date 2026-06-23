from pathlib import Path

import cv2
import numpy as np
import pytest

from digital_hoarding_detector.screenshot import (
    analyze_screenshot,
    analyze_screenshot_batch,
)


def _mock_mobile_screenshot() -> np.ndarray:
    image = np.full((800, 400, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (399, 55), (35, 35, 35), -1)
    cv2.rectangle(image, (20, 80), (380, 150), (225, 235, 245), -1)

    for row in range(12):
        top = 180 + row * 42
        cv2.rectangle(image, (25, top), (270, top + 8), (40, 40, 40), -1)
        cv2.rectangle(image, (25, top + 15), (350, top + 22), (90, 90, 90), -1)

    return image


def _mock_natural_photo(seed: int = 7) -> np.ndarray:
    random_generator = np.random.default_rng(seed)
    image = random_generator.integers(0, 256, (800, 400, 3), dtype=np.uint8)
    return cv2.GaussianBlur(image, (21, 21), 0)


def test_detects_text_heavy_mobile_screenshot() -> None:
    result = analyze_screenshot(_mock_mobile_screenshot())

    assert result.is_screenshot is True
    assert result.text_region_count >= 10
    assert result.score >= 0.55


def test_does_not_flag_natural_photo_texture() -> None:
    result = analyze_screenshot(_mock_natural_photo())

    assert result.is_screenshot is False
    assert result.score < 0.55


def test_accepts_image_path(tmp_path: Path) -> None:
    image_path = tmp_path / "screenshot.png"
    assert cv2.imwrite(str(image_path), _mock_mobile_screenshot())

    result = analyze_screenshot(image_path)

    assert result.is_screenshot is True


def test_batch_analysis_preserves_order() -> None:
    results = analyze_screenshot_batch(
        [_mock_mobile_screenshot(), _mock_natural_photo()]
    )

    assert [result.is_screenshot for result in results] == [True, False]


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_rejects_threshold_outside_unit_interval(threshold: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        analyze_screenshot(_mock_mobile_screenshot(), threshold=threshold)


def test_rejects_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        analyze_screenshot("missing-screenshot.png")
