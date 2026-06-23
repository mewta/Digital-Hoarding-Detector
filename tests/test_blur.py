from pathlib import Path

import cv2
import numpy as np
import pytest

from digital_hoarding_detector.blur import analyze_blur, analyze_blur_batch


def _checkerboard(size: int = 256, block_size: int = 8) -> np.ndarray:
    row_indices, column_indices = np.indices((size, size))
    pattern = ((row_indices // block_size + column_indices // block_size) % 2) * 255
    return pattern.astype(np.uint8)


def test_blurred_image_has_lower_variance_than_sharp_image() -> None:
    sharp_image = _checkerboard()
    blurred_image = cv2.GaussianBlur(sharp_image, (31, 31), 0)

    sharp_result = analyze_blur(sharp_image, threshold=100.0)
    blurred_result = analyze_blur(blurred_image, threshold=100.0)

    assert blurred_result.variance < sharp_result.variance
    assert sharp_result.is_blurry is False
    assert blurred_result.is_blurry is True


def test_accepts_image_file_path(tmp_path: Path) -> None:
    image_path = tmp_path / "sharp.png"
    assert cv2.imwrite(str(image_path), _checkerboard())

    result = analyze_blur(image_path, threshold=100.0)

    assert result.is_blurry is False


def test_batch_analysis_preserves_order() -> None:
    sharp_image = _checkerboard()
    blurred_image = cv2.GaussianBlur(sharp_image, (31, 31), 0)

    results = analyze_blur_batch([sharp_image, blurred_image], threshold=100.0)

    assert [result.is_blurry for result in results] == [False, True]


@pytest.mark.parametrize("threshold", [-1.0, -0.01])
def test_rejects_negative_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        analyze_blur(_checkerboard(), threshold=threshold)


def test_rejects_missing_image_file() -> None:
    with pytest.raises(FileNotFoundError):
        analyze_blur("missing-image.jpg")


def test_rejects_empty_image_array() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        analyze_blur(np.array([], dtype=np.uint8))
