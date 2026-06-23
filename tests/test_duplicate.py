from pathlib import Path

import cv2
import numpy as np
import pytest

from digital_hoarding_detector.duplicate import (
    difference_hash,
    find_duplicate_groups,
    hamming_distance,
)


def _sample_image(seed: int, size: int = 256) -> np.ndarray:
    random_generator = np.random.default_rng(seed)
    image = random_generator.integers(0, 256, (size, size, 3), dtype=np.uint8)
    return cv2.GaussianBlur(image, (7, 7), 0)


def test_difference_hash_is_stable_for_identical_images() -> None:
    image = _sample_image(seed=1)

    assert difference_hash(image) == difference_hash(image.copy())


def test_hamming_distance_counts_different_bits() -> None:
    assert hamming_distance(0b101010, 0b111000) == 2


def test_groups_exact_duplicates_and_ignores_unique_images() -> None:
    first_image = _sample_image(seed=1)
    exact_copy = first_image.copy()
    unique_image = _sample_image(seed=2)

    result = find_duplicate_groups(
        [first_image, exact_copy, unique_image],
        max_distance=0,
    )

    assert len(result.groups) == 1
    assert result.groups[0].image_indices == (0, 1)
    assert result.duplicate_count == 1
    assert result.image_count == 3


def test_detects_resized_near_duplicate() -> None:
    original_image = _sample_image(seed=3)
    resized_image = cv2.resize(original_image, (180, 180))

    result = find_duplicate_groups(
        [original_image, resized_image],
        max_distance=5,
    )

    assert result.groups[0].image_indices == (0, 1)


def test_accepts_image_paths(tmp_path: Path) -> None:
    image = _sample_image(seed=4)
    first_path = tmp_path / "first.jpg"
    second_path = tmp_path / "second.jpg"
    assert cv2.imwrite(str(first_path), image)
    assert cv2.imwrite(str(second_path), image)

    result = find_duplicate_groups([first_path, second_path], max_distance=2)

    assert result.duplicate_count == 1


def test_empty_collection_has_no_duplicate_groups() -> None:
    result = find_duplicate_groups([])

    assert result.groups == ()
    assert result.duplicate_count == 0


@pytest.mark.parametrize("max_distance", [-1, 65])
def test_rejects_invalid_hash_distance(max_distance: int) -> None:
    with pytest.raises(ValueError, match="between 0 and 64"):
        find_duplicate_groups([], max_distance=max_distance)


def test_rejects_invalid_hash_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        difference_hash(_sample_image(seed=5), hash_size=0)
