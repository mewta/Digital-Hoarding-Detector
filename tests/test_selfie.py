from pathlib import Path

import cv2
import numpy as np
import pytest

import digital_hoarding_detector.selfie as selfie_module
from digital_hoarding_detector.selfie import (
    FaceBox,
    detect_faces,
    extract_face_embedding,
    face_similarity,
    find_similar_selfies,
)


def _synthetic_face(seed: int, size: int = 160) -> np.ndarray:
    random_generator = np.random.default_rng(seed)
    grayscale = random_generator.integers(0, 256, (size, size), dtype=np.uint8)
    grayscale = cv2.GaussianBlur(grayscale, (11, 11), 0)
    return cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)


def test_embedding_is_stable_for_minor_image_change() -> None:
    image = _synthetic_face(seed=1)
    modified_image = cv2.convertScaleAbs(image, alpha=1.0, beta=8)
    face = FaceBox(0, 0, image.shape[1], image.shape[0])

    original_embedding = extract_face_embedding(image, face)
    modified_embedding = extract_face_embedding(modified_image, face)

    assert face_similarity(original_embedding, modified_embedding) > 0.95


def test_groups_images_with_similar_primary_faces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_image = _synthetic_face(seed=2)
    similar_image = cv2.GaussianBlur(first_image, (3, 3), 0)
    different_image = _synthetic_face(seed=9)

    def return_full_image_face(
        image: np.ndarray,
        minimum_face_ratio: float = 0.03,
    ) -> tuple[FaceBox, ...]:
        del minimum_face_ratio
        return (FaceBox(0, 0, image.shape[1], image.shape[0]),)

    monkeypatch.setattr(selfie_module, "detect_faces", return_full_image_face)

    result = find_similar_selfies(
        [first_image, similar_image, different_image],
        similarity_threshold=0.90,
    )

    assert result.groups[0].image_indices == (0, 1)
    assert result.cluster_count == 1
    assert result.face_counts == (1, 1, 1)


def test_image_without_detectable_face_is_not_grouped() -> None:
    blank_image = np.full((300, 300, 3), 255, dtype=np.uint8)

    result = find_similar_selfies([blank_image])

    assert result.images_with_faces == ()
    assert result.groups == ()
    assert result.face_counts == (0,)


def test_detect_faces_accepts_image_path(tmp_path: Path) -> None:
    image_path = tmp_path / "blank.png"
    assert cv2.imwrite(
        str(image_path),
        np.full((200, 200, 3), 255, dtype=np.uint8),
    )

    assert detect_faces(image_path) == ()


def test_rejects_face_box_outside_image() -> None:
    image = _synthetic_face(seed=3)

    with pytest.raises(ValueError, match="inside the image"):
        extract_face_embedding(image, FaceBox(100, 100, 100, 100))


@pytest.mark.parametrize("threshold", [-1.01, 1.01])
def test_rejects_invalid_similarity_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="between -1 and 1"):
        find_similar_selfies([], similarity_threshold=threshold)


@pytest.mark.parametrize("minimum_face_ratio", [0.0, 1.01])
def test_rejects_invalid_minimum_face_ratio(minimum_face_ratio: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        detect_faces(
            np.full((100, 100, 3), 255, dtype=np.uint8),
            minimum_face_ratio=minimum_face_ratio,
        )
