"""Face detection and similar-selfie grouping using OpenCV."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

ImageArray: TypeAlias = NDArray[np.uint8]
FaceEmbedding: TypeAlias = NDArray[np.float32]
ImageInput: TypeAlias = str | Path | ImageArray

DEFAULT_SIMILARITY_THRESHOLD = 0.88


@dataclass(frozen=True, slots=True)
class FaceBox:
    """Detected face location in pixel coordinates."""

    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass(frozen=True, slots=True)
class SelfieGroup:
    """Uploaded image indices containing visually similar primary faces."""

    image_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SelfieAnalysis:
    """Face detections and similar-selfie groups for an image collection."""

    groups: tuple[SelfieGroup, ...]
    images_with_faces: tuple[int, ...]
    face_counts: tuple[int, ...]

    @property
    def cluster_count(self) -> int:
        return len(self.groups)


def _load_color_image(image: ImageInput) -> ImageArray:
    """Load a path or validate an in-memory image as BGR."""
    if isinstance(image, (str, Path)):
        image_path = Path(image)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")

        loaded_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if loaded_image is None:
            raise ValueError(f"Unable to decode image file: {image_path}")
        return loaded_image

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a file path or a NumPy array")
    if image.size == 0:
        raise ValueError("image array must not be empty")
    if image.dtype != np.uint8:
        raise ValueError("image array must use uint8 pixel values")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("image array must be grayscale, BGR, or BGRA")
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


@lru_cache(maxsize=1)
def _face_detector() -> cv2.CascadeClassifier:
    """Load OpenCV's bundled frontal-face cascade once."""
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Unable to load face detector: {cascade_path}")
    return detector


def detect_faces(
    image: ImageInput,
    minimum_face_ratio: float = 0.03,
) -> tuple[FaceBox, ...]:
    """Detect frontal faces and return largest detections first."""
    if not 0 < minimum_face_ratio <= 1:
        raise ValueError("minimum_face_ratio must be between 0 and 1")

    color_image = _load_color_image(image)
    grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    grayscale_image = cv2.equalizeHist(grayscale_image)

    height, width = grayscale_image.shape
    minimum_face_side = max(20, int(min(height, width) * np.sqrt(minimum_face_ratio)))
    detections = _face_detector().detectMultiScale(
        grayscale_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(minimum_face_side, minimum_face_side),
    )

    faces = tuple(
        sorted(
            (
                FaceBox(int(x), int(y), int(face_width), int(face_height))
                for x, y, face_width, face_height in detections
            ),
            key=lambda face: face.area,
            reverse=True,
        )
    )
    return faces


def extract_face_embedding(image: ImageInput, face: FaceBox) -> FaceEmbedding:
    """Create a normalized low-frequency DCT embedding from a face crop."""
    color_image = _load_color_image(image)
    image_height, image_width = color_image.shape[:2]

    if face.width <= 0 or face.height <= 0:
        raise ValueError("face dimensions must be positive")
    if (
        face.x < 0
        or face.y < 0
        or face.x + face.width > image_width
        or face.y + face.height > image_height
    ):
        raise ValueError("face box must be inside the image")

    face_crop = color_image[
        face.y : face.y + face.height,
        face.x : face.x + face.width,
    ]
    grayscale_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    normalized_face = cv2.equalizeHist(
        cv2.resize(grayscale_face, (64, 64), interpolation=cv2.INTER_AREA)
    )

    dct_coefficients = cv2.dct(normalized_face.astype(np.float32) / 255.0)
    embedding = dct_coefficients[:16, :16].flatten()
    embedding = embedding[1:]
    embedding -= embedding.mean()

    norm = float(np.linalg.norm(embedding))
    if norm == 0:
        return np.zeros_like(embedding, dtype=np.float32)
    return (embedding / norm).astype(np.float32)


def face_similarity(
    first_embedding: FaceEmbedding,
    second_embedding: FaceEmbedding,
) -> float:
    """Return cosine similarity between equal-sized normalized embeddings."""
    if first_embedding.shape != second_embedding.shape:
        raise ValueError("face embeddings must have matching shapes")
    if first_embedding.ndim != 1:
        raise ValueError("face embeddings must be one-dimensional")

    first_norm = float(np.linalg.norm(first_embedding))
    second_norm = float(np.linalg.norm(second_embedding))
    if first_norm == 0 or second_norm == 0:
        return 0.0

    similarity = np.dot(first_embedding, second_embedding) / (
        first_norm * second_norm
    )
    return float(np.clip(similarity, -1.0, 1.0))


def find_similar_selfies(
    images: Sequence[ImageInput],
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    minimum_face_ratio: float = 0.03,
) -> SelfieAnalysis:
    """Group images whose largest detected faces have similar embeddings."""
    if not -1 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be between -1 and 1")

    loaded_images = [_load_color_image(image) for image in images]
    face_lists = [
        detect_faces(image, minimum_face_ratio=minimum_face_ratio)
        for image in loaded_images
    ]
    face_counts = tuple(len(faces) for faces in face_lists)
    images_with_faces = tuple(
        index for index, faces in enumerate(face_lists) if faces
    )

    embeddings = {
        image_index: extract_face_embedding(
            loaded_images[image_index],
            face_lists[image_index][0],
        )
        for image_index in images_with_faces
    }
    parents = {image_index: image_index for image_index in images_with_faces}

    def find(image_index: int) -> int:
        while parents[image_index] != image_index:
            parents[image_index] = parents[parents[image_index]]
            image_index = parents[image_index]
        return image_index

    def union(first_index: int, second_index: int) -> None:
        first_root = find(first_index)
        second_root = find(second_index)
        if first_root != second_root:
            parents[second_root] = first_root

    for position, first_index in enumerate(images_with_faces):
        for second_index in images_with_faces[position + 1 :]:
            if (
                face_similarity(embeddings[first_index], embeddings[second_index])
                >= similarity_threshold
            ):
                union(first_index, second_index)

    components: dict[int, list[int]] = {}
    for image_index in images_with_faces:
        components.setdefault(find(image_index), []).append(image_index)

    groups = tuple(
        SelfieGroup(tuple(indices))
        for indices in components.values()
        if len(indices) > 1
    )
    return SelfieAnalysis(
        groups=groups,
        images_with_faces=images_with_faces,
        face_counts=face_counts,
    )
