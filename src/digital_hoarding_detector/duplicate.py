"""Duplicate and near-duplicate image detection using perceptual hashes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

ImageArray: TypeAlias = NDArray[np.uint8]
ImageInput: TypeAlias = str | Path | ImageArray

DEFAULT_HASH_SIZE = 8
DEFAULT_MAX_DISTANCE = 5


@dataclass(frozen=True, slots=True)
class DuplicateGroup:
    """Indices of uploaded images considered duplicates of one another."""

    image_indices: tuple[int, ...]

    @property
    def duplicate_count(self) -> int:
        """Number of removable images when the first image is retained."""
        return len(self.image_indices) - 1


@dataclass(frozen=True, slots=True)
class DuplicateAnalysis:
    """Duplicate groups found in an uploaded image collection."""

    groups: tuple[DuplicateGroup, ...]
    image_count: int

    @property
    def duplicate_count(self) -> int:
        """Total removable duplicate images across all groups."""
        return sum(group.duplicate_count for group in self.groups)


def _load_grayscale_image(image: ImageInput) -> ImageArray:
    """Load an image as grayscale from a path or validated NumPy array."""
    if isinstance(image, (str, Path)):
        image_path = Path(image)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")

        grayscale_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if grayscale_image is None:
            raise ValueError(f"Unable to decode image file: {image_path}")
        return grayscale_image

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a file path or a NumPy array")
    if image.size == 0:
        raise ValueError("image array must not be empty")
    if image.dtype != np.uint8:
        raise ValueError("image array must use uint8 pixel values")
    if image.ndim == 2:
        return image
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("image array must be grayscale, BGR, or BGRA")

    conversion = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(image, conversion)


def difference_hash(
    image: ImageInput,
    hash_size: int = DEFAULT_HASH_SIZE,
) -> int:
    """Create a perceptual difference hash that tolerates minor image changes."""
    if hash_size <= 0:
        raise ValueError("hash_size must be positive")

    grayscale_image = _load_grayscale_image(image)
    resized_image = cv2.resize(
        grayscale_image,
        (hash_size + 1, hash_size),
        interpolation=cv2.INTER_AREA,
    )
    difference_bits = resized_image[:, 1:] > resized_image[:, :-1]

    hash_value = 0
    for bit in difference_bits.flat:
        hash_value = (hash_value << 1) | int(bit)
    return hash_value


def hamming_distance(first_hash: int, second_hash: int) -> int:
    """Count differing bits between two non-negative integer hashes."""
    if first_hash < 0 or second_hash < 0:
        raise ValueError("hash values must be non-negative")
    return (first_hash ^ second_hash).bit_count()


def find_duplicate_groups(
    images: Sequence[ImageInput],
    max_distance: int = DEFAULT_MAX_DISTANCE,
    hash_size: int = DEFAULT_HASH_SIZE,
) -> DuplicateAnalysis:
    """Group exact and near-duplicate images by perceptual hash distance.

    Images form an undirected graph where an edge connects hashes within the
    configured Hamming distance. Connected components with at least two images
    are returned as duplicate groups.
    """
    maximum_hash_distance = hash_size * hash_size
    if max_distance < 0 or max_distance > maximum_hash_distance:
        raise ValueError(
            f"max_distance must be between 0 and {maximum_hash_distance}"
        )

    hashes = [difference_hash(image, hash_size) for image in images]
    parents = list(range(len(images)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first_index: int, second_index: int) -> None:
        first_root = find(first_index)
        second_root = find(second_index)
        if first_root != second_root:
            parents[second_root] = first_root

    for first_index, first_hash in enumerate(hashes):
        for second_index in range(first_index + 1, len(hashes)):
            if hamming_distance(first_hash, hashes[second_index]) <= max_distance:
                union(first_index, second_index)

    components: dict[int, list[int]] = {}
    for image_index in range(len(images)):
        components.setdefault(find(image_index), []).append(image_index)

    groups = tuple(
        DuplicateGroup(tuple(indices))
        for indices in components.values()
        if len(indices) > 1
    )
    return DuplicateAnalysis(groups=groups, image_count=len(images))
