"""Blur detection based on the variance of the Laplacian."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

ImageArray: TypeAlias = NDArray[np.uint8]
ImageInput: TypeAlias = str | Path | ImageArray

DEFAULT_BLUR_THRESHOLD = 100.0


@dataclass(frozen=True, slots=True)
class BlurAnalysis:
    """Result of analyzing one image for blur."""

    variance: float
    threshold: float
    is_blurry: bool


def _load_image(image: ImageInput) -> ImageArray:
    """Load an image input and validate its shape and type."""
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
    if image.ndim not in (2, 3):
        raise ValueError("image array must be grayscale, BGR, or BGRA")
    if image.ndim == 3 and image.shape[2] not in (3, 4):
        raise ValueError("color image array must have 3 or 4 channels")

    return image


def _to_grayscale(image: ImageArray) -> ImageArray:
    """Convert a validated image to grayscale."""
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def analyze_blur(
    image: ImageInput,
    threshold: float = DEFAULT_BLUR_THRESHOLD,
) -> BlurAnalysis:
    """Measure image sharpness and classify it using a Laplacian threshold.

    A lower variance indicates fewer high-frequency edges and therefore a
    blurrier image. The threshold is dataset-dependent and should be tuned
    against representative phone photos.
    """
    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    grayscale_image = _to_grayscale(_load_image(image))
    laplacian_variance = float(cv2.Laplacian(grayscale_image, cv2.CV_64F).var())

    return BlurAnalysis(
        variance=laplacian_variance,
        threshold=float(threshold),
        is_blurry=laplacian_variance < threshold,
    )


def analyze_blur_batch(
    images: list[ImageInput],
    threshold: float = DEFAULT_BLUR_THRESHOLD,
) -> list[BlurAnalysis]:
    """Analyze multiple images while preserving their input order."""
    return [analyze_blur(image, threshold) for image in images]
