"""Rule-based screenshot detection using lightweight OpenCV heuristics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

ImageArray: TypeAlias = NDArray[np.uint8]
ImageInput: TypeAlias = str | Path | ImageArray

DEFAULT_SCREENSHOT_THRESHOLD = 0.55


@dataclass(frozen=True, slots=True)
class ScreenshotAnalysis:
    """Signals and classification produced for one image."""

    score: float
    text_region_count: int
    text_coverage: float
    edge_density: float
    flat_region_ratio: float
    is_screenshot: bool


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


def _text_layout_signals(grayscale_image: ImageArray) -> tuple[int, float]:
    """Estimate text-like regions from connected components."""
    height, width = grayscale_image.shape
    minimum_dimension = max(3, min(height, width))
    block_size = max(3, (minimum_dimension // 16) | 1)

    binary_image = cv2.adaptiveThreshold(
        grayscale_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        7,
    )
    horizontal_kernel_width = max(2, width // 160)
    connected_image = cv2.morphologyEx(
        binary_image,
        cv2.MORPH_CLOSE,
        np.ones((1, horizontal_kernel_width), dtype=np.uint8),
    )
    contours, _ = cv2.findContours(
        connected_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    text_region_count = 0
    text_region_area = 0
    image_area = height * width

    for contour in contours:
        x, y, region_width, region_height = cv2.boundingRect(contour)
        del x, y
        relative_width = region_width / width
        relative_height = region_height / height
        region_area = region_width * region_height

        if (
            0.004 <= relative_width <= 0.85
            and 0.004 <= relative_height <= 0.12
            and region_area >= 6
        ):
            text_region_count += 1
            text_region_area += region_area

    return text_region_count, text_region_area / image_area


def analyze_screenshot(
    image: ImageInput,
    threshold: float = DEFAULT_SCREENSHOT_THRESHOLD,
) -> ScreenshotAnalysis:
    """Classify an image from text layout, edges, and flat-color regions.

    The detector is intended as a fast prototype heuristic. It works best for
    UI, chat, document, and browser screenshots. A trained classifier can later
    replace it without changing the application-facing result structure.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    color_image = _load_color_image(image)
    grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    image_area = grayscale_image.size

    text_region_count, text_coverage = _text_layout_signals(grayscale_image)

    edges = cv2.Canny(grayscale_image, 80, 180)
    edge_density = float(np.count_nonzero(edges) / image_area)

    horizontal_gradient = cv2.Sobel(
        grayscale_image,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )
    vertical_gradient = cv2.Sobel(
        grayscale_image,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )
    gradient_magnitude = cv2.magnitude(horizontal_gradient, vertical_gradient)
    flat_region_ratio = float(np.mean(gradient_magnitude < 20))

    text_count_score = min(text_region_count / 20, 1.0)
    text_coverage_score = min(text_coverage / 0.12, 1.0)
    edge_score = min(edge_density / 0.12, 1.0)
    flatness_score = min(flat_region_ratio / 0.80, 1.0)

    score = (
        0.35 * text_count_score
        + 0.25 * text_coverage_score
        + 0.15 * edge_score
        + 0.25 * flatness_score
    )
    score = float(min(max(score, 0.0), 1.0))

    return ScreenshotAnalysis(
        score=score,
        text_region_count=text_region_count,
        text_coverage=text_coverage,
        edge_density=edge_density,
        flat_region_ratio=flat_region_ratio,
        is_screenshot=score >= threshold,
    )


def analyze_screenshot_batch(
    images: Sequence[ImageInput],
    threshold: float = DEFAULT_SCREENSHOT_THRESHOLD,
) -> list[ScreenshotAnalysis]:
    """Analyze multiple images while preserving upload order."""
    return [analyze_screenshot(image, threshold) for image in images]
