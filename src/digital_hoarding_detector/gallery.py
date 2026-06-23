"""Aggregate detector results into a gallery cleanup report."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .blur import BlurAnalysis, analyze_blur_batch
from .duplicate import DuplicateAnalysis, find_duplicate_groups
from .screenshot import ScreenshotAnalysis, analyze_screenshot_batch
from .selfie import SelfieAnalysis, find_similar_selfies

ImageArray = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class GalleryImage:
    """Decoded uploaded image and its source metadata."""

    name: str
    image: ImageArray
    size_bytes: int


@dataclass(frozen=True, slots=True)
class GalleryReport:
    """Combined cleanup analysis for one uploaded gallery."""

    images: tuple[GalleryImage, ...]
    blur_results: tuple[BlurAnalysis, ...]
    screenshot_results: tuple[ScreenshotAnalysis, ...]
    duplicate_analysis: DuplicateAnalysis
    selfie_analysis: SelfieAnalysis
    cleanup_candidate_indices: tuple[int, ...]
    potential_savings_bytes: int

    @property
    def blurry_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, result in enumerate(self.blur_results)
            if result.is_blurry
        )

    @property
    def screenshot_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, result in enumerate(self.screenshot_results)
            if result.is_screenshot
        )


def analyze_gallery(
    images: Sequence[GalleryImage],
    *,
    blur_threshold: float = 100.0,
    duplicate_max_distance: int = 5,
    screenshot_threshold: float = 0.55,
    selfie_similarity_threshold: float = 0.88,
) -> GalleryReport:
    """Run every detector and produce a deduplicated cleanup estimate."""
    image_arrays = [gallery_image.image for gallery_image in images]

    blur_results = tuple(analyze_blur_batch(image_arrays, blur_threshold))
    screenshot_results = tuple(
        analyze_screenshot_batch(image_arrays, screenshot_threshold)
    )
    duplicate_analysis = find_duplicate_groups(
        image_arrays,
        max_distance=duplicate_max_distance,
    )
    selfie_analysis = find_similar_selfies(
        image_arrays,
        similarity_threshold=selfie_similarity_threshold,
    )

    cleanup_candidates: set[int] = {
        index
        for index, result in enumerate(blur_results)
        if result.is_blurry
    }
    cleanup_candidates.update(
        index
        for index, result in enumerate(screenshot_results)
        if result.is_screenshot
    )
    for group in duplicate_analysis.groups:
        cleanup_candidates.update(group.image_indices[1:])
    for group in selfie_analysis.groups:
        cleanup_candidates.update(group.image_indices[1:])

    cleanup_candidate_indices = tuple(sorted(cleanup_candidates))
    potential_savings_bytes = sum(
        images[index].size_bytes for index in cleanup_candidate_indices
    )

    return GalleryReport(
        images=tuple(images),
        blur_results=blur_results,
        screenshot_results=screenshot_results,
        duplicate_analysis=duplicate_analysis,
        selfie_analysis=selfie_analysis,
        cleanup_candidate_indices=cleanup_candidate_indices,
        potential_savings_bytes=potential_savings_bytes,
    )


def format_storage_size(size_bytes: int) -> str:
    """Format a non-negative byte count using binary storage units."""
    if size_bytes < 0:
        raise ValueError("size_bytes must be non-negative")

    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.2f} {unit}"
        size /= 1024

    raise AssertionError("unreachable")
