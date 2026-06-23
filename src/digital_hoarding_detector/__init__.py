"""Computer-vision utilities for detecting digital gallery clutter."""

from .blur import BlurAnalysis, analyze_blur, analyze_blur_batch
from .duplicate import (
    DuplicateAnalysis,
    DuplicateGroup,
    difference_hash,
    find_duplicate_groups,
    hamming_distance,
)
from .screenshot import (
    ScreenshotAnalysis,
    analyze_screenshot,
    analyze_screenshot_batch,
)

__all__ = [
    "BlurAnalysis",
    "DuplicateAnalysis",
    "DuplicateGroup",
    "ScreenshotAnalysis",
    "analyze_blur",
    "analyze_blur_batch",
    "analyze_screenshot",
    "analyze_screenshot_batch",
    "difference_hash",
    "find_duplicate_groups",
    "hamming_distance",
]
