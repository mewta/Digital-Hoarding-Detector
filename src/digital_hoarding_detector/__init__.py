"""Computer-vision utilities for detecting digital gallery clutter."""

from .blur import BlurAnalysis, analyze_blur, analyze_blur_batch
from .duplicate import (
    DuplicateAnalysis,
    DuplicateGroup,
    difference_hash,
    find_duplicate_groups,
    hamming_distance,
)
from .gallery import GalleryImage, GalleryReport, analyze_gallery, format_storage_size
from .screenshot import (
    ScreenshotAnalysis,
    analyze_screenshot,
    analyze_screenshot_batch,
)
from .selfie import (
    FaceBox,
    SelfieAnalysis,
    SelfieGroup,
    detect_faces,
    extract_face_embedding,
    face_similarity,
    find_similar_selfies,
)

__all__ = [
    "BlurAnalysis",
    "DuplicateAnalysis",
    "DuplicateGroup",
    "FaceBox",
    "GalleryImage",
    "GalleryReport",
    "ScreenshotAnalysis",
    "SelfieAnalysis",
    "SelfieGroup",
    "analyze_blur",
    "analyze_blur_batch",
    "analyze_gallery",
    "analyze_screenshot",
    "analyze_screenshot_batch",
    "difference_hash",
    "detect_faces",
    "extract_face_embedding",
    "face_similarity",
    "find_duplicate_groups",
    "find_similar_selfies",
    "format_storage_size",
    "hamming_distance",
]
