import numpy as np
import pytest

import digital_hoarding_detector.gallery as gallery_module
from digital_hoarding_detector.blur import BlurAnalysis
from digital_hoarding_detector.duplicate import DuplicateAnalysis, DuplicateGroup
from digital_hoarding_detector.gallery import (
    GalleryImage,
    analyze_gallery,
    format_storage_size,
)
from digital_hoarding_detector.screenshot import ScreenshotAnalysis
from digital_hoarding_detector.selfie import SelfieAnalysis, SelfieGroup


def _gallery_images() -> list[GalleryImage]:
    return [
        GalleryImage(
            name=f"image-{index}.jpg",
            image=np.full((20, 20, 3), index, dtype=np.uint8),
            size_bytes=size,
        )
        for index, size in enumerate((100, 200, 300, 400))
    ]


def test_aggregates_cleanup_candidates_without_double_counting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gallery_module,
        "analyze_blur_batch",
        lambda images, threshold: [
            BlurAnalysis(200, threshold, False),
            BlurAnalysis(20, threshold, True),
            BlurAnalysis(200, threshold, False),
            BlurAnalysis(200, threshold, False),
        ],
    )
    monkeypatch.setattr(
        gallery_module,
        "analyze_screenshot_batch",
        lambda images, threshold: [
            ScreenshotAnalysis(0.1, 0, 0, 0, 0, False),
            ScreenshotAnalysis(0.8, 10, 0.1, 0.1, 0.8, True),
            ScreenshotAnalysis(0.1, 0, 0, 0, 0, False),
            ScreenshotAnalysis(0.1, 0, 0, 0, 0, False),
        ],
    )
    monkeypatch.setattr(
        gallery_module,
        "find_duplicate_groups",
        lambda images, max_distance: DuplicateAnalysis(
            groups=(DuplicateGroup((0, 1)),),
            image_count=4,
        ),
    )
    monkeypatch.setattr(
        gallery_module,
        "find_similar_selfies",
        lambda images, similarity_threshold: SelfieAnalysis(
            groups=(SelfieGroup((2, 3)),),
            images_with_faces=(2, 3),
            face_counts=(0, 0, 1, 1),
        ),
    )

    report = analyze_gallery(_gallery_images())

    assert report.cleanup_candidate_indices == (1, 3)
    assert report.potential_savings_bytes == 600
    assert report.blurry_indices == (1,)
    assert report.screenshot_indices == (1,)


@pytest.mark.parametrize(
    ("size_bytes", "formatted"),
    [
        (0, "0 B"),
        (1023, "1023 B"),
        (1024, "1.00 KB"),
        (1024 * 1024, "1.00 MB"),
        (1024 * 1024 * 1024, "1.00 GB"),
    ],
)
def test_formats_storage_size(size_bytes: int, formatted: str) -> None:
    assert format_storage_size(size_bytes) == formatted


def test_rejects_negative_storage_size() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        format_storage_size(-1)
