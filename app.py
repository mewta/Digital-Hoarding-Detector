"""Streamlit web interface for the Digital Hoarding Detector."""

from __future__ import annotations

import cv2
import numpy as np
import streamlit as st

from digital_hoarding_detector.gallery import (
    GalleryImage,
    GalleryReport,
    analyze_gallery,
    format_storage_size,
)

SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png"]


def decode_uploaded_image(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> GalleryImage:
    """Decode one Streamlit upload into the application image model."""
    file_bytes = uploaded_file.getvalue()
    encoded_image = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to decode {uploaded_file.name}")

    return GalleryImage(
        name=uploaded_file.name,
        image=image,
        size_bytes=len(file_bytes),
    )


def display_image_grid(
    report: GalleryReport,
    indices: tuple[int, ...],
    captions: list[str] | None = None,
) -> None:
    """Display selected uploaded images in a responsive three-column grid."""
    if not indices:
        st.info("No images found in this category.")
        return

    columns = st.columns(3)
    for position, image_index in enumerate(indices):
        gallery_image = report.images[image_index]
        caption = (
            captions[position]
            if captions is not None
            else gallery_image.name
        )
        columns[position % 3].image(
            cv2.cvtColor(gallery_image.image, cv2.COLOR_BGR2RGB),
            caption=caption,
            use_container_width=True,
        )


def display_grouped_images(
    report: GalleryReport,
    groups: tuple[tuple[int, ...], ...],
    label: str,
) -> None:
    """Display duplicate or selfie groups with filenames."""
    if not groups:
        st.info(f"No {label.lower()} found.")
        return

    for group_number, indices in enumerate(groups, start=1):
        st.markdown(f"**{label} {group_number}**")
        display_image_grid(report, indices)


def display_report(report: GalleryReport) -> None:
    """Render cleanup metrics and visual findings."""
    st.subheader("Gallery Cleanup Report")

    metric_columns = st.columns(5)
    metric_columns[0].metric(
        "Duplicates",
        report.duplicate_analysis.duplicate_count,
    )
    metric_columns[1].metric("Blurry", len(report.blurry_indices))
    metric_columns[2].metric("Screenshots", len(report.screenshot_indices))
    metric_columns[3].metric(
        "Selfie groups",
        report.selfie_analysis.cluster_count,
    )
    metric_columns[4].metric(
        "Potential savings",
        format_storage_size(report.potential_savings_bytes),
    )

    st.caption(
        "Storage savings are estimated from unique cleanup candidates. "
        "Review every suggestion before deleting files."
    )

    duplicate_groups = tuple(
        group.image_indices for group in report.duplicate_analysis.groups
    )
    selfie_groups = tuple(
        group.image_indices for group in report.selfie_analysis.groups
    )

    duplicate_tab, blur_tab, screenshot_tab, selfie_tab, cleanup_tab = st.tabs(
        [
            "Duplicates",
            "Blurry",
            "Screenshots",
            "Similar selfies",
            "Cleanup candidates",
        ]
    )

    with duplicate_tab:
        display_grouped_images(report, duplicate_groups, "Duplicate group")

    with blur_tab:
        blur_captions = [
            (
                f"{report.images[index].name} · "
                f"sharpness {report.blur_results[index].variance:.1f}"
            )
            for index in report.blurry_indices
        ]
        display_image_grid(report, report.blurry_indices, blur_captions)

    with screenshot_tab:
        screenshot_captions = [
            (
                f"{report.images[index].name} · "
                f"confidence {report.screenshot_results[index].score:.0%}"
            )
            for index in report.screenshot_indices
        ]
        display_image_grid(
            report,
            report.screenshot_indices,
            screenshot_captions,
        )

    with selfie_tab:
        display_grouped_images(report, selfie_groups, "Selfie group")

    with cleanup_tab:
        display_image_grid(report, report.cleanup_candidate_indices)


def main() -> None:
    """Configure and run the Streamlit application."""
    st.set_page_config(
        page_title="Digital Hoarding Detector",
        page_icon=None,
        layout="wide",
    )
    st.title("Digital Hoarding Detector")
    st.write(
        "Upload phone-gallery images to identify duplicates, blurry photos, "
        "screenshots, and similar selfies."
    )

    uploaded_files = st.file_uploader(
        "Upload images",
        type=SUPPORTED_IMAGE_TYPES,
        accept_multiple_files=True,
        help="Select multiple JPG, JPEG, or PNG images from your gallery.",
    )

    with st.expander("Detection settings"):
        blur_threshold = st.slider(
            "Blur threshold",
            min_value=20,
            max_value=500,
            value=100,
            help="Higher values classify more images as blurry.",
        )
        duplicate_distance = st.slider(
            "Duplicate sensitivity",
            min_value=0,
            max_value=15,
            value=5,
            help="Higher values allow more visual differences.",
        )
        screenshot_threshold = st.slider(
            "Screenshot confidence threshold",
            min_value=0.30,
            max_value=0.90,
            value=0.55,
            step=0.05,
        )
        selfie_threshold = st.slider(
            "Selfie similarity threshold",
            min_value=0.70,
            max_value=0.99,
            value=0.88,
            step=0.01,
        )

    if st.button(
        "Analyze Gallery",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    ):
        decoded_images: list[GalleryImage] = []
        decoding_errors: list[str] = []

        with st.spinner(f"Analyzing {len(uploaded_files)} images..."):
            for uploaded_file in uploaded_files:
                try:
                    decoded_images.append(decode_uploaded_image(uploaded_file))
                except ValueError as error:
                    decoding_errors.append(str(error))

            if decoded_images:
                report = analyze_gallery(
                    decoded_images,
                    blur_threshold=float(blur_threshold),
                    duplicate_max_distance=duplicate_distance,
                    screenshot_threshold=screenshot_threshold,
                    selfie_similarity_threshold=selfie_threshold,
                )
                st.session_state["gallery_report"] = report

        for error in decoding_errors:
            st.warning(error)

    report = st.session_state.get("gallery_report")
    if report is not None:
        display_report(report)


if __name__ == "__main__":
    main()
