"""
Image processing utilities - functional architecture.

This package provides focused image processing utilities as pure functions:
- converters: Format conversions (NumPy, PIL, base64, color spaces)
- processors: Image operations (thumbnail, resize)
- geometry: Geometric calculations (angles, contour properties)
- roi: ROI validation and extraction
- overlay: Vision detection result visualization (still class-based)
- colors: HSV color definitions and color detection utilities

All utilities are re-exported from this module for convenient access.
"""

# Color utilities
from core.image.colors import (
    COLOR_DEFINITIONS,
    count_colors_vectorized,
    create_color_mask_vectorized,
    get_available_colors,
    hsv_to_color_name,
    is_achromatic,
    is_color_match,
)

# Converter functions
from core.image.converters import (
    encode_image_to_base64,
    ensure_bgr,
    ensure_grayscale,
    from_base64,
    to_base64,
)

# Geometry functions
from core.image.geometry import calculate_contour_properties, normalize_angle

# Overlay rendering functions
from core.image.overlay import (
    render_aruco_markers,
    render_color_detection,
    render_edge_contours,
    render_edge_detection,
    render_objects,
    render_rotation_analysis,
    render_template_matches,
)

# Processor functions
from core.image.processors import create_thumbnail, resize_image

# ROI functions
from core.image.roi import extract_roi

__all__ = [
    # Converter functions
    "to_base64",
    "from_base64",
    "encode_image_to_base64",
    "ensure_bgr",
    "ensure_grayscale",
    # Processor functions
    "create_thumbnail",
    "resize_image",
    # Geometry functions
    "normalize_angle",
    "calculate_contour_properties",
    # ROI functions
    "extract_roi",
    # Overlay rendering functions
    "render_template_matches",
    "render_edge_contours",
    "render_color_detection",
    "render_aruco_markers",
    "render_rotation_analysis",
    "render_edge_detection",
    "render_objects",
    # Color utilities
    "COLOR_DEFINITIONS",
    "count_colors_vectorized",
    "create_color_mask_vectorized",
    "get_available_colors",
    "hsv_to_color_name",
    "is_achromatic",
    "is_color_match",
]
