"""
Overlay rendering utilities for vision detection results.

Provides consistent visualization of detection results across different
vision algorithms (template matching, edge detection, color detection).

This package is organized into three modules:
- primitives: Basic drawing functions (boxes, labels, points, etc.)
- objects: Object-specific rendering (templates, edges, colors, ArUco, rotation)
- reference: Reference plane visualization with coordinate systems

All functions are re-exported at the package level for backwards compatibility.
"""

# Re-export constants and primitives
from image.overlay import (
    COLOR_FAILURE,
    COLOR_INFO,
    COLOR_SUCCESS,
    DEFAULT_FONT,
    DEFAULT_FONT_SCALE,
    DEFAULT_LINE_TYPE,
    DEFAULT_THICKNESS,
    draw_bounding_box,
    draw_center_point,
    draw_confidence,
    draw_contour,
    draw_label,
    draw_rotation_indicator,
    render_aruco_markers,
    render_color_detection,
    render_edge_contours,
    render_edge_detection,
    render_objects,
    render_reference_plane,
    render_rotation_analysis,
    render_template_matches,
)

__all__ = [
    # Constants
    "COLOR_SUCCESS",
    "COLOR_FAILURE",
    "COLOR_INFO",
    "DEFAULT_FONT",
    "DEFAULT_FONT_SCALE",
    "DEFAULT_THICKNESS",
    "DEFAULT_LINE_TYPE",
    # Primitive drawing functions
    "draw_bounding_box",
    "draw_label",
    "draw_confidence",
    "draw_center_point",
    "draw_contour",
    "draw_rotation_indicator",
    # Object rendering functions
    "render_template_matches",
    "render_edge_contours",
    "render_color_detection",
    "render_aruco_markers",
    "render_rotation_analysis",
    "render_edge_detection",
    "render_objects",
    # Reference plane rendering
    "render_reference_plane",
]
