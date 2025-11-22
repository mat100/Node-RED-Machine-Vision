"""
Object-specific rendering functions for vision detection results.

This module provides rendering functions for different types of vision
objects (templates, edges, colors, ArUco markers, rotation analysis).
Each renderer knows how to visualize its specific object type with
appropriate annotations and overlays.
"""

import logging
from typing import List, Optional

import cv2
import numpy as np

from image.converters import ensure_bgr
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
)
from models import VisionObject

logger = logging.getLogger(__name__)


def render_template_matches(
    image: np.ndarray,
    objects: List[VisionObject],
    thickness: int = DEFAULT_THICKNESS,
) -> np.ndarray:
    """
    Render template matching results.

    Args:
        image: Input image
        objects: List of detected template matches
        thickness: Line thickness for drawing

    Returns:
        Image with overlays
    """
    result = image.copy()
    for obj in objects:
        bbox = obj.bounding_box
        # Draw bounding box
        draw_bounding_box(result, bbox.x, bbox.y, bbox.width, bbox.height, COLOR_SUCCESS, thickness)
        # Draw confidence
        draw_confidence(result, obj.confidence, bbox.x, bbox.y, COLOR_SUCCESS)
    return result


def render_edge_contours(
    image: np.ndarray,
    objects: List[VisionObject],
    show_centers: bool = True,
    thickness: int = DEFAULT_THICKNESS,
) -> np.ndarray:
    """
    Render edge detection results.

    Args:
        image: Input image
        objects: List of detected contours
        show_centers: Whether to show center points
        thickness: Line thickness for drawing

    Returns:
        Image with overlays
    """
    result = image.copy()
    for obj in objects:
        bbox = obj.bounding_box
        # Draw bounding box
        draw_bounding_box(result, bbox.x, bbox.y, bbox.width, bbox.height, COLOR_INFO, thickness)
        # Draw center if requested
        if show_centers:
            draw_center_point(result, obj.center.x, obj.center.y, COLOR_INFO)
    return result


def render_color_detection(
    image: np.ndarray,
    obj: VisionObject,
    expected_color: Optional[str] = None,
    contour_points: Optional[list] = None,
    thickness: int = DEFAULT_THICKNESS,
    font: int = DEFAULT_FONT,
    font_scale: float = DEFAULT_FONT_SCALE,
    line_type: int = DEFAULT_LINE_TYPE,
) -> np.ndarray:
    """
    Render color detection results.

    Args:
        image: Input image
        obj: Detected color region object
        expected_color: Expected color (if color matching was performed)
        contour_points: Optional contour points (to show analyzed region)
        thickness: Line thickness for drawing
        font: OpenCV font type
        font_scale: Font scale factor
        line_type: Line type for anti-aliasing

    Returns:
        Image with overlays
    """
    result = image.copy()
    bbox = obj.bounding_box
    x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height

    # Determine color based on match status
    is_match = obj.properties.get("match", True)
    color = COLOR_SUCCESS if (is_match or expected_color is None) else COLOR_FAILURE

    # Draw contour if used for masking
    if contour_points is not None:
        try:
            contour = np.array(contour_points, dtype=np.int32)
            # Draw cyan contour outline to show masked region
            cv2.drawContours(result, [contour], -1, (255, 255, 0), 2)  # Cyan
        except (ValueError, cv2.error) as e:
            logger.warning(
                f"Failed to draw contour in color detection: {e}. Falling back to bbox only."
            )
            # Fall back to just bbox if contour drawing fails

    # Draw ROI rectangle (bounding box)
    draw_bounding_box(result, x, y, w, h, color, thickness)

    # Add text with dominant color
    dominant_color = obj.properties.get("dominant_color", "unknown")
    confidence_pct = obj.confidence * 100
    text = f"{dominant_color} ({confidence_pct:.1f}%)"
    draw_label(result, text, x, y - 10, color, False, font, font_scale, thickness, line_type)

    # Add match/fail indicator if expected color was provided
    if expected_color is not None:
        status_text = "MATCH" if is_match else "FAIL"
        draw_label(
            result, status_text, x, y + h + 25, color, False, font, font_scale, thickness, line_type
        )

    return result


def render_aruco_markers(
    image: np.ndarray,
    objects: List[VisionObject],
    show_ids: bool = True,
    show_rotation: bool = True,
    thickness: int = DEFAULT_THICKNESS,
    font: int = DEFAULT_FONT,
    font_scale: float = 1.0,
    line_type: int = DEFAULT_LINE_TYPE,
) -> np.ndarray:
    """
    Render ArUco marker detection results.

    Args:
        image: Input image
        objects: List of detected ArUco markers
        show_ids: Whether to show marker IDs
        show_rotation: Whether to show rotation indicators
        thickness: Line thickness for drawing
        font: OpenCV font type
        font_scale: Font scale factor
        line_type: Line type for anti-aliasing

    Returns:
        Image with overlays
    """
    result = ensure_bgr(image)

    for obj in objects:
        bbox = obj.bounding_box
        marker_id = obj.properties.get("marker_id", "?")
        corners = obj.properties.get("corners", None)

        # Draw marker corners if available
        if corners is not None:
            corners_array = np.array(corners, dtype=np.int32)
            # Draw marker outline
            cv2.polylines(result, [corners_array], True, COLOR_SUCCESS, 2, line_type)

        # Draw bounding box
        draw_bounding_box(result, bbox.x, bbox.y, bbox.width, bbox.height, COLOR_SUCCESS, thickness)

        # Draw marker ID
        if show_ids:
            text = f"ID:{marker_id}"
            draw_label(
                result,
                text,
                bbox.x,
                bbox.y - 10,
                COLOR_SUCCESS,
                False,
                font,
                font_scale,
                thickness,
                line_type,
            )

        # Draw rotation indicator
        if show_rotation and obj.rotation is not None:
            draw_center_point(result, obj.center.x, obj.center.y, COLOR_INFO, radius=5)
            draw_rotation_indicator(
                result, obj.center.x, obj.center.y, obj.rotation, length=40, thickness=thickness
            )

    # Add summary text
    text = f"Markers: {len(objects)}"
    cv2.putText(result, text, (10, 30), font, font_scale, (255, 255, 255), thickness, line_type)

    return result


def render_rotation_analysis(
    image: np.ndarray,
    obj: VisionObject,
    contour: Optional[np.ndarray] = None,
    method: str = "unknown",
    thickness: int = DEFAULT_THICKNESS,
    font: int = DEFAULT_FONT,
    font_scale: float = 0.7,
    line_type: int = DEFAULT_LINE_TYPE,
) -> np.ndarray:
    """
    Render rotation detection results.

    Args:
        image: Input image
        obj: Rotation analysis object
        contour: Optional contour to draw
        method: Detection method name
        thickness: Line thickness for drawing
        font: OpenCV font type
        font_scale: Font scale factor
        line_type: Line type for anti-aliasing

    Returns:
        Image with overlays
    """
    result = ensure_bgr(image)
    bbox = obj.bounding_box

    # Draw contour if provided
    if contour is not None:
        # Convert to int32 and reshape if needed
        contour_array = np.array(contour, dtype=np.int32)
        if len(contour_array.shape) == 3:
            # Already in correct shape (N, 1, 2)
            pass
        elif len(contour_array.shape) == 2:
            # Reshape from (N, 2) to (N, 1, 2)
            contour_array = contour_array.reshape(-1, 1, 2)

        # Draw contour (only if it has points)
        if contour_array.size > 0:
            cv2.drawContours(result, [contour_array], -1, (0, 255, 0), 2, line_type)

    # Draw bounding box
    draw_bounding_box(result, bbox.x, bbox.y, bbox.width, bbox.height, COLOR_INFO, thickness)

    # Draw center point
    draw_center_point(result, obj.center.x, obj.center.y, (0, 0, 255), radius=5)

    # Draw rotation indicator
    if obj.rotation is not None:
        draw_rotation_indicator(
            result, obj.center.x, obj.center.y, obj.rotation, length=50, thickness=thickness
        )

    # Add rotation text
    text = f"Rotation: {obj.rotation:.1f}deg ({method})"
    cv2.putText(result, text, (10, 30), font, font_scale, (255, 255, 255), thickness, line_type)

    return result


def render_edge_detection(
    original: np.ndarray,
    objects: List[VisionObject],
    show_centers: bool = True,
    thickness: int = DEFAULT_THICKNESS,
    font: int = DEFAULT_FONT,
    font_scale: float = 1.0,
    line_type: int = DEFAULT_LINE_TYPE,
) -> np.ndarray:
    """
    Render edge detection results with contours and annotations.

    Args:
        original: Original image
        objects: Detected contour objects
        show_centers: Whether to show center points
        thickness: Line thickness for drawing
        font: OpenCV font type
        font_scale: Font scale factor
        line_type: Line type for anti-aliasing

    Returns:
        Annotated image as np.ndarray
    """
    # Overlay on original
    overlay = ensure_bgr(original)

    # Draw contours
    for i, obj in enumerate(objects):
        # Get contour from object if available
        contour_points = obj.contour
        if contour_points:
            contour = np.array(contour_points, dtype=np.int32)
            # Green for largest, yellow for others
            color = COLOR_SUCCESS if i == 0 else COLOR_INFO
            draw_contour(overlay, contour, color, thickness=2)

        # Draw bounding box
        bbox = obj.bounding_box
        draw_bounding_box(
            overlay, bbox.x, bbox.y, bbox.width, bbox.height, (255, 0, 0), thickness=1
        )

        # Draw center point
        if show_centers:
            draw_center_point(overlay, obj.center.x, obj.center.y, (0, 0, 255))

    # Add text info
    text = f"Contours: {len(objects)}"
    cv2.putText(overlay, text, (10, 30), font, font_scale, (255, 255, 255), thickness, line_type)

    return overlay


def render_objects(
    image: np.ndarray,
    objects: List[VisionObject],
    object_type: Optional[str] = None,
    thickness: int = DEFAULT_THICKNESS,
    **kwargs,
) -> np.ndarray:
    """
    Auto-detect and render objects based on type.

    Args:
        image: Input image
        objects: List of vision objects
        object_type: Override object type detection
        thickness: Line thickness for drawing
        **kwargs: Additional rendering options

    Returns:
        Image with appropriate overlays
    """
    if not objects:
        return image.copy()

    # Detect type from first object if not specified
    if object_type is None:
        object_type = objects[0].object_type

    # Route to appropriate renderer
    if object_type == "template_match":
        return render_template_matches(image, objects, thickness)
    elif object_type == "edge_contour":
        show_centers = kwargs.get("show_centers", True)
        return render_edge_contours(image, objects, show_centers, thickness)
    elif object_type == "color_region":
        expected_color = kwargs.get("expected_color", None)
        return render_color_detection(image, objects[0], expected_color, None, thickness)
    elif object_type == "aruco_marker":
        show_ids = kwargs.get("show_ids", True)
        show_rotation = kwargs.get("show_rotation", True)
        return render_aruco_markers(image, objects, show_ids, show_rotation, thickness)
    elif object_type == "rotation_analysis":
        contour = kwargs.get("contour", None)
        method = kwargs.get("method", "unknown")
        return render_rotation_analysis(image, objects[0], contour, method, thickness)
    else:
        # Default: simple bounding boxes
        result = image.copy()
        for obj in objects:
            bbox = obj.bounding_box
            draw_bounding_box(
                result, bbox.x, bbox.y, bbox.width, bbox.height, COLOR_INFO, thickness
            )
        return result
