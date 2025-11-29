"""
Overlay rendering utilities for vision detection results.

Provides consistent visualization of detection results across different
vision algorithms (template matching, edge detection, color detection,
ArUco markers, rotation analysis, and reference planes).

This module combines:
- Primitives: Basic drawing functions (boxes, labels, points, arrows)
- Objects: Object-specific rendering (templates, edges, colors, ArUco, rotation)
- Reference: Reference plane visualization with coordinate systems
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from models import ReferenceObject, VisionObject

from .converters import ensure_bgr

logger = logging.getLogger(__name__)


# ==============================================================================
# Constants and Default Parameters
# ==============================================================================

# Default colors (BGR format)
COLOR_SUCCESS = (0, 255, 0)  # Green
COLOR_FAILURE = (0, 0, 255)  # Red
COLOR_INFO = (255, 255, 0)  # Cyan

# Default rendering parameters
DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX
DEFAULT_FONT_SCALE = 0.5
DEFAULT_THICKNESS = 2
DEFAULT_LINE_TYPE = cv2.LINE_AA


# ==============================================================================
# Primitive Drawing Functions
# ==============================================================================


def draw_bounding_box(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    color: Tuple[int, int, int] = COLOR_SUCCESS,
    thickness: int = DEFAULT_THICKNESS,
    line_type: int = DEFAULT_LINE_TYPE,
) -> np.ndarray:
    """
    Draw a bounding box on the image.

    Args:
        image: Input image
        x: Top-left x coordinate
        y: Top-left y coordinate
        width: Box width
        height: Box height
        color: Box color in BGR format
        thickness: Line thickness
        line_type: Line type for anti-aliasing

    Returns:
        Image with bounding box drawn
    """
    pt1 = (x, y)
    pt2 = (x + width, y + height)
    cv2.rectangle(image, pt1, pt2, color, thickness, line_type)
    return image


def draw_label(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: Tuple[int, int, int] = COLOR_SUCCESS,
    background: bool = False,
    font: int = DEFAULT_FONT,
    font_scale: float = DEFAULT_FONT_SCALE,
    thickness: int = DEFAULT_THICKNESS,
    line_type: int = DEFAULT_LINE_TYPE,
) -> np.ndarray:
    """
    Draw text label on the image.

    Args:
        image: Input image
        text: Text to draw
        x: Text x coordinate
        y: Text y coordinate
        color: Text color in BGR format
        background: Whether to draw background rectangle
        font: OpenCV font type
        font_scale: Font scale factor
        thickness: Line thickness
        line_type: Line type for anti-aliasing

    Returns:
        Image with label drawn
    """
    if background:
        # Calculate text size for background
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        # Draw background rectangle
        cv2.rectangle(
            image,
            (x, y - text_height - baseline),
            (x + text_width, y),
            color,
            -1,  # Filled
        )
        # Draw text in white over background
        text_color = (255, 255, 255)
    else:
        text_color = color

    cv2.putText(
        image,
        text,
        (x, y),
        font,
        font_scale,
        text_color,
        thickness,
        line_type,
    )
    return image


def draw_confidence(
    image: np.ndarray,
    confidence: float,
    x: int,
    y: int,
    color: Tuple[int, int, int] = COLOR_SUCCESS,
    font: int = DEFAULT_FONT,
    font_scale: float = DEFAULT_FONT_SCALE,
    thickness: int = DEFAULT_THICKNESS,
    line_type: int = DEFAULT_LINE_TYPE,
) -> np.ndarray:
    """
    Draw confidence score above bounding box.

    Args:
        image: Input image
        confidence: Confidence value (0.0-1.0)
        x: X coordinate
        y: Y coordinate (top of bounding box)
        color: Text color
        font: OpenCV font type
        font_scale: Font scale factor
        thickness: Line thickness
        line_type: Line type for anti-aliasing

    Returns:
        Image with confidence drawn
    """
    text = f"{confidence:.2f}"
    return draw_label(image, text, x, y - 5, color, False, font, font_scale, thickness, line_type)


def draw_center_point(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    color: Tuple[int, int, int] = COLOR_SUCCESS,
    radius: int = 3,
    line_type: int = DEFAULT_LINE_TYPE,
) -> np.ndarray:
    """
    Draw center point marker.

    Args:
        image: Input image
        center_x: Center x coordinate
        center_y: Center y coordinate
        color: Marker color
        radius: Circle radius
        line_type: Line type for anti-aliasing

    Returns:
        Image with center point drawn
    """
    cv2.circle(
        image,
        (int(center_x), int(center_y)),
        radius,
        color,
        -1,  # Filled
        line_type,
    )
    return image


def draw_contour(
    image: np.ndarray,
    contour: np.ndarray,
    color: Tuple[int, int, int] = COLOR_INFO,
    thickness: int = DEFAULT_THICKNESS,
    line_type: int = DEFAULT_LINE_TYPE,
) -> np.ndarray:
    """
    Draw a contour on the image.

    Args:
        image: Input image
        contour: Contour points (OpenCV format)
        color: Contour color in BGR format
        thickness: Line thickness
        line_type: Line type for anti-aliasing

    Returns:
        Image with contour drawn
    """
    cv2.drawContours(image, [contour.astype(np.int32)], -1, color, thickness, line_type)
    return image


def draw_rotation_indicator(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    angle_deg: float,
    length: int = 50,
    color: Tuple[int, int, int] = (0, 165, 255),  # Orange
    thickness: int = DEFAULT_THICKNESS,
    line_type: int = DEFAULT_LINE_TYPE,
    arrow_tip_length: float = 0.3,
) -> np.ndarray:
    """
    Draw rotation indicator arrow from center showing angle.

    Args:
        image: Input image
        center_x: Center x coordinate
        center_y: Center y coordinate
        angle_deg: Rotation angle in degrees
        length: Arrow length in pixels
        color: Arrow color
        thickness: Line thickness
        line_type: Line type for anti-aliasing
        arrow_tip_length: Arrow tip size ratio

    Returns:
        Image with rotation arrow drawn
    """
    center = (int(center_x), int(center_y))

    # Calculate end point from angle
    angle_rad = np.radians(angle_deg)
    end_x = int(center_x + length * np.cos(angle_rad))
    end_y = int(center_y + length * np.sin(angle_rad))

    # Draw arrowed line
    cv2.arrowedLine(
        image,
        center,
        (end_x, end_y),
        color,
        thickness,
        line_type,
        tipLength=arrow_tip_length,
    )

    return image


# ==============================================================================
# Object-Specific Rendering Functions
# ==============================================================================


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


def render_feature_matches(
    image: np.ndarray,
    objects: List[VisionObject],
    thickness: int = DEFAULT_THICKNESS,
) -> np.ndarray:
    """
    Render feature-based template matching results.

    Draws rotated bounding box (polygon) based on transformed corners,
    rotation indicator arrow, and confidence score.

    Args:
        image: Input image
        objects: List of detected feature matches
        thickness: Line thickness for drawing

    Returns:
        Image with overlays
    """
    result = image.copy()

    for obj in objects:
        # Get transformed corners from properties
        corners = obj.properties.get("corners")

        if corners and len(corners) == 4:
            # Draw rotated bounding box as polygon
            pts = np.array(corners, dtype=np.int32)
            cv2.polylines(result, [pts], isClosed=True, color=COLOR_SUCCESS, thickness=thickness)

            # Draw corner markers
            for pt in pts:
                cv2.circle(result, tuple(pt), 4, COLOR_SUCCESS, -1)

            # Draw rotation indicator arrow from center
            center = (int(obj.center.x), int(obj.center.y))
            rotation = obj.rotation if obj.rotation is not None else 0

            # Arrow length proportional to object size
            arrow_len = min(obj.bounding_box.width, obj.bounding_box.height) // 3
            angle_rad = np.radians(rotation)
            end_x = int(center[0] + arrow_len * np.cos(angle_rad))
            end_y = int(center[1] - arrow_len * np.sin(angle_rad))  # Negative because Y is inverted
            cv2.arrowedLine(result, center, (end_x, end_y), (0, 165, 255), thickness)

            # Draw confidence label
            label_x = int(pts[:, 0].min())
            label_y = int(pts[:, 1].min()) - 10
            draw_confidence(result, obj.confidence, label_x, label_y, COLOR_SUCCESS)

            # Draw rotation angle
            rotation_text = f"{rotation:.1f}deg"
            cv2.putText(
                result,
                rotation_text,
                (label_x, label_y - 20),
                DEFAULT_FONT,
                DEFAULT_FONT_SCALE,
                COLOR_SUCCESS,
                1,
                DEFAULT_LINE_TYPE,
            )
        else:
            # Fallback to axis-aligned bounding box
            bbox = obj.bounding_box
            draw_bounding_box(
                result, bbox.x, bbox.y, bbox.width, bbox.height, COLOR_SUCCESS, thickness
            )
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


# ==============================================================================
# Reference Plane Visualization
# ==============================================================================


def render_reference_plane(
    image: np.ndarray,
    reference_object: ReferenceObject,
    markers: List[VisionObject],
    grid_spacing_mm: Optional[float] = None,
    thickness: int = DEFAULT_THICKNESS,
    font: int = DEFAULT_FONT,
    font_scale: float = 0.6,
    line_type: int = DEFAULT_LINE_TYPE,
) -> np.ndarray:
    """
    Render reference plane visualization with markers, axes, grid, and origin.

    Args:
        image: Input image
        reference_object: Reference object containing homography and metadata
        markers: List of detected ArUco markers
        grid_spacing_mm: Grid spacing in millimeters (auto if None)
        thickness: Line thickness for drawing
        font: OpenCV font type
        font_scale: Font scale factor
        line_type: Line type for anti-aliasing

    Returns:
        Image with reference plane visualization
    """
    # Start with markers rendered
    result = render_aruco_markers(
        image, markers, show_ids=True, show_rotation=True, thickness=thickness, font_scale=0.8
    )

    metadata = reference_object.metadata
    ref_type = reference_object.type
    units = reference_object.units
    homography = np.array(reference_object.homography_matrix, dtype=np.float32)

    # Get origin point in pixels
    origin_px = _get_origin_point(metadata, ref_type, markers)
    if origin_px is None:
        return result  # Can't visualize without origin

    # Draw coordinate axes
    _draw_coordinate_axes(
        result,
        origin_px,
        metadata,
        ref_type,
        homography,
        units,
        thickness,
        font,
        font_scale,
        line_type,
    )

    # Draw grid for plane mode
    if ref_type == "plane":
        _draw_reference_grid(result, metadata, homography, grid_spacing_mm, thickness, line_type)

    # Draw origin marker (prominent)
    _draw_origin_marker(result, origin_px, thickness, font, font_scale, line_type)

    return result


def _get_origin_point(
    metadata: dict, ref_type: str, markers: List[VisionObject]
) -> Optional[Tuple[int, int]]:
    """Get origin point in pixel coordinates."""
    if ref_type == "single_marker":
        origin_point = metadata.get("origin_point_px")
        if origin_point:
            return (int(origin_point[0]), int(origin_point[1]))
    elif ref_type == "plane":
        # Get origin corner marker
        origin = metadata.get("origin", "top_left")
        marker_ids_map = metadata.get("marker_ids", {})
        origin_marker_id = marker_ids_map.get(origin)

        # Find the marker with this ID
        for marker in markers:
            if marker.properties.get("marker_id") == origin_marker_id:
                return (int(marker.center.x), int(marker.center.y))

    return None


def _draw_coordinate_axes(
    image: np.ndarray,
    origin_px: Tuple[int, int],
    metadata: dict,
    ref_type: str,
    homography: np.ndarray,
    units: str,
    thickness: int,
    font: int,
    font_scale: float,
    line_type: int = DEFAULT_LINE_TYPE,
) -> None:
    """Draw X and Y coordinate axes with labels and arrows."""
    # Determine axis directions and lengths
    if ref_type == "single_marker":
        # For single marker, use rotation reference
        rotation_deg = metadata.get("reference_rotation_deg", 0)
        axis_length_px = 100  # Fixed length for visualization

        # X axis (red)
        x_angle = np.radians(rotation_deg)
        x_end = (
            int(origin_px[0] + axis_length_px * np.cos(x_angle)),
            int(origin_px[1] + axis_length_px * np.sin(x_angle)),
        )
        cv2.arrowedLine(image, origin_px, x_end, (0, 0, 255), thickness, line_type, tipLength=0.2)
        cv2.putText(
            image, f"X({units})", x_end, font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA
        )

        # Y axis (green)
        y_angle = np.radians(rotation_deg + 90)
        y_end = (
            int(origin_px[0] + axis_length_px * np.cos(y_angle)),
            int(origin_px[1] + axis_length_px * np.sin(y_angle)),
        )
        cv2.arrowedLine(image, origin_px, y_end, (0, 255, 0), thickness, line_type, tipLength=0.2)
        cv2.putText(
            image, f"Y({units})", y_end, font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA
        )

    elif ref_type == "plane":
        # For plane mode, use actual corner positions
        x_dir = metadata.get("x_direction", "right")
        y_dir = metadata.get("y_direction", "down")
        width_mm = metadata.get("width_mm", 100)
        height_mm = metadata.get("height_mm", 100)

        # Invert homography to transform from mm to pixels
        # (homography transforms pixel → mm, we need mm → pixel)
        inv_homography = np.linalg.inv(homography)

        # Transform reference points to pixel coordinates
        # X-axis endpoint (in mm)
        x_ref_mm = np.array([[width_mm, 0, 1]], dtype=np.float32).T
        x_ref_px = inv_homography @ x_ref_mm
        x_ref_px = x_ref_px[:2] / x_ref_px[2]
        x_end = (int(x_ref_px[0]), int(x_ref_px[1]))

        # Y-axis endpoint (in mm)
        y_ref_mm = np.array([[0, height_mm, 1]], dtype=np.float32).T
        y_ref_px = inv_homography @ y_ref_mm
        y_ref_px = y_ref_px[:2] / y_ref_px[2]
        y_end = (int(y_ref_px[0]), int(y_ref_px[1]))

        # Draw axes with arrows
        cv2.arrowedLine(image, origin_px, x_end, (0, 0, 255), thickness, line_type, tipLength=0.15)
        cv2.putText(
            image,
            f"X({units}) {x_dir}",
            x_end,
            font,
            font_scale,
            (0, 0, 255),
            thickness,
            cv2.LINE_AA,
        )

        cv2.arrowedLine(image, origin_px, y_end, (0, 255, 0), thickness, line_type, tipLength=0.15)
        cv2.putText(
            image,
            f"Y({units}) {y_dir}",
            y_end,
            font,
            font_scale,
            (0, 255, 0),
            thickness,
            cv2.LINE_AA,
        )


def _draw_reference_grid(
    image: np.ndarray,
    metadata: dict,
    homography: np.ndarray,
    grid_spacing_mm: Optional[float],
    thickness: int,
    line_type: int,
) -> None:
    """Draw grid overlay showing physical measurements."""
    width_mm = metadata.get("width_mm", 100)
    height_mm = metadata.get("height_mm", 100)

    # Auto-determine grid spacing if not provided
    if grid_spacing_mm is None:
        # Aim for ~5-10 grid lines
        grid_spacing_mm = max(10, round(min(width_mm, height_mm) / 8 / 10) * 10)

    # Grid color (semi-transparent cyan)
    grid_color = (255, 200, 0)  # Cyan

    # Invert homography to transform from mm to pixels
    inv_homography = np.linalg.inv(homography)

    # Draw vertical grid lines (constant X in mm)
    x_mm = 0
    while x_mm <= width_mm:
        # Transform two points along this vertical line
        pt1_mm = np.array([[x_mm, 0, 1]], dtype=np.float32).T
        pt2_mm = np.array([[x_mm, height_mm, 1]], dtype=np.float32).T

        pt1_px = inv_homography @ pt1_mm
        pt1_px = pt1_px[:2] / pt1_px[2]
        pt2_px = inv_homography @ pt2_mm
        pt2_px = pt2_px[:2] / pt2_px[2]

        cv2.line(
            image,
            (int(pt1_px[0]), int(pt1_px[1])),
            (int(pt2_px[0]), int(pt2_px[1])),
            grid_color,
            thickness,
            line_type,
        )

        x_mm += grid_spacing_mm

    # Draw horizontal grid lines (constant Y in mm)
    y_mm = 0
    while y_mm <= height_mm:
        pt1_mm = np.array([[0, y_mm, 1]], dtype=np.float32).T
        pt2_mm = np.array([[width_mm, y_mm, 1]], dtype=np.float32).T

        pt1_px = inv_homography @ pt1_mm
        pt1_px = pt1_px[:2] / pt1_px[2]
        pt2_px = inv_homography @ pt2_mm
        pt2_px = pt2_px[:2] / pt2_px[2]

        cv2.line(
            image,
            (int(pt1_px[0]), int(pt1_px[1])),
            (int(pt2_px[0]), int(pt2_px[1])),
            grid_color,
            thickness,
            line_type,
        )

        y_mm += grid_spacing_mm


def _draw_origin_marker(
    image: np.ndarray,
    origin_px: Tuple[int, int],
    thickness: int,
    font: int,
    font_scale: float,
    line_type: int,
) -> None:
    """Draw prominent origin marker at [0,0] point."""
    # Draw large circle
    cv2.circle(image, origin_px, 10, (255, 0, 255), thickness, line_type)  # Magenta circle
    # Draw crosshair
    cross_len = 15
    cv2.line(
        image,
        (origin_px[0] - cross_len, origin_px[1]),
        (origin_px[0] + cross_len, origin_px[1]),
        (255, 0, 255),
        thickness,
        line_type,
    )
    cv2.line(
        image,
        (origin_px[0], origin_px[1] - cross_len),
        (origin_px[0], origin_px[1] + cross_len),
        (255, 0, 255),
        thickness,
        line_type,
    )
    # Label
    cv2.putText(
        image,
        "[0,0]",
        (origin_px[0] + 15, origin_px[1] - 15),
        font,
        font_scale,
        (255, 0, 255),
        thickness,
        cv2.LINE_AA,
    )
