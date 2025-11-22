"""
Reference plane visualization functions.

This module provides visualization for ArUco-based reference frames,
including coordinate axes, grids, and origin markers. Used for displaying
calibrated measurement planes with real-world units.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from image.overlay import DEFAULT_FONT, DEFAULT_LINE_TYPE, DEFAULT_THICKNESS, render_aruco_markers
from models import ReferenceObject, VisionObject


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
