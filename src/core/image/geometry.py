"""
Geometric calculations for image processing.

Handles geometric operations:
- Angle normalization and conversions
- Contour property calculations
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def normalize_angle(
    angle_rad: float, angle_format: str = "0_360", return_radians: bool = False
) -> float:
    """
    Normalize angle to requested format.

    Args:
        angle_rad: Angle in radians
        angle_format: Output format - "0_360", "-180_180", or "0_180"
        return_radians: If True, return angle in radians instead of degrees

    Returns:
        Normalized angle in degrees (or radians if return_radians=True)

    Raises:
        ValueError: If angle_format is invalid
    """
    # Convert to degrees
    angle = float(np.degrees(angle_rad))

    # Normalize to requested format
    if angle_format == "0_360":
        while angle < 0:
            angle += 360
        while angle >= 360:
            angle -= 360
    elif angle_format == "-180_180":
        while angle < -180:
            angle += 360
        while angle > 180:
            angle -= 360
    elif angle_format == "0_180":
        while angle < 0:
            angle += 180
        while angle >= 180:
            angle -= 180
    else:
        raise ValueError(
            f"Invalid angle_format: {angle_format}. Must be '0_360', '-180_180', or '0_180'"
        )

    # Convert back to radians if requested
    if return_radians:
        return float(np.radians(angle))

    return angle


def calculate_contour_properties(contour: np.ndarray) -> dict:
    """
    Calculate standard geometric properties for a contour.

    Args:
        contour: OpenCV contour (NumPy array)

    Returns:
        Dictionary with contour properties:
            - area: Contour area
            - perimeter: Contour perimeter
            - center: Centroid as (x, y) tuple
            - center_x: Center X coordinate
            - center_y: Center Y coordinate
            - bounding_box: Bounding rectangle as (x, y, w, h) tuple
            - x, y, width, height: Individual bounding box components
    """
    # Calculate area and perimeter
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))

    # Calculate centroid using moments
    M = cv2.moments(contour)
    if M["m00"] != 0:
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
    else:
        center_x, center_y = 0, 0

    # Calculate bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)

    return {
        "area": area,
        "perimeter": perimeter,
        "center": (center_x, center_y),
        "center_x": center_x,
        "center_y": center_y,
        "bounding_box": (x, y, w, h),
        "x": x,
        "y": y,
        "width": w,
        "height": h,
    }
