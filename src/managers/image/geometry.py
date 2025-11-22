"""
Geometric calculations for image processing.

Handles geometric operations:
- Angle normalization and conversions
- Contour property calculations
"""

import logging
from typing import Any, Dict, Tuple

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


def calculate_contour_properties(contour: np.ndarray) -> Dict[str, Any]:
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


# === Homography and Transform Functions ===


def compute_homography_from_points(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """
    Compute homography matrix from source to destination points.

    Args:
        src_points: Source points (Nx2) in image coordinates
        dst_points: Destination points (Nx2) in reference coordinates

    Returns:
        3x3 homography matrix

    Raises:
        ValueError: If points cannot form valid homography (need at least 4 points)
    """
    if len(src_points) < 4 or len(dst_points) < 4:
        raise ValueError("Need at least 4 point correspondences for homography")

    if len(src_points) != len(dst_points):
        raise ValueError("Source and destination must have same number of points")

    # Ensure float32 type for OpenCV
    src = np.array(src_points, dtype=np.float32).reshape(-1, 1, 2)
    dst = np.array(dst_points, dtype=np.float32).reshape(-1, 1, 2)

    # Find homography using RANSAC for robustness
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

    if H is None:
        raise ValueError("Failed to compute homography - points may be coplanar or degenerate")

    return H


def create_affine_transform(
    origin: Tuple[float, float], rotation_deg: float, scale: float
) -> np.ndarray:
    """
    Create affine transformation matrix (simplified homography) for single marker reference.

    Args:
        origin: Origin point (x, y) in image coordinates
        rotation_deg: Rotation offset in degrees
        scale: Uniform scale factor (mm per pixel)

    Returns:
        3x3 homography matrix (affine transform embedded in homography)
    """
    # Convert rotation to radians
    theta = np.radians(rotation_deg)

    # Create rotation matrix
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Affine transformation: scale, rotate, translate
    # H = T * R * S where:
    # - S scales coordinates
    # - R rotates around origin
    # - T translates to new origin

    H = np.array(
        [
            [scale * cos_t, -scale * sin_t, -origin[0] * scale * cos_t + origin[1] * scale * sin_t],
            [scale * sin_t, scale * cos_t, -origin[0] * scale * sin_t - origin[1] * scale * cos_t],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    return H


def transform_point_homography(
    point: Tuple[float, float], homography: np.ndarray
) -> Tuple[float, float]:
    """
    Transform point using homography matrix.

    Args:
        point: Point (x, y) in source coordinates
        homography: 3x3 homography matrix

    Returns:
        Transformed point (x, y) in destination coordinates
    """
    # Convert to homogeneous coordinates
    p = np.array([point[0], point[1], 1.0], dtype=np.float64)

    # Apply homography
    p_transformed = homography @ p

    # Convert back from homogeneous (divide by w)
    if abs(p_transformed[2]) < 1e-10:
        logger.warning("Homography transformation resulted in point at infinity")
        return (float("nan"), float("nan"))

    x = p_transformed[0] / p_transformed[2]
    y = p_transformed[1] / p_transformed[2]

    return (float(x), float(y))


def transform_rotation_homography(
    rotation_deg: float, homography: np.ndarray, center: Tuple[float, float]
) -> float:
    """
    Transform rotation angle accounting for reference frame rotation.

    Uses homography to determine rotation offset by comparing transformed
    direction vector to original.

    Args:
        rotation_deg: Original rotation in image coordinates (degrees)
        homography: 3x3 homography matrix
        center: Center point of rotation in image coordinates

    Returns:
        Transformed rotation in reference frame (degrees, 0-360 range)
    """
    # Create a unit vector in the direction of the original rotation
    theta = np.radians(rotation_deg)
    direction = np.array([np.cos(theta), np.sin(theta)])

    # Transform center point and point offset by direction
    center_transformed = transform_point_homography(center, homography)
    offset_point = (center[0] + direction[0] * 10, center[1] + direction[1] * 10)
    offset_transformed = transform_point_homography(offset_point, homography)

    # Calculate new direction vector
    dx = offset_transformed[0] - center_transformed[0]
    dy = offset_transformed[1] - center_transformed[1]

    # Calculate transformed rotation
    transformed_rotation = np.degrees(np.arctan2(dy, dx))

    # Normalize to 0-360
    while transformed_rotation < 0:
        transformed_rotation += 360
    while transformed_rotation >= 360:
        transformed_rotation -= 360

    return float(transformed_rotation)


def estimate_scale_from_homography(homography: np.ndarray) -> Tuple[float, float]:
    """
    Estimate scale factors (x and y) from homography matrix.

    For affine transforms, scale is uniform. For perspective transforms,
    scale varies across the image - this returns approximate scale at origin.

    Args:
        homography: 3x3 homography matrix

    Returns:
        Tuple (scale_x, scale_y) - scaling factors in each dimension
    """
    # Extract upper-left 2x2 submatrix (linear part)
    M = homography[:2, :2]

    # Compute SVD to extract scaling
    U, S, Vt = np.linalg.svd(M)

    # Singular values are the scale factors
    scale_x = float(S[0])
    scale_y = float(S[1])

    return (scale_x, scale_y)


def transform_dimensions_homography(
    width: float, height: float, homography: np.ndarray, center: Tuple[float, float]
) -> Tuple[float, float]:
    """
    Transform bounding box dimensions using homography.

    Transforms the 4 corners of a bounding box and computes new dimensions.

    Args:
        width: Original width in pixels
        height: Original height in pixels
        homography: 3x3 homography matrix
        center: Center point (x, y) in image coordinates

    Returns:
        Tuple (transformed_width, transformed_height) in reference units
    """
    # Define corners relative to center
    half_w = width / 2.0
    half_h = height / 2.0

    corners = [
        (center[0] - half_w, center[1] - half_h),  # top-left
        (center[0] + half_w, center[1] - half_h),  # top-right
        (center[0] + half_w, center[1] + half_h),  # bottom-right
        (center[0] - half_w, center[1] + half_h),  # bottom-left
    ]

    # Transform all corners
    transformed_corners = [transform_point_homography(c, homography) for c in corners]

    # Find bounding box of transformed corners
    xs = [c[0] for c in transformed_corners]
    ys = [c[1] for c in transformed_corners]

    transformed_width = max(xs) - min(xs)
    transformed_height = max(ys) - min(ys)

    return (float(transformed_width), float(transformed_height))
