"""
Rotation detection algorithms for machine vision.
Calculates object orientation from contours using various methods.
"""

from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from core.constants import VisionConstants
from core.enums import AngleRange, RotationMethod
from schemas import ROI, Point, VisionObject, VisionObjectType


class RotationDetector:
    """Rotation detection processor."""

    def __init__(self):
        """Initialize rotation detector."""

    def detect(
        self,
        image: np.ndarray,
        contour: List,
        method: RotationMethod = RotationMethod.MIN_AREA_RECT,
        angle_range: AngleRange = AngleRange.RANGE_0_360,
        roi: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Detect rotation angle of object from contour.

        Args:
            image: Input image (for visualization)
            contour: Contour points [[x1,y1], [x2,y2], ...]
            method: Rotation calculation method
            angle_range: Output angle range format
            roi: Optional ROI for context (for visualization only)

        Returns:
            Dictionary with rotation detection results
        """
        from core.image.geometry import calculate_contour_properties

        # Convert contour to numpy array
        contour_array = np.array(contour, dtype=np.float32)

        if len(contour_array.shape) == 2:
            # Reshape to Nx1x2 if needed (OpenCV format)
            contour_array = contour_array.reshape((-1, 1, 2))

        # Validate contour has enough points
        if (
            len(contour_array) < VisionConstants.MIN_POINTS_ELLIPSE_FIT
            and method == RotationMethod.ELLIPSE_FIT
        ):
            raise ValueError(
                f"Ellipse fitting requires at least {VisionConstants.MIN_POINTS_ELLIPSE_FIT} "
                f"points, got {len(contour_array)}"
            )

        if len(contour_array) < VisionConstants.MIN_POINTS_ROTATION:
            raise ValueError(
                f"Rotation detection requires at least {VisionConstants.MIN_POINTS_ROTATION} "
                f"points, got {len(contour_array)}"
            )

        # Calculate rotation based on method
        if method == RotationMethod.MIN_AREA_RECT:
            angle, center, confidence = self._detect_min_area_rect(contour_array)
        elif method == RotationMethod.ELLIPSE_FIT:
            angle, center, confidence = self._detect_ellipse_fit(contour_array)
        elif method == RotationMethod.PCA:
            angle, center, confidence = self._detect_pca(contour_array)
        else:
            raise ValueError(f"Unknown rotation method: {method}")

        # Convert angle to requested range
        angle = self._convert_angle_range(angle, angle_range)

        # Calculate contour properties using utility function
        props = calculate_contour_properties(contour_array)
        x, y, w, h = props["bounding_box"]
        area = props["area"]
        perimeter = props["perimeter"]

        # Create VisionObject
        obj = VisionObject(
            object_id="rotation_analysis",
            object_type=VisionObjectType.ROTATION_ANALYSIS.value,
            bounding_box=ROI(x=x, y=y, width=w, height=h),
            center=center,
            confidence=confidence,
            area=area,
            perimeter=perimeter,
            rotation=angle,
            properties={
                "method": method.value,
                "angle_range": angle_range.value,
                "absolute_angle": angle,  # Same as rotation for now (reference added in Node-RED)
            },
            contour=contour,  # Preserve original contour
        )

        # Create visualization using overlay rendering function
        from core.image.overlay import render_rotation_analysis

        image_result = render_rotation_analysis(
            image, obj, contour=contour_array, method=method.value
        )

        return {
            "success": True,
            "method": method,
            "objects": [obj],
            "image": image_result,
        }

    def _detect_min_area_rect(self, contour: np.ndarray) -> tuple:
        """
        Detect rotation using minimum area rectangle.

        Args:
            contour: Contour points (Nx1x2)

        Returns:
            (angle, center, confidence)
        """
        from core.image.geometry import normalize_angle

        # Fit minimum area rectangle
        rect = cv2.minAreaRect(contour)
        center_tuple, (width, height), angle = rect

        # OpenCV's minAreaRect returns angle in range -90 to 0
        # Convert to 0-360 range with 0° = horizontal right
        # Note: OpenCV angle is from the horizontal to the first side (width side)

        # Adjust based on aspect ratio
        if width < height:
            angle = angle + 90

        # Normalize to 0-360 (convert to radians first for utility function)
        angle_rad = np.radians(angle)
        angle = normalize_angle(angle_rad, angle_format="0_360")

        center = Point(x=float(center_tuple[0]), y=float(center_tuple[1]))
        confidence = VisionConstants.CONFIDENCE_FULL

        return float(angle), center, confidence

    def _detect_ellipse_fit(self, contour: np.ndarray) -> tuple:
        """
        Detect rotation using ellipse fitting.

        Args:
            contour: Contour points (Nx1x2), must have >= 5 points

        Returns:
            (angle, center, confidence)
        """
        from core.image.geometry import normalize_angle

        # Fit ellipse
        ellipse = cv2.fitEllipse(contour)
        center_tuple, axes, angle = ellipse

        # OpenCV's fitEllipse returns angle in range 0-180
        # This is the angle of the major axis from horizontal

        # Normalize to 0-360 (convert to radians first for utility function)
        angle_rad = np.radians(angle)
        angle = normalize_angle(angle_rad, angle_format="0_360")

        center = Point(x=float(center_tuple[0]), y=float(center_tuple[1]))

        # Calculate confidence based on how well ellipse fits the contour
        # (simplified - could be improved with actual error metric)
        confidence = VisionConstants.CONFIDENCE_HIGH

        return float(angle), center, confidence

    def _detect_pca(self, contour: np.ndarray) -> tuple:
        """
        Detect rotation using PCA (Principal Component Analysis).
        Most robust method - finds dominant orientation axis.

        Args:
            contour: Contour points (Nx1x2)

        Returns:
            (angle, center, confidence)
        """
        from core.image.geometry import normalize_angle

        # Reshape contour to 2D array of points
        points = contour.reshape(-1, 2).astype(np.float32)

        # Calculate mean (center)
        mean = np.mean(points, axis=0)
        center = Point(x=float(mean[0]), y=float(mean[1]))

        # Center the points
        centered_points = points - mean

        # Calculate covariance matrix
        cov_matrix = np.cov(centered_points.T)

        # Get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Sort by eigenvalue (largest first)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Principal component (dominant direction)
        principal_axis = eigenvectors[:, 0]

        # Calculate angle from principal axis (normalized to 0-360)
        angle_rad = np.arctan2(principal_axis[1], principal_axis[0])
        angle = normalize_angle(angle_rad, angle_format="0_360")

        # Calculate confidence from eigenvalue ratio
        # Higher ratio = more elongated = more confident in rotation
        if eigenvalues[1] > 0:
            ratio = eigenvalues[0] / eigenvalues[1]
            confidence = min(
                VisionConstants.CONFIDENCE_FULL, ratio / VisionConstants.CONFIDENCE_SCALING_FACTOR
            )
        else:
            confidence = VisionConstants.CONFIDENCE_FULL

        return angle, center, confidence

    def _convert_angle_range(self, angle: float, range_type: AngleRange) -> float:
        """
        Convert angle to requested range.

        Args:
            angle: Input angle in degrees (assumed 0-360)
            range_type: Desired output range

        Returns:
            Angle in requested range
        """
        from core.image.geometry import normalize_angle

        # Map AngleRange enum to ImageGeometry format strings
        format_map = {
            AngleRange.RANGE_0_360: "0_360",
            AngleRange.RANGE_NEG180_180: "-180_180",
            AngleRange.RANGE_0_180: "0_180",
        }

        # Convert to radians, normalize, and return
        angle_rad = np.radians(angle)
        return normalize_angle(angle_rad, angle_format=format_map[range_type])
