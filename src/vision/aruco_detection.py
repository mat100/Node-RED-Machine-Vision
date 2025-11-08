"""
ArUco marker detection for machine vision.
Detects fiducial markers and calculates their position and rotation.
"""

from typing import Any, Dict, Optional

import cv2
import numpy as np

from schemas import ROI, Point, VisionObject, VisionObjectType


class ArucoDetector:
    """ArUco marker detector."""

    def __init__(self):
        """Initialize ArUco detector."""

        # Dictionary mapping for OpenCV ArUco
        self.aruco_dicts = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
            "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
            "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
            "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
            "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
            "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
            "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
            "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
            "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
            "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
            "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
        }

    def detect(
        self,
        image: np.ndarray,
        dictionary: str = "DICT_4X4_50",
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Detect ArUco markers in image.

        Args:
            image: Input image (BGR or grayscale)
            dictionary: ArUco dictionary name
            params: Detection parameters (optional)

        Returns:
            Dictionary with detection results
        """
        if params is None:
            params = {}

        # Get ArUco dictionary
        if dictionary not in self.aruco_dicts:
            raise ValueError(f"Unknown ArUco dictionary: {dictionary}")

        aruco_dict = cv2.aruco.getPredefinedDictionary(self.aruco_dicts[dictionary])
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect markers (OpenCV 4.8+ API with ArucoDetector)
        corners, ids, _ = detector.detectMarkers(gray)

        # Process detected markers
        objects = []
        if ids is not None and len(ids) > 0:
            for i, (corner, marker_id) in enumerate(zip(corners, ids.flatten())):
                marker_obj = self._process_marker(corner[0], int(marker_id), i)
                objects.append(marker_obj)

        # Create visualization using overlay rendering function
        from core.image.overlay import render_aruco_markers

        image_result = render_aruco_markers(image, objects, show_ids=True, show_rotation=True)

        return {
            "success": True,
            "dictionary": dictionary,
            "objects": objects,
            "image": image_result,
        }

    def _process_marker(self, corners: np.ndarray, marker_id: int, index: int):
        """
        Process single marker corners into VisionObject.

        Args:
            corners: 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            marker_id: ArUco marker ID
            index: Index in detection list

        Returns:
            VisionObject with marker information
        """
        from core.image.geometry import normalize_angle

        # Calculate center point
        center_x = float(np.mean(corners[:, 0]))
        center_y = float(np.mean(corners[:, 1]))

        # Calculate bounding box
        x_min = float(np.min(corners[:, 0]))
        y_min = float(np.min(corners[:, 1]))
        x_max = float(np.max(corners[:, 0]))
        y_max = float(np.max(corners[:, 1]))

        # Calculate rotation from marker orientation
        # ArUco corners: top-left, top-right, bottom-right, bottom-left
        # Calculate angle from center to top-right corner
        top_left = corners[0]
        top_right = corners[1]

        # Vector from top-left to top-right (top edge)
        dx = top_right[0] - top_left[0]
        dy = top_right[1] - top_left[1]

        # Calculate angle (0° = right, normalized to 0-360°)
        angle_rad = np.arctan2(dy, dx)
        angle_deg = normalize_angle(angle_rad, angle_format="0_360")

        # Calculate area
        # Using shoelace formula for polygon area
        area = float(cv2.contourArea(corners))

        # Calculate perimeter
        perimeter = float(cv2.arcLength(corners, True))

        # Create VisionObject
        obj = VisionObject(
            object_id=f"aruco_{marker_id}",
            object_type=VisionObjectType.ARUCO_MARKER.value,
            bounding_box=ROI(
                x=int(x_min),
                y=int(y_min),
                width=int(x_max - x_min),
                height=int(y_max - y_min),
            ),
            center=Point(x=center_x, y=center_y),
            confidence=1.0,  # ArUco detection is binary (found/not found)
            area=area,
            perimeter=perimeter,
            rotation=angle_deg,
            properties={
                "marker_id": marker_id,
                "corners": corners.tolist(),
                "index": index,
            },
        )

        return obj
