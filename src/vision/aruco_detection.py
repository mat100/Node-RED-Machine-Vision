"""
ArUco marker detection for machine vision.
Detects fiducial markers and calculates their position and rotation.

Supports three detection modes:
- MARKERS: Standard detection of all visible markers
- SINGLE: Single marker reference with uniform scaling
- PLANE: 4-marker plane reference with perspective transform
"""

from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from common.base import ROI, Point
from common.enums import ArucoDetectionMode, VisionObjectType
from schemas import ReferenceObject, VisionObject
from vision.base_detector import BaseDetector


class ArucoDetector(BaseDetector):
    """ArUco marker detector."""

    def __init__(self):
        """Initialize ArUco detector."""
        super().__init__()

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
        mode: str = "markers",
        single_config: Optional[Dict[str, Any]] = None,
        plane_config: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Detect ArUco markers in image with optional reference frame creation.

        Args:
            image: Input image (BGR or grayscale)
            dictionary: ArUco dictionary name
            mode: Detection mode (markers, single, plane)
            single_config: Single marker configuration (required if mode=single)
            plane_config: Plane configuration (required if mode=plane)
            params: Detection parameters (optional)

        Returns:
            Dictionary with detection results and optional reference_object
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

        # Create reference_object based on mode
        reference_object = None
        if mode == ArucoDetectionMode.SINGLE.value or mode == "single":
            reference_object = self._create_single_reference(objects, single_config, image.shape)
        elif mode == ArucoDetectionMode.PLANE.value or mode == "plane":
            reference_object = self._create_plane_reference(objects, plane_config, image.shape)

        # Create visualization using overlay rendering function
        from core.image.overlay import render_aruco_markers

        image_result = render_aruco_markers(image, objects, show_ids=True, show_rotation=True)

        result = {
            "success": True,
            "dictionary": dictionary,
            "objects": objects,
            "image": image_result,
        }

        # Add reference_object if created
        if reference_object is not None:
            result["reference_object"] = reference_object

        return result

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

    def _create_single_reference(
        self,
        objects: List[VisionObject],
        config: Optional[Dict[str, Any]],
        image_shape: tuple,
    ) -> Optional[ReferenceObject]:
        """
        Create reference object from single marker.

        Args:
            objects: List of detected markers
            config: Single marker configuration (marker_id, marker_size_mm, origin,
                rotation_reference)
            image_shape: Image shape (height, width, channels)

        Returns:
            ReferenceObject or None if marker not found
        """
        if config is None:
            raise ValueError("single_config is required for single reference mode")

        marker_id = config.get("marker_id")
        marker_size_mm = config.get("marker_size_mm")
        origin = config.get("origin", "marker_center")
        rotation_reference = config.get("rotation_reference", "marker_rotation")

        if marker_id is None or marker_size_mm is None:
            raise ValueError("single_config must contain marker_id and marker_size_mm")

        # Find the specified marker
        marker = None
        for obj in objects:
            if obj.properties.get("marker_id") == marker_id:
                marker = obj
                break

        if marker is None:
            raise ValueError(f"Reference marker {marker_id} not found in image")

        # Calculate marker size in pixels (average of width and height)
        marker_width_px = marker.bounding_box.width
        marker_height_px = marker.bounding_box.height
        marker_size_px = (marker_width_px + marker_height_px) / 2.0

        # Calculate scale (mm per pixel)
        scale = marker_size_mm / marker_size_px

        # Determine origin point
        if origin == "marker_center":
            origin_point = (marker.center.x, marker.center.y)
        elif origin == "marker_top_left":
            corners = marker.properties.get("corners", [])
            if corners:
                origin_point = tuple(corners[0])  # Top-left corner
            else:
                origin_point = (marker.bounding_box.x, marker.bounding_box.y)
        elif origin == "marker_bottom_left":
            corners = marker.properties.get("corners", [])
            if corners:
                origin_point = tuple(corners[3])  # Bottom-left corner
            else:
                origin_point = (
                    marker.bounding_box.x,
                    marker.bounding_box.y + marker.bounding_box.height,
                )
        else:
            origin_point = (marker.center.x, marker.center.y)

        # Determine rotation offset
        if rotation_reference == "marker_rotation":
            rotation_offset = marker.rotation if marker.rotation is not None else 0.0
        else:  # image_axes
            rotation_offset = 0.0

        # Create affine transformation homography
        from core.image.geometry import create_affine_transform

        homography = create_affine_transform(origin_point, rotation_offset, scale)

        # Create reference object
        reference_obj = ReferenceObject(
            type="single_marker",
            units="mm",
            homography_matrix=homography.tolist(),
            metadata={
                "marker_id": marker_id,
                "marker_size_mm": marker_size_mm,
                "marker_size_px": marker_size_px,
                "scale_mm_per_pixel": scale,
                "origin": origin,
                "origin_point_px": origin_point,
                "rotation_reference": rotation_reference,
                "reference_rotation_deg": rotation_offset,
            },
        )

        return reference_obj

    def _create_plane_reference(
        self,
        objects: List[VisionObject],
        config: Optional[Dict[str, Any]],
        image_shape: tuple,
    ) -> Optional[ReferenceObject]:
        """
        Create reference object from 4-marker plane.

        Args:
            objects: List of detected markers
            config: Plane configuration (marker_ids, width_mm, height_mm, origin, directions)
            image_shape: Image shape (height, width, channels)

        Returns:
            ReferenceObject or None if not all markers found
        """
        if config is None:
            raise ValueError("plane_config is required for plane reference mode")

        marker_ids = config.get("marker_ids")
        width_mm = config.get("width_mm")
        height_mm = config.get("height_mm")
        origin = config.get("origin", "top_left")
        x_direction = config.get("x_direction", "right")
        y_direction = config.get("y_direction", "down")

        if not marker_ids or width_mm is None or height_mm is None:
            raise ValueError("plane_config must contain marker_ids, width_mm, and height_mm")

        # Find all 4 markers
        required_corners = ["top_left", "top_right", "bottom_right", "bottom_left"]
        marker_map = {}

        for corner_name in required_corners:
            marker_id = marker_ids.get(corner_name)
            if marker_id is None:
                raise ValueError(f"Missing marker_id for {corner_name} in plane_config")

            # Find marker
            found = False
            for obj in objects:
                if obj.properties.get("marker_id") == marker_id:
                    marker_map[corner_name] = obj
                    found = True
                    break

            if not found:
                raise ValueError(f"Plane marker {corner_name} (ID {marker_id}) not found in image")

        # Extract center points of markers (in pixels)
        src_points = np.array(
            [
                [marker_map["top_left"].center.x, marker_map["top_left"].center.y],
                [marker_map["top_right"].center.x, marker_map["top_right"].center.y],
                [marker_map["bottom_right"].center.x, marker_map["bottom_right"].center.y],
                [marker_map["bottom_left"].center.x, marker_map["bottom_left"].center.y],
            ],
            dtype=np.float32,
        )

        # Define destination points in reference frame (mm)
        # Based on origin and directions
        dst_points = self._compute_plane_reference_points(
            width_mm, height_mm, origin, x_direction, y_direction
        )

        # Compute homography
        from core.image.geometry import compute_homography_from_points

        homography = compute_homography_from_points(src_points, dst_points)

        # Create reference object
        reference_obj = ReferenceObject(
            type="plane",
            units="mm",
            homography_matrix=homography.tolist(),
            metadata={
                "marker_ids": marker_ids,
                "markers_found": {
                    corner: {
                        "marker_id": marker_ids[corner],
                        "center_px": [marker_map[corner].center.x, marker_map[corner].center.y],
                    }
                    for corner in required_corners
                },
                "width_mm": width_mm,
                "height_mm": height_mm,
                "origin": origin,
                "x_direction": x_direction,
                "y_direction": y_direction,
            },
        )

        return reference_obj

    def _compute_plane_reference_points(
        self,
        width_mm: float,
        height_mm: float,
        origin: str,
        x_direction: str,
        y_direction: str,
    ) -> np.ndarray:
        """
        Compute destination points for plane in reference frame coordinates.

        Args:
            width_mm: Plane width in mm
            height_mm: Plane height in mm
            origin: Origin corner (top_left, top_right, bottom_left, bottom_right)
            x_direction: X axis direction (right, left, up, down)
            y_direction: Y axis direction (right, left, up, down)

        Returns:
            4x2 array of destination points [top_left, top_right, bottom_right, bottom_left]
        """
        # Define corners in standard configuration (origin at top_left, X right, Y down)
        standard_points = {
            "top_left": [0, 0],
            "top_right": [width_mm, 0],
            "bottom_right": [width_mm, height_mm],
            "bottom_left": [0, height_mm],
        }

        # Transform based on origin
        if origin == "top_left":
            pass  # Already correct
        elif origin == "top_right":
            # Shift so top_right is at origin
            for key in standard_points:
                standard_points[key][0] -= width_mm
        elif origin == "bottom_left":
            # Shift so bottom_left is at origin
            for key in standard_points:
                standard_points[key][1] -= height_mm
        elif origin == "bottom_right":
            # Shift so bottom_right is at origin
            for key in standard_points:
                standard_points[key][0] -= width_mm
                standard_points[key][1] -= height_mm

        # Apply axis directions (flip if needed)
        if x_direction == "left":
            for key in standard_points:
                standard_points[key][0] = -standard_points[key][0]
        if y_direction == "up":
            for key in standard_points:
                standard_points[key][1] = -standard_points[key][1]

        # Return in order: top_left, top_right, bottom_right, bottom_left
        return np.array(
            [
                standard_points["top_left"],
                standard_points["top_right"],
                standard_points["bottom_right"],
                standard_points["bottom_left"],
            ],
            dtype=np.float32,
        )
