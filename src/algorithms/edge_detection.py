"""
Edge detection algorithms for machine vision.
"""

from typing import Any, Dict, Optional

import cv2
import numpy as np

from algorithms.base_detector import BaseDetector
from domain_types import ROI, EdgeMethod, Point, VisionConstants, VisionObjectType
from image.geometry import calculate_contour_properties
from image.overlay import render_edge_detection
from models import VisionObject


class EdgeDetector(BaseDetector):
    """Edge detection processor."""

    def __init__(self):
        """Initialize edge detector."""
        super().__init__()

    def detect(
        self,
        image: np.ndarray,
        method: EdgeMethod = EdgeMethod.CANNY,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform edge detection on image.

        Args:
            image: Input image (BGR or grayscale)
            method: Edge detection method
            params: Method-specific parameters

        Returns:
            Dictionary with edge detection results
        """
        from image.converters import ensure_grayscale

        if params is None:
            params = {}

        # Convert to grayscale if needed
        gray = ensure_grayscale(image)

        # Apply edge detection
        if method == EdgeMethod.CANNY:
            edges = self._detect_canny(gray, params)
        elif method == EdgeMethod.SOBEL:
            edges = self._detect_sobel(gray, params)
        elif method == EdgeMethod.LAPLACIAN:
            edges = self._detect_laplacian(gray, params)
        elif method == EdgeMethod.PREWITT:
            edges = self._detect_prewitt(gray, params)
        elif method == EdgeMethod.SCHARR:
            edges = self._detect_scharr(gray, params)
        elif method == EdgeMethod.MORPHOLOGICAL_GRADIENT:
            edges = self._detect_morphological_gradient(gray, params)
        else:
            raise ValueError(f"Unknown edge detection method: {method}")

        # Find contours on the edge image
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter and analyze contours
        filtered_contours = self._filter_contours(contours, params)

        # Convert to DetectedObject instances
        objects = self._contours_to_objects(filtered_contours, method.value)

        # Create visualization using overlay rendering function
        show_centers = params.get("show_centers", True)
        annotated_image = render_edge_detection(image, objects, show_centers=show_centers)

        return {
            "success": True,
            "method": method,
            "objects": objects,
            "image": annotated_image,
        }

    def _detect_canny(self, gray: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Apply Canny edge detection."""
        low_threshold = params.get("canny_low", 50)
        high_threshold = params.get("canny_high", 150)
        aperture_size = params.get("canny_aperture", 3)
        l2_gradient = params.get("canny_l2_gradient", False)

        return cv2.Canny(
            gray,
            low_threshold,
            high_threshold,
            apertureSize=aperture_size,
            L2gradient=l2_gradient,
        )

    def _detect_sobel(self, gray: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Apply Sobel edge detection."""
        kernel_size = int(params.get("sobel_kernel", 3))
        scale = float(params.get("sobel_scale", 1.0))
        delta = float(params.get("sobel_delta", 0.0))

        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size, scale=scale, delta=delta)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size, scale=scale, delta=delta)

        # Combine gradients
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Threshold
        threshold = float(params.get("sobel_threshold", 50.0))
        edges = np.uint8(magnitude > threshold) * 255

        return edges

    def _detect_laplacian(self, gray: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Apply Laplacian edge detection."""
        kernel_size = int(params.get("laplacian_kernel", 3))
        scale = float(params.get("laplacian_scale", 1.0))
        delta = float(params.get("laplacian_delta", 0.0))

        # Apply Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel_size, scale=scale, delta=delta)

        # Convert to absolute values and threshold
        laplacian = np.abs(laplacian)
        threshold = float(params.get("laplacian_threshold", 30.0))
        edges = np.uint8(laplacian > threshold) * 255

        return edges

    def _detect_prewitt(self, gray: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Apply Prewitt edge detection."""
        # Prewitt kernels
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

        # Apply filters
        grad_x = cv2.filter2D(gray, cv2.CV_32F, kernel_x)
        grad_y = cv2.filter2D(gray, cv2.CV_32F, kernel_y)

        # Combine gradients
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Threshold
        threshold = float(params.get("prewitt_threshold", 50.0))
        edges = np.uint8(magnitude > threshold) * 255

        return edges

    def _detect_scharr(self, gray: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Apply Scharr edge detection."""
        scale = float(params.get("scharr_scale", 1.0))
        delta = float(params.get("scharr_delta", 0.0))

        # Compute gradients using Scharr operator
        grad_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0, scale=scale, delta=delta)
        grad_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1, scale=scale, delta=delta)

        # Combine gradients
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Threshold
        threshold = float(params.get("scharr_threshold", 50.0))
        edges = np.uint8(magnitude > threshold) * 255

        return edges

    def _detect_morphological_gradient(
        self, gray: np.ndarray, params: Dict[str, Any]
    ) -> np.ndarray:
        """Apply morphological gradient edge detection."""
        kernel_size = int(params.get("morph_kernel", 3))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Morphological gradient
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

        # Threshold
        threshold = float(params.get("morph_threshold", 30.0))
        edges = np.uint8(gradient > threshold) * 255

        return edges

    def _filter_contours(self, contours: list, params: Dict[str, Any]) -> list:
        """Filter contours based on parameters."""

        min_area = float(params.get("min_contour_area", 10.0))
        max_area = float(params.get("max_contour_area", float("inf")))
        min_perimeter = float(
            params.get(
                "min_contour_perimeter",
                0.0,
            )
        )
        max_perimeter = float(params.get("max_contour_perimeter", float("inf")))

        filtered = []
        for contour in contours:
            # Calculate all contour properties using utility function
            props = calculate_contour_properties(contour)
            area = props["area"]
            perimeter = props["perimeter"]

            # Apply filters
            if area < min_area or area > max_area:
                continue
            if perimeter < min_perimeter or perimeter > max_perimeter:
                continue

            # Extract individual properties
            cx, cy = props["center"]
            x, y, w, h = props["bounding_box"]

            # Approximated contour
            epsilon = VisionConstants.CONTOUR_APPROX_EPSILON_FACTOR * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Flatten contour from shape (N, 1, 2) to [[x, y], [x, y], ...]
            contour_points = contour.reshape(-1, 2).tolist()

            filtered.append(
                {
                    "contour": contour_points,
                    "area": float(area),
                    "perimeter": float(perimeter),
                    "center": {"x": cx, "y": cy},
                    "bounding_box": {"x": x, "y": y, "width": w, "height": h},
                    "vertex_count": len(approx),
                    "is_closed": True,
                }
            )

        # Sort by area (largest first)
        filtered.sort(key=lambda x: x["area"], reverse=True)

        # Limit number of contours
        max_contours = int(params.get("max_contours", 100))
        return filtered[:max_contours]

    def _contours_to_objects(self, contours: list, method: str):
        """Convert contour dicts to VisionObject instances."""
        objects = []
        for i, contour_dict in enumerate(contours):
            obj = VisionObject(
                object_id=f"contour_{i}",
                object_type=VisionObjectType.EDGE_CONTOUR.value,
                bounding_box=ROI(**contour_dict["bounding_box"]),
                center=Point(**contour_dict["center"]),
                confidence=1.0,  # Contours are binary (found/not found)
                area=contour_dict["area"],
                perimeter=contour_dict["perimeter"],
                properties={
                    "method": method,
                    "vertex_count": contour_dict["vertex_count"],
                    "is_closed": contour_dict["is_closed"],
                },
                contour=contour_dict["contour"],
            )
            objects.append(obj)
        return objects
