"""
Color detection algorithms for machine vision.

Provides automatic dominant color detection using histogram analysis.
"""

import logging
from typing import Dict, Optional

import cv2
import numpy as np

from schemas import ROI, Point, VisionObject, VisionObjectType


class ColorDetector:
    """Color detection processor using histogram analysis."""

    def __init__(self):
        """Initialize color detector."""

        self.logger = logging.getLogger(__name__)

    def detect(
        self,
        image: np.ndarray,
        roi: Optional[Dict[str, int]] = None,
        contour_points: Optional[list] = None,
        use_contour_mask: bool = True,
        expected_color: Optional[str] = None,
        min_percentage: float = 50.0,
        method: str = "histogram",
    ) -> Dict:
        """
        Detect dominant color in image or ROI using histogram analysis.

        Args:
            image: Input image (BGR format)
            roi: Optional region of interest {x, y, width, height}
            contour_points: Optional contour points for masking
            use_contour_mask: Whether to use contour mask (if contour_points provided)
            expected_color: Expected color name (or None to just detect)
            min_percentage: Minimum percentage for color match
            method: Detection method (only "histogram" supported)

        Returns:
            Dictionary with detection results
        """
        # Extract ROI if specified
        if roi is not None:
            x, y, w, h = roi["x"], roi["y"], roi["width"], roi["height"]
            roi_image = image[y : y + h, x : x + w]
        else:
            roi_image = image
            x, y, w, h = 0, 0, image.shape[1], image.shape[0]

        # Convert to HSV
        hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)

        # Create contour mask if requested and available
        mask = None
        analyzed_pixels = roi_image.shape[0] * roi_image.shape[1]

        if use_contour_mask and contour_points:
            try:
                # Convert contour to ROI-relative coordinates
                contour_array = np.array(contour_points)
                roi_contour = contour_array - [x, y]

                # Create binary mask
                mask = np.zeros(roi_image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [roi_contour.astype(np.int32)], 255)

                # Count actual analyzed pixels
                analyzed_pixels = cv2.countNonZero(mask)
            except Exception as e:
                # If mask creation fails, fall back to full ROI
                self.logger.warning(f"Contour mask creation failed: {e}")
                mask = None

        # Detect dominant colors using histogram analysis
        color_info = self._detect_histogram(hsv, mask)

        # Check if it matches expected color
        match = False
        if expected_color is not None:
            dominant_color = color_info["dominant_color"]
            dominant_percentage = color_info["color_percentages"].get(dominant_color, 0)
            match = (dominant_color == expected_color) and (dominant_percentage >= min_percentage)

        confidence = color_info["color_percentages"].get(color_info["dominant_color"], 0) / 100.0

        # Create VisionObject
        vision_object = VisionObject(
            object_id="color_0",
            object_type=VisionObjectType.COLOR_REGION.value,
            bounding_box=ROI(x=x, y=y, width=w, height=h),
            center=Point(x=float(x + w / 2), y=float(y + h / 2)),
            confidence=confidence,
            area=float(analyzed_pixels),
            properties={
                "dominant_color": color_info["dominant_color"],
                "color_percentages": color_info["all_colors"],
                "hsv_mean": color_info.get("hsv_mean", [0, 0, 0]),
                "expected_color": expected_color,
                "match": match,
                "method": method,
            },
        )

        # Create visualization using overlay rendering function
        from core.image.overlay import render_color_detection

        image_result = render_color_detection(
            image, vision_object, expected_color=expected_color, contour_points=contour_points
        )

        return {
            "objects": [vision_object],
            "image": image_result,
            "success": True,
            "method": method,
        }

    def _detect_histogram(self, hsv: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict:
        """
        Detect dominant color using histogram peak detection (fast).

        Uses vectorized NumPy operations for 10-50x performance improvement
        over nested Python loops.

        Args:
            hsv: HSV image
            mask: Optional binary mask (only analyze masked pixels)

        Returns:
            Dictionary with color information
        """
        from core.image import count_colors_vectorized

        h, s, v = cv2.split(hsv)

        # Apply mask if provided
        if mask is not None:
            # Extract only masked pixels to avoid counting zeros as black
            masked_pixels = hsv[mask > 0]
            if len(masked_pixels) == 0:
                # Empty mask, return default
                total_pixels = 1
                color_counts = {"black": 0}
            else:
                h_masked = masked_pixels[:, 0]
                s_masked = masked_pixels[:, 1]
                v_masked = masked_pixels[:, 2]
                total_pixels = len(masked_pixels)
                # Reshape back to 2D for vectorized counting
                h = h_masked.reshape(-1, 1)
                s = s_masked.reshape(-1, 1)
                v = v_masked.reshape(-1, 1)
                color_counts = count_colors_vectorized(h, s, v)
        else:
            total_pixels = hsv.shape[0] * hsv.shape[1]
            # Use vectorized color counting (much faster than pixel iteration)
            color_counts = count_colors_vectorized(h, s, v)

        # Calculate percentages
        color_percentages = {
            color: (count / total_pixels) * 100 for color, count in color_counts.items()
        }

        # Find dominant color
        dominant_color = max(color_percentages, key=color_percentages.get)

        # Calculate mean HSV for dominant color
        hsv_mean = [int(np.mean(h)), int(np.mean(s)), int(np.mean(v))]

        return {
            "dominant_color": dominant_color,
            "color_percentages": color_percentages,
            "all_colors": {k: round(v, 1) for k, v in color_percentages.items() if v > 0},
            "hsv_mean": hsv_mean,
        }
