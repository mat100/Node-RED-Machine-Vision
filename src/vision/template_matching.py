"""
Template matching algorithms for machine vision.

Provides template matching using OpenCV with multiple correlation methods.
"""

import logging
from typing import Any, Dict

import cv2
import numpy as np

from schemas import ROI, Point, VisionObject, VisionObjectType


class TemplateDetector:
    """Template matching processor using OpenCV methods."""

    def __init__(self):
        """Initialize template detector."""

        self.logger = logging.getLogger(__name__)

    def detect(
        self,
        image: np.ndarray,
        template: np.ndarray,
        template_id: str,
        params: Dict[str, Any],
    ) -> Dict:
        """
        Perform template matching on image.

        Args:
            image: Input image (BGR format)
            template: Template image to search for
            template_id: Template identifier for metadata
            params: Detection parameters dict

        Returns:
            Dictionary with detection results
        """
        # Extract params
        method = params.get("method", "TM_CCOEFF_NORMED")
        threshold = params.get("threshold", 0.8)

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            search_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            search_gray = image

        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template

        # Perform template matching
        cv_method = getattr(cv2, method)
        result = cv2.matchTemplate(search_gray, template_gray, cv_method)

        # Find matches above threshold
        detected_objects = []
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # For SQDIFF methods, lower is better
        if method in ["TM_SQDIFF", "TM_SQDIFF_NORMED"]:
            if min_val <= (1 - threshold):
                score = 1 - min_val
                loc = min_loc
            else:
                score = 0
                loc = None
        else:
            if max_val >= threshold:
                score = max_val
                loc = max_loc
            else:
                score = 0
                loc = None

        # Create VisionObject if match found
        if loc is not None:
            x = loc[0]
            y = loc[1]
            w = template.shape[1]
            h = template.shape[0]

            detected_objects.append(
                VisionObject(
                    object_id="match_0",
                    object_type=VisionObjectType.TEMPLATE_MATCH.value,
                    bounding_box=ROI(x=x, y=y, width=w, height=h),
                    center=Point(x=float(x + w // 2), y=float(y + h // 2)),
                    confidence=min(float(score), 1.0),
                    rotation=0.0,
                    properties={
                        "template_id": template_id,
                        "method": method,
                        "scale": 1.0,
                        "raw_score": float(score),
                    },
                )
            )

        # Create visualization using overlay rendering function
        from core.image.overlay import render_template_matches

        if detected_objects:
            result_image = render_template_matches(image, detected_objects)
        else:
            result_image = image.copy()

        return {
            "success": True,
            "objects": detected_objects,
            "image": result_image,
        }
