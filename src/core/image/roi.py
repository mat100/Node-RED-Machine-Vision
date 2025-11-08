"""
Region of Interest (ROI) handler for the Machine Vision Flow system.

Provides utility functions for ROI validation and image extraction.
Works with the unified ROI Pydantic model from schemas.

NOTE: ROI geometric operations (intersection, union, etc.) are in the ROI model itself.
"""

import logging
from typing import Dict, Optional, Union

import numpy as np

from schemas import ROI

logger = logging.getLogger(__name__)


def extract_roi(
    image: np.ndarray, roi: Union[ROI, Dict], safe_mode: bool = True, padding_value: int = 0
) -> Optional[np.ndarray]:
    """
    Extract ROI from image.

    Args:
        image: Input image
        roi: ROI object or dictionary
        safe_mode: If True, clip ROI to image bounds
        padding_value: Value to use for padding if ROI extends beyond image

    Returns:
        Extracted ROI or None if invalid
    """
    # Convert to ROI object if needed
    if isinstance(roi, dict):
        roi = ROI.from_dict(roi)

    img_height, img_width = image.shape[:2]

    if safe_mode:
        # Clip ROI to image bounds
        roi = roi.clip(img_width, img_height)

        if roi.width <= 0 or roi.height <= 0:
            logger.warning(f"ROI becomes empty after clipping: {roi.to_dict()}")
            return None

        return image[roi.y : roi.y2, roi.x : roi.x2].copy()

    else:
        # Strict mode - check bounds
        try:
            roi.validate_with_constraints(image_width=img_width, image_height=img_height)
        except ValueError as e:
            logger.warning(f"Invalid ROI: {e}")
            return None

        return image[roi.y : roi.y2, roi.x : roi.x2].copy()
