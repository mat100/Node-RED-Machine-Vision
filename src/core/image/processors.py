"""
Image processing operations.

Handles image manipulation tasks using OpenCV:
- Thumbnail creation
- Resizing
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from core.image.converters import to_base64

logger = logging.getLogger(__name__)


def create_thumbnail(
    image: np.ndarray, width: int = 320, maintain_aspect: bool = True
) -> Tuple[np.ndarray, str]:
    """
    Create thumbnail from image using OpenCV.

    Args:
        image: Input image as NumPy array (BGR format)
        width: Target width in pixels
        maintain_aspect: If True, maintain aspect ratio

    Returns:
        Tuple of (thumbnail as NumPy array, thumbnail as base64 string)
    """
    try:
        # Get original dimensions
        h, w = image.shape[:2]

        # Calculate new size
        if maintain_aspect:
            aspect_ratio = h / w
            height = int(width * aspect_ratio)
        else:
            height = width

        # Resize image using high-quality Lanczos interpolation
        # INTER_LANCZOS4 is equivalent to PIL's LANCZOS resampling
        thumbnail_array = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)

        # Convert to base64 (JPEG with quality 70)
        thumbnail_base64 = to_base64(thumbnail_array, format="JPEG", quality=70)

        return thumbnail_array, thumbnail_base64

    except Exception as e:
        logger.error(f"Failed to create thumbnail: {e}")
        raise


def resize_image(
    image: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
    max_dimension: Optional[int] = None,
) -> np.ndarray:
    """
    Resize image with various options.

    Args:
        image: Input image as NumPy array
        width: Target width (if height not specified, maintains aspect)
        height: Target height (if width not specified, maintains aspect)
        max_dimension: Maximum dimension (width or height)

    Returns:
        Resized image as NumPy array
    """
    h, w = image.shape[:2]

    if max_dimension:
        # Scale to fit within max_dimension
        scale = min(max_dimension / w, max_dimension / h)
        if scale < 1:
            width = int(w * scale)
            height = int(h * scale)
        else:
            return image

    elif width and not height:
        # Scale by width, maintain aspect
        height = int(h * width / w)

    elif height and not width:
        # Scale by height, maintain aspect
        width = int(w * height / h)

    elif not width and not height:
        return image

    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
