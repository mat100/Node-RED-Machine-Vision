"""
Image format conversion utilities.

Handles conversions between different image formats using OpenCV:
- NumPy arrays (OpenCV BGR format)
- Base64 encoded strings
- Grayscale/color conversions
"""

import base64
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def to_base64(image: np.ndarray, format: str = "JPEG", quality: int = 85) -> str:
    """
    Convert image to base64 string using OpenCV.

    Args:
        image: Input image as NumPy array (BGR format)
        format: Image format (JPEG, PNG, etc.)
        quality: JPEG quality (1-100, ignored for PNG)

    Returns:
        Base64 encoded string
    """
    try:
        # Handle bytes input
        if isinstance(image, bytes):
            return base64.b64encode(image).decode("utf-8")

        # Prepare encoding parameters
        ext = f".{format.lower()}" if not format.startswith(".") else format.lower()

        if ext in [".jpg", ".jpeg"]:
            params = [cv2.IMWRITE_JPEG_QUALITY, quality, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
        elif ext == ".png":
            # Map quality to PNG compression (0-9 scale, higher = more compression)
            compression = 9 - int(quality / 11)
            params = [cv2.IMWRITE_PNG_COMPRESSION, max(0, min(9, compression))]
        else:
            params = []

        # Encode image to bytes
        success, buffer = cv2.imencode(ext, image, params)

        if not success:
            raise ValueError(f"Failed to encode image to {format}")

        return base64.b64encode(buffer).decode("utf-8")

    except Exception as e:
        logger.error(f"Failed to convert image to base64: {e}")
        raise


def from_base64(base64_string: str) -> np.ndarray:
    """
    Convert base64 string to NumPy array using OpenCV.

    Args:
        base64_string: Base64 encoded image

    Returns:
        NumPy array in BGR format (OpenCV)
    """
    try:
        # Decode base64
        image_bytes = base64.b64decode(base64_string)

        # Convert bytes to NumPy array
        nparr = np.frombuffer(image_bytes, np.uint8)

        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to decode base64 image")

        return image

    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        raise


def encode_image_to_base64(image: np.ndarray, format: str = ".png") -> str:
    """
    Encode OpenCV image (NumPy array) to base64 string.

    This is a simpler alternative to to_base64() specifically for OpenCV images.

    Args:
        image: OpenCV image (NumPy array)
        format: Image format ('.png', '.jpg', etc.)

    Returns:
        Base64 encoded string
    """
    try:
        _, buffer = cv2.imencode(format, image)
        return base64.b64encode(buffer).decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to encode image to base64: {e}")
        raise


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """
    Ensure image is in BGR format (convert from grayscale if needed).

    Args:
        image: Input image (grayscale or BGR)

    Returns:
        Image in BGR format
    """
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Ensure image is grayscale (convert from BGR if needed).

    Args:
        image: Input image (grayscale or BGR)

    Returns:
        Grayscale image
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()
