"""
Test pattern generation utilities.

Provides utilities for generating test images with various patterns,
ArUco markers, and visual elements for development and testing.
"""

import logging
import time
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def create_gradient_background(width: int, height: int) -> np.ndarray:
    """
    Create an image with gradient background.

    Args:
        width: Image width
        height: Image height

    Returns:
        BGR image with gradient background
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        img[i, :] = [i * 255 // height, 100, 255 - i * 255 // height]
    return img


def add_aruco_marker(
    image: np.ndarray,
    marker_id: int,
    position: Tuple[int, int],
    marker_size: int = 200,
    border_size: int = 50,
    aruco_dict_type: int = cv2.aruco.DICT_4X4_50,
) -> bool:
    """
    Add ArUco marker to image at specified position.

    Args:
        image: Image to add marker to (modified in-place)
        marker_id: ArUco marker ID
        position: (x, y) position for top-left corner
        marker_size: Size of marker in pixels
        border_size: White border around marker
        aruco_dict_type: ArUco dictionary type

    Returns:
        True if marker was added successfully, False otherwise
    """
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        total_size = marker_size + 2 * border_size

        # Check if marker fits in image
        x_pos, y_pos = position
        height, width = image.shape[:2]
        if y_pos + total_size >= height or x_pos + total_size >= width:
            logger.warning(f"ArUco marker doesn't fit at position {position}")
            return False

        # Generate marker
        marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

        # Create white border
        marker_with_border = np.ones((total_size, total_size), dtype=np.uint8) * 255
        marker_with_border[
            border_size : border_size + marker_size, border_size : border_size + marker_size
        ] = marker_image

        # Place marker on image
        image[y_pos : y_pos + total_size, x_pos : x_pos + total_size] = cv2.cvtColor(
            marker_with_border, cv2.COLOR_GRAY2BGR
        )

        return True

    except Exception as e:
        logger.warning(f"Failed to add ArUco marker: {e}")
        return False


def add_text(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float = 1.5,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> None:
    """
    Add text to image.

    Args:
        image: Image to add text to (modified in-place)
        text: Text to add
        position: (x, y) position for text
        font_scale: Font scale factor
        color: Text color (BGR)
        thickness: Text thickness
    """
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
    )


def create_test_image_with_markers(
    width: int = 1920,
    height: int = 1080,
    text: Optional[str] = None,
    add_timestamp: bool = True,
    add_grid: bool = True,
    add_shapes: bool = True,
) -> np.ndarray:
    """
    Create comprehensive test image with ArUco markers and test patterns.

    This is suitable for camera testing and computer vision algorithm validation.

    Args:
        width: Image width
        height: Image height
        text: Optional text to add to image
        add_timestamp: Whether to add timestamp
        add_grid: Whether to add grid overlay
        add_shapes: Whether to add test shapes (rectangles, circles)

    Returns:
        BGR test image
    """
    # Create gradient background
    img = create_gradient_background(width, height)

    # Add ArUco markers
    marker_size = min(200, width // 8)
    border_size = marker_size // 4

    # Marker in top-left
    add_aruco_marker(img, 0, (50, 50), marker_size, border_size)

    # Marker in top-right (if space)
    if width > 800:
        total_size = marker_size + 2 * border_size
        x_pos2 = width - total_size - 50
        add_aruco_marker(img, 5, (x_pos2, 50), marker_size, border_size)

    # Add test shapes for edge/rotation detection
    if add_shapes and width > 1200:
        cv2.rectangle(img, (800, 400), (1000, 600), (255, 100, 100), -1)  # Blue
        cv2.rectangle(img, (1100, 450), (1250, 550), (100, 255, 100), -1)  # Green
        if height > 850:
            cv2.circle(img, (950, 800), 80, (100, 100, 255), -1)  # Red circle

    # Add main text
    if text:
        text_y = height - 130 if add_timestamp else height - 50
        add_text(img, text, (width // 2 - 200, text_y), 2)

    # Add timestamp
    if add_timestamp:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        add_text(img, timestamp, (50, height - 30), 1)

    # Add grid
    if add_grid:
        grid_step_x = width // 10
        grid_step_y = height // 10
        for x in range(0, width, grid_step_x):
            cv2.line(img, (x, 0), (x, height), (50, 50, 50), 1)
        for y in range(0, height, grid_step_y):
            cv2.line(img, (0, y), (width, y), (50, 50, 50), 1)

    return img


def create_simple_test_image(
    width: int,
    height: int,
    text: Optional[str] = None,
) -> np.ndarray:
    """
    Create simple test image with single ArUco marker.

    Suitable for camera preview and basic testing.

    Args:
        width: Image width
        height: Image height
        text: Optional text to add

    Returns:
        BGR test image
    """
    # Create gradient background
    img = create_gradient_background(width, height)

    # Add single ArUco marker
    marker_size = min(200, width // 8)
    border_size = marker_size // 4
    add_aruco_marker(img, 0, (50, 50), marker_size, border_size)

    # Add text if provided
    if text:
        text_x = width // 2 - len(text) * 10
        text_y = height - 50
        add_text(img, text, (text_x, text_y), 1.5)

    return img
