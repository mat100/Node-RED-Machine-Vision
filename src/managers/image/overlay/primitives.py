"""
Basic primitive drawing functions for vision overlays.

This module provides low-level drawing utilities that are used by
higher-level rendering functions. All functions modify images in-place
for performance and return the modified image for convenience.
"""

from typing import Tuple

import cv2
import numpy as np

# Default colors (BGR format)
COLOR_SUCCESS = (0, 255, 0)  # Green
COLOR_FAILURE = (0, 0, 255)  # Red
COLOR_INFO = (255, 255, 0)  # Cyan

# Default rendering parameters
DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX
DEFAULT_FONT_SCALE = 0.5
DEFAULT_THICKNESS = 2
DEFAULT_LINE_TYPE = cv2.LINE_AA


def draw_bounding_box(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    color: Tuple[int, int, int] = COLOR_SUCCESS,
    thickness: int = DEFAULT_THICKNESS,
    line_type: int = DEFAULT_LINE_TYPE,
) -> np.ndarray:
    """
    Draw a bounding box on the image.

    Args:
        image: Input image
        x: Top-left x coordinate
        y: Top-left y coordinate
        width: Box width
        height: Box height
        color: Box color in BGR format
        thickness: Line thickness
        line_type: Line type for anti-aliasing

    Returns:
        Image with bounding box drawn
    """
    pt1 = (x, y)
    pt2 = (x + width, y + height)
    cv2.rectangle(image, pt1, pt2, color, thickness, line_type)
    return image


def draw_label(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: Tuple[int, int, int] = COLOR_SUCCESS,
    background: bool = False,
    font: int = DEFAULT_FONT,
    font_scale: float = DEFAULT_FONT_SCALE,
    thickness: int = DEFAULT_THICKNESS,
    line_type: int = DEFAULT_LINE_TYPE,
) -> np.ndarray:
    """
    Draw text label on the image.

    Args:
        image: Input image
        text: Text to draw
        x: Text x coordinate
        y: Text y coordinate
        color: Text color in BGR format
        background: Whether to draw background rectangle
        font: OpenCV font type
        font_scale: Font scale factor
        thickness: Line thickness
        line_type: Line type for anti-aliasing

    Returns:
        Image with label drawn
    """
    if background:
        # Calculate text size for background
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        # Draw background rectangle
        cv2.rectangle(
            image,
            (x, y - text_height - baseline),
            (x + text_width, y),
            color,
            -1,  # Filled
        )
        # Draw text in white over background
        text_color = (255, 255, 255)
    else:
        text_color = color

    cv2.putText(
        image,
        text,
        (x, y),
        font,
        font_scale,
        text_color,
        thickness,
        line_type,
    )
    return image


def draw_confidence(
    image: np.ndarray,
    confidence: float,
    x: int,
    y: int,
    color: Tuple[int, int, int] = COLOR_SUCCESS,
    font: int = DEFAULT_FONT,
    font_scale: float = DEFAULT_FONT_SCALE,
    thickness: int = DEFAULT_THICKNESS,
    line_type: int = DEFAULT_LINE_TYPE,
) -> np.ndarray:
    """
    Draw confidence score above bounding box.

    Args:
        image: Input image
        confidence: Confidence value (0.0-1.0)
        x: X coordinate
        y: Y coordinate (top of bounding box)
        color: Text color
        font: OpenCV font type
        font_scale: Font scale factor
        thickness: Line thickness
        line_type: Line type for anti-aliasing

    Returns:
        Image with confidence drawn
    """
    text = f"{confidence:.2f}"
    return draw_label(image, text, x, y - 5, color, False, font, font_scale, thickness, line_type)


def draw_center_point(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    color: Tuple[int, int, int] = COLOR_SUCCESS,
    radius: int = 3,
    line_type: int = DEFAULT_LINE_TYPE,
) -> np.ndarray:
    """
    Draw center point marker.

    Args:
        image: Input image
        center_x: Center x coordinate
        center_y: Center y coordinate
        color: Marker color
        radius: Circle radius
        line_type: Line type for anti-aliasing

    Returns:
        Image with center point drawn
    """
    cv2.circle(
        image,
        (int(center_x), int(center_y)),
        radius,
        color,
        -1,  # Filled
        line_type,
    )
    return image


def draw_contour(
    image: np.ndarray,
    contour: np.ndarray,
    color: Tuple[int, int, int] = COLOR_INFO,
    thickness: int = DEFAULT_THICKNESS,
    line_type: int = DEFAULT_LINE_TYPE,
) -> np.ndarray:
    """
    Draw a contour on the image.

    Args:
        image: Input image
        contour: Contour points (OpenCV format)
        color: Contour color in BGR format
        thickness: Line thickness
        line_type: Line type for anti-aliasing

    Returns:
        Image with contour drawn
    """
    cv2.drawContours(image, [contour.astype(np.int32)], -1, color, thickness, line_type)
    return image


def draw_rotation_indicator(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    angle_deg: float,
    length: int = 50,
    color: Tuple[int, int, int] = (0, 165, 255),  # Orange
    thickness: int = DEFAULT_THICKNESS,
    line_type: int = DEFAULT_LINE_TYPE,
    arrow_tip_length: float = 0.3,
) -> np.ndarray:
    """
    Draw rotation indicator arrow from center showing angle.

    Args:
        image: Input image
        center_x: Center x coordinate
        center_y: Center y coordinate
        angle_deg: Rotation angle in degrees
        length: Arrow length in pixels
        color: Arrow color
        thickness: Line thickness
        line_type: Line type for anti-aliasing
        arrow_tip_length: Arrow tip size ratio

    Returns:
        Image with rotation arrow drawn
    """
    center = (int(center_x), int(center_y))

    # Calculate end point from angle
    angle_rad = np.radians(angle_deg)
    end_x = int(center_x + length * np.cos(angle_rad))
    end_y = int(center_y + length * np.sin(angle_rad))

    # Draw arrowed line
    cv2.arrowedLine(
        image,
        center,
        (end_x, end_y),
        color,
        thickness,
        line_type,
        tipLength=arrow_tip_length,
    )

    return image
