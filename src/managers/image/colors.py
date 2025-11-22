"""
Color definitions for automatic color detection.

Defines standard color ranges in HSV color space for robust color matching.
"""

from typing import Dict, List, Optional

import numpy as np

# Standard color definitions in HSV space
# Format: {color_name: {hue_ranges, sat_min, sat_max, val_min, val_max}}
COLOR_DEFINITIONS: Dict[str, Dict] = {
    # Chromatic colors (have hue)
    "red": {
        "hue_ranges": [(0, 15), (165, 180)],  # Red wraps around the hue circle
        "sat_min": 100,
        "sat_max": 255,
        "val_min": 80,
        "val_max": 255,
    },
    "orange": {
        "hue_ranges": [(16, 30)],
        "sat_min": 100,
        "sat_max": 255,
        "val_min": 80,
        "val_max": 255,
    },
    "yellow": {
        "hue_ranges": [(31, 45)],
        "sat_min": 100,
        "sat_max": 255,
        "val_min": 100,
        "val_max": 255,
    },
    "green": {
        "hue_ranges": [(46, 90)],
        "sat_min": 100,
        "sat_max": 255,
        "val_min": 80,
        "val_max": 255,
    },
    "cyan": {
        "hue_ranges": [(91, 110)],
        "sat_min": 100,
        "sat_max": 255,
        "val_min": 80,
        "val_max": 255,
    },
    "blue": {
        "hue_ranges": [(111, 140)],
        "sat_min": 100,
        "sat_max": 255,
        "val_min": 80,
        "val_max": 255,
    },
    "purple": {
        "hue_ranges": [(141, 164)],
        "sat_min": 100,
        "sat_max": 255,
        "val_min": 80,
        "val_max": 255,
    },
    # Achromatic colors (no hue - use saturation and value only)
    "white": {
        "hue_ranges": None,  # Hue is irrelevant
        "sat_min": 0,
        "sat_max": 30,
        "val_min": 200,
        "val_max": 255,
    },
    "black": {
        "hue_ranges": None,
        "sat_min": 0,
        "sat_max": 50,
        "val_min": 0,
        "val_max": 50,
    },
    "gray": {
        "hue_ranges": None,
        "sat_min": 0,
        "sat_max": 30,
        "val_min": 51,
        "val_max": 199,
    },
}


def is_color_match(h: int, s: int, v: int, color_name: str) -> bool:
    """
    Check if HSV values match a given color definition.

    Args:
        h: Hue value (0-179 in OpenCV)
        s: Saturation value (0-255)
        v: Value/brightness (0-255)
        color_name: Name of the color to match against

    Returns:
        True if the HSV values match the color definition

    Raises:
        ValueError: If color_name is not in COLOR_DEFINITIONS
    """
    if color_name not in COLOR_DEFINITIONS:
        raise ValueError(
            f"Unknown color: {color_name}. Available: {list(COLOR_DEFINITIONS.keys())}"
        )

    definition = COLOR_DEFINITIONS[color_name]

    # Check saturation and value first (applies to all colors)
    if not (definition["sat_min"] <= s <= definition["sat_max"]):
        return False
    if not (definition["val_min"] <= v <= definition["val_max"]):
        return False

    # For achromatic colors, saturation and value checks are sufficient
    if definition["hue_ranges"] is None:
        return True

    # For chromatic colors, also check hue
    for hue_min, hue_max in definition["hue_ranges"]:
        if hue_min <= h <= hue_max:
            return True

    return False


def hsv_to_color_name(h: int, s: int, v: int) -> Optional[str]:
    """
    Map HSV values to the best matching color name.

    Args:
        h: Hue value (0-179 in OpenCV)
        s: Saturation value (0-255)
        v: Value/brightness (0-255)

    Returns:
        Color name if a match is found, None otherwise
    """
    # Check achromatic colors first (order matters: white, black, gray)
    if is_color_match(h, s, v, "white"):
        return "white"
    if is_color_match(h, s, v, "black"):
        return "black"
    if is_color_match(h, s, v, "gray"):
        return "gray"

    # Check chromatic colors
    for color_name in ["red", "orange", "yellow", "green", "cyan", "blue", "purple"]:
        if is_color_match(h, s, v, color_name):
            return color_name

    return None


def get_available_colors() -> List[str]:
    """Get list of all available color names."""
    return list(COLOR_DEFINITIONS.keys())


def is_achromatic(color_name: str) -> bool:
    """Check if a color is achromatic (white, black, gray)."""
    if color_name not in COLOR_DEFINITIONS:
        raise ValueError(f"Unknown color: {color_name}")
    return COLOR_DEFINITIONS[color_name]["hue_ranges"] is None


def create_color_mask_vectorized(
    h: np.ndarray, s: np.ndarray, v: np.ndarray, color_name: str
) -> np.ndarray:
    """
    Create a boolean mask for a color using vectorized NumPy operations.

    This is much faster than pixel-by-pixel iteration for large images.

    Args:
        h: Hue channel as 2D NumPy array (0-179)
        s: Saturation channel as 2D NumPy array (0-255)
        v: Value channel as 2D NumPy array (0-255)
        color_name: Name of the color to match

    Returns:
        Boolean mask where True indicates pixels matching the color

    Raises:
        ValueError: If color_name is not in COLOR_DEFINITIONS
    """
    if color_name not in COLOR_DEFINITIONS:
        raise ValueError(
            f"Unknown color: {color_name}. Available: {list(COLOR_DEFINITIONS.keys())}"
        )

    definition = COLOR_DEFINITIONS[color_name]

    # Check saturation and value (vectorized)
    sat_mask = (s >= definition["sat_min"]) & (s <= definition["sat_max"])
    val_mask = (v >= definition["val_min"]) & (v <= definition["val_max"])
    mask = sat_mask & val_mask

    # For achromatic colors, saturation and value are sufficient
    if definition["hue_ranges"] is None:
        return mask

    # For chromatic colors, also check hue
    hue_mask = np.zeros_like(h, dtype=bool)
    for hue_min, hue_max in definition["hue_ranges"]:
        hue_mask |= (h >= hue_min) & (h <= hue_max)

    return mask & hue_mask


def count_colors_vectorized(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> Dict[str, int]:
    """
    Count pixels for each color using vectorized operations.

    This is 10-50x faster than nested Python loops.

    Args:
        h: Hue channel as 2D NumPy array (0-179)
        s: Saturation channel as 2D NumPy array (0-255)
        v: Value channel as 2D NumPy array (0-255)

    Returns:
        Dictionary mapping color names to pixel counts
    """
    color_counts = {}

    # Process achromatic colors first (they have priority)
    for color in ["white", "black", "gray"]:
        mask = create_color_mask_vectorized(h, s, v, color)
        color_counts[color] = int(np.sum(mask))

    # Create combined achromatic mask to exclude from chromatic colors
    achromatic_mask = np.zeros_like(h, dtype=bool)
    for color in ["white", "black", "gray"]:
        achromatic_mask |= create_color_mask_vectorized(h, s, v, color)

    # Process chromatic colors (excluding achromatic pixels)
    for color in ["red", "orange", "yellow", "green", "cyan", "blue", "purple"]:
        mask = create_color_mask_vectorized(h, s, v, color) & ~achromatic_mask
        color_counts[color] = int(np.sum(mask))

    return color_counts
