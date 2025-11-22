"""
Utility functions for domain-agnostic operations.

This module consolidates all utility functions used throughout the system:
- Camera identifier: Parse camera ID strings
- Decorators: Utility decorators (timer, etc.)
- Enum converter: Enum parsing and conversion

All utilities are pure functions with no dependencies on other project modules.
"""

import logging
import time
from contextlib import contextmanager
from typing import Any, Generator, Tuple, Type, TypeVar, Union

logger = logging.getLogger(__name__)

# ==============================================================================
# Camera Identifier Utilities
# ==============================================================================

# Camera type constants
TYPE_USB = "usb"
TYPE_TEST = "test"
TYPE_IP = "ip"


def parse_camera_id(camera_id: str) -> Tuple[str, Union[int, str, None]]:
    """
    Parse camera ID into type and source.

    Args:
        camera_id: Camera identifier string

    Returns:
        Tuple of (camera_type, source)
        - camera_type: "usb", "test", or "ip"
        - source: int for USB index, str for IP address, None for test

    Examples:
        >>> parse_camera_id("usb_0")
        ("usb", 0)
        >>> parse_camera_id("usb_1")
        ("usb", 1)
        >>> parse_camera_id("test")
        ("test", None)
        >>> parse_camera_id("invalid")
        ("usb", 0)  # Default fallback
    """
    if not camera_id:
        logger.warning("Empty camera_id provided, defaulting to usb_0")
        return (TYPE_USB, 0)

    # Test camera
    if camera_id == TYPE_TEST:
        return (TYPE_TEST, None)

    # USB camera
    if camera_id.startswith(f"{TYPE_USB}_"):
        try:
            source = int(camera_id.split("_")[1])
            return (TYPE_USB, source)
        except (IndexError, ValueError) as e:
            logger.warning(f"Invalid USB camera ID '{camera_id}': {e}, defaulting to usb_0")
            return (TYPE_USB, 0)

    # IP camera (future)
    if camera_id.startswith(f"{TYPE_IP}_"):
        try:
            ip_address = camera_id.split("_", 1)[1]
            return (TYPE_IP, ip_address)
        except IndexError as e:
            logger.warning(f"Invalid IP camera ID '{camera_id}': {e}")
            return (TYPE_USB, 0)

    # Unknown format - default to USB 0
    logger.warning(f"Unknown camera ID format '{camera_id}', defaulting to usb_0")
    return (TYPE_USB, 0)


# ==============================================================================
# Decorators
# ==============================================================================


@contextmanager
def timer() -> Generator[dict, None, None]:
    """
    Context manager to measure execution time.

    Usage:
        with timer() as t:
            # ... code to time ...
            pass
        print(f"Took {t['ms']}ms")

    Yields:
        Dictionary with 'ms' key containing processing time in milliseconds
    """
    result = {"ms": 0}
    start_time = time.time()
    try:
        yield result
    finally:
        result["ms"] = int((time.time() - start_time) * 1000)


# ==============================================================================
# Enum Converter Utilities
# ==============================================================================

T = TypeVar("T")


def parse_enum(value: Any, enum_class: Type[T], default: T, normalize: bool = False) -> T:
    """
    Parse value to enum with fallback to default.

    Unifies enum parsing logic across all detection methods.

    Args:
        value: Value to parse (string, enum, or None)
        enum_class: Enum class to parse to
        default: Default enum value if parsing fails
        normalize: Whether to lowercase string before parsing
            (for case-insensitive matching)

    Returns:
        Parsed enum value or default

    Example:
        >>> method = parse_enum(
        ...     "CANNY", EdgeMethod, EdgeMethod.CANNY, normalize=True
        ... )
        >>> # Returns EdgeMethod.CANNY for "canny", "Canny", "CANNY"
    """
    # Already an enum instance
    if isinstance(value, enum_class):
        return value

    # None or missing value
    if value is None:
        return default

    # String value - try to parse
    try:
        str_value = value.lower() if normalize else value
        return enum_class(str_value)
    except (ValueError, AttributeError):
        return default


def enum_to_string(value: Any) -> str:
    """
    Convert enum to string value, or pass through if already string.

    Unifies enum → string conversion across all detection methods.

    Args:
        value: Enum instance or string

    Returns:
        String value (enum.value if enum, otherwise the value itself)

    Example:
        >>> dictionary_str = enum_to_string(ArucoDict.DICT_4X4_50)
        >>> # Returns "DICT_4X4_50"
    """
    return value.value if hasattr(value, "value") else value
