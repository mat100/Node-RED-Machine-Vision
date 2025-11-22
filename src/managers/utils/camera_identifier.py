"""
Camera identifier parsing and formatting utilities.

Provides unified camera ID handling to eliminate duplicated string parsing logic.
"""

import logging
from typing import Tuple, Union

logger = logging.getLogger(__name__)

# Camera type constants
TYPE_USB = "usb"
TYPE_TEST = "test"
TYPE_IP = "ip"


def parse(camera_id: str) -> Tuple[str, Union[int, str, None]]:
    """
    Parse camera ID into type and source.

    Args:
        camera_id: Camera identifier string

    Returns:
        Tuple of (camera_type, source)
        - camera_type: "usb", "test", or "ip"
        - source: int for USB index, str for IP address, None for test

    Examples:
        >>> parse("usb_0")
        ("usb", 0)
        >>> parse("usb_1")
        ("usb", 1)
        >>> parse("test")
        ("test", None)
        >>> parse("invalid")
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
