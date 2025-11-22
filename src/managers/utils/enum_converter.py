"""
Enum conversion utilities.

Provides standardized methods for converting between enums and strings,
with support for case-insensitive parsing and fallback defaults.
"""

from typing import Any, Type, TypeVar

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
        >>> method = EnumConverter.parse_enum(
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
        >>> dictionary_str = EnumConverter.enum_to_string(ArucoDict.DICT_4X4_50)
        >>> # Returns "DICT_4X4_50"
    """
    return value.value if hasattr(value, "value") else value
