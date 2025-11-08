"""
Base schemas for detection parameters.

Provides base classes that eliminate code duplication across
all detection parameter models.
"""

from typing import Any, Dict

from pydantic import BaseModel


class BaseDetectionParams(BaseModel):
    """
    Base class for all detection parameter models.

    Provides common functionality including:
    - to_dict() method with enum conversion
    - Consistent configuration
    - Standardized serialization

    All detection parameter classes should inherit from this base class
    to ensure consistent behavior across the vision system.
    """

    class Config:
        """Pydantic configuration."""

        extra = "forbid"  # Reject unknown fields
        use_enum_values = False  # Keep enums as enum instances

    def to_dict(self) -> Dict[str, Any]:
        """
        Export parameters to dictionary for detector functions.

        Automatically converts enum values to strings, which is required
        by most OpenCV and vision processing functions.

        Returns:
            Dictionary with enum values converted to strings

        Example:
            >>> params = EdgeDetectionParams(method=EdgeMethod.CANNY)
            >>> data = params.to_dict()
            >>> # Returns {"method": "canny", ...}
        """
        data = self.dict(exclude_none=True)

        # Convert enum values to strings
        for key, value in data.items():
            if hasattr(value, "value"):
                data[key] = value.value

        return data

    def get_param(self, name: str, default: Any = None) -> Any:
        """
        Safely get a parameter value.

        Args:
            name: Parameter name
            default: Default value if parameter doesn't exist

        Returns:
            Parameter value or default

        Example:
            >>> threshold = params.get_param("threshold", 100)
        """
        return getattr(self, name, default)
