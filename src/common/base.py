"""
Base data models - fundamental types without dependencies.

This module contains basic Pydantic models used throughout the system:
- Point: 2D point with x, y coordinates
- ROI: Region of Interest with geometric operations

IMPORTANT: This module must NOT import from schemas, core, services, or api
to avoid circular dependencies.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class Point(BaseModel):
    """2D Point"""

    x: float
    y: float


class ROI(BaseModel):
    """
    Region of Interest - unified implementation.

    Represents a rectangular region in an image with utility methods for
    geometric operations, validation, and conversions.
    """

    x: int = Field(..., ge=0, description="X coordinate")
    y: int = Field(..., ge=0, description="Y coordinate")
    width: int = Field(..., gt=0, description="Width")
    height: int = Field(..., gt=0, description="Height")

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary for service layer compatibility."""
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ROI":
        """Create ROI from dictionary."""
        return cls(
            x=int(data.get("x", 0)),
            y=int(data.get("y", 0)),
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
        )

    @classmethod
    def from_points(cls, x1: int, y1: int, x2: int, y2: int) -> "ROI":
        """Create ROI from two corner points."""
        return cls(x=min(x1, x2), y=min(y1, y2), width=abs(x2 - x1), height=abs(y2 - y1))

    @property
    def x2(self) -> int:
        """Get right edge coordinate."""
        return self.x + self.width

    @property
    def y2(self) -> int:
        """Get bottom edge coordinate."""
        return self.y + self.height

    @property
    def center_point(self) -> tuple[int, int]:
        """Get center point of ROI as (x, y) tuple."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area_pixels(self) -> int:
        """Get area of ROI in pixels."""
        return self.width * self.height

    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is inside ROI."""
        return self.x <= x < self.x2 and self.y <= y < self.y2

    def intersects(self, other: "ROI") -> bool:
        """Check if this ROI intersects with another."""
        return not (
            self.x2 <= other.x or other.x2 <= self.x or self.y2 <= other.y or other.y2 <= self.y
        )

    def intersection(self, other: "ROI") -> Optional["ROI"]:
        """Get intersection with another ROI, or None if no intersection."""
        if not self.intersects(other):
            return None

        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)

        return ROI.from_points(x1, y1, x2, y2)

    def union(self, other: "ROI") -> "ROI":
        """Get union with another ROI (smallest rectangle containing both)."""
        x1 = min(self.x, other.x)
        y1 = min(self.y, other.y)
        x2 = max(self.x2, other.x2)
        y2 = max(self.y2, other.y2)

        return ROI.from_points(x1, y1, x2, y2)

    def scale(self, factor: float, from_center: bool = False) -> "ROI":
        """
        Scale ROI by factor.

        Args:
            factor: Scale factor (e.g., 1.5 for 150%)
            from_center: If True, scale from center; if False, scale from top-left

        Returns:
            New scaled ROI
        """
        new_width = int(self.width * factor)
        new_height = int(self.height * factor)

        if from_center:
            # Scale from center
            cx, cy = self.center_point
            new_x = cx - new_width // 2
            new_y = cy - new_height // 2
        else:
            # Scale from top-left
            new_x = self.x
            new_y = self.y

        return ROI(x=new_x, y=new_y, width=new_width, height=new_height)

    def expand(self, pixels: int) -> "ROI":
        """Expand ROI by pixels in all directions."""
        return ROI(
            x=self.x - pixels,
            y=self.y - pixels,
            width=self.width + 2 * pixels,
            height=self.height + 2 * pixels,
        )

    def clip(self, image_width: int, image_height: int) -> "ROI":
        """
        Clip ROI to image bounds.

        Args:
            image_width: Maximum width (image width)
            image_height: Maximum height (image height)

        Returns:
            Clipped ROI that fits within image bounds
        """
        x = max(0, min(self.x, image_width))
        y = max(0, min(self.y, image_height))
        x2 = max(0, min(self.x2, image_width))
        y2 = max(0, min(self.y2, image_height))

        return ROI.from_points(x, y, x2, y2)

    def is_valid(
        self, image_width: Optional[int] = None, image_height: Optional[int] = None
    ) -> bool:
        """
        Check if ROI is valid.

        Args:
            image_width: Optional image width for bounds checking
            image_height: Optional image height for bounds checking

        Returns:
            True if ROI is valid
        """
        # Basic validation (Pydantic already ensures width/height > 0 and x/y >= 0)
        # But we check again for runtime safety
        if self.width <= 0 or self.height <= 0:
            return False

        if self.x < 0 or self.y < 0:
            return False

        # Image bounds validation if provided
        if image_width is not None and self.x2 > image_width:
            return False

        if image_height is not None and self.y2 > image_height:
            return False

        return True

    def validate_with_constraints(
        self,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        min_size: int = 1,
        max_size: Optional[int] = None,
    ) -> None:
        """
        Validate ROI with size constraints.

        Args:
            image_width: Optional image width for bounds checking
            image_height: Optional image height for bounds checking
            min_size: Minimum width/height (default: 1)
            max_size: Optional maximum width/height

        Raises:
            ValueError: If ROI is invalid with descriptive error message
        """
        # Check basic validity (coordinates and dimensions)
        if self.width < min_size or self.height < min_size:
            raise ValueError(f"ROI too small: {self.width}x{self.height} (min: {min_size})")

        if max_size and (self.width > max_size or self.height > max_size):
            raise ValueError(f"ROI too large: {self.width}x{self.height} (max: {max_size})")

        if self.x < 0 or self.y < 0:
            raise ValueError(f"ROI has negative coordinates: ({self.x}, {self.y})")

        # Image bounds validation if provided
        if image_width is not None or image_height is not None:
            if not self.is_valid(image_width, image_height):
                raise ValueError(
                    f"ROI {self.to_dict()} exceeds image bounds {image_width}x{image_height}"
                )
