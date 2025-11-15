"""
Common models shared across the API.

This module contains fundamental data models used throughout the vision system:
- VisionObject for detection results
- VisionResponse for unified API responses
- Size for image dimensions

Note: Point and ROI have been moved to common.base to avoid circular dependencies.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Import base types from common package (no circular dependency)
from common.base import ROI, Point
from schemas.reference import ReferenceObject


class Size(BaseModel):
    """Image size"""

    width: int
    height: int


class VisionObject(BaseModel):
    """
    Universal interface for any vision processing object.
    (camera capture, contour, template, color region, etc.)
    Provides standardized location, geometry, and quality information.
    """

    # Identification
    object_id: str = Field(..., description="Unique ID of this object")
    object_type: str = Field(
        ...,
        description="Type: camera_capture, edge_contour, template_match, etc.",
    )

    # Position & Geometry
    bounding_box: ROI = Field(
        ..., description="Bounding box of detected object in {x, y, width, height} format"
    )
    center: Point = Field(..., description="Center point of the object")

    # Quality
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0 - 1.0)")

    # Optional geometry
    area: Optional[float] = Field(None, description="Area in pixels")
    perimeter: Optional[float] = Field(None, description="Perimeter in pixels")
    rotation: Optional[float] = Field(None, description="Rotation in degrees (0-360)")

    # Plane coordinates (when reference frame is applied)
    plane_position: Optional[Point] = Field(
        None, description="Position in reference plane coordinates (units from reference_object)"
    )
    plane_rotation: Optional[float] = Field(
        None, description="Rotation in reference plane (degrees, when reference applied)"
    )

    # Type-specific properties
    properties: Dict[str, Any] = Field(default_factory=dict, description="Type-specific properties")

    # Raw data (optional)
    contour: Optional[List] = Field(None, description="Contour points for edge detection")


class VisionResponse(BaseModel):
    """
    Simplified response for all vision processing APIs.

    Contains detected objects, visualization thumbnail, and optional reference
    frame for coordinate transformation (when using single/plane detection modes).
    """

    objects: List[VisionObject] = Field(default_factory=list, description="List of vision objects")
    thumbnail_base64: str = Field(..., description="Base64-encoded thumbnail with visualization")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    reference_object: Optional[ReferenceObject] = Field(
        None,
        description=(
            "Reference frame for coordinate transformation (present when using "
            "ArUco single or plane detection modes)"
        ),
    )


class ArucoReferenceResponse(BaseModel):
    """
    Response for ArUco reference frame creation.

    Contains the created reference object, detected markers used for calibration,
    visualization thumbnail, and processing time.
    """

    reference_object: ReferenceObject = Field(
        ...,
        description=(
            "Reference frame created from ArUco markers. Contains homography matrix "
            "and metadata for transforming coordinates to real-world units."
        ),
    )
    markers: List[VisionObject] = Field(
        ...,
        description="Detected ArUco markers used to create the reference frame",
    )
    thumbnail_base64: str = Field(
        ..., description="Base64-encoded thumbnail with markers visualized"
    )
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
