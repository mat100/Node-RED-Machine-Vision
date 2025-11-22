"""
Reference frame schemas for ArUco-based coordinate transformation.

This module defines configuration and output schemas for single-marker and
multi-marker plane reference systems used to transform image coordinates
into real-world measurements.
"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field, validator


class SingleConfig(BaseModel):
    """
    Configuration for single marker reference mode.

    Uses one ArUco marker with known physical size to establish a reference
    coordinate system. Suitable for perpendicular camera views where uniform
    scaling is acceptable.
    """

    marker_id: int = Field(
        description="ArUco marker ID to use as reference",
        ge=0,
    )

    marker_size_mm: float = Field(
        description="Physical marker size in millimeters (outer edge to outer edge)",
        gt=0,
    )

    origin: str = Field(
        default="marker_center",
        description=(
            "Origin point for coordinate system: "
            "marker_center | marker_top_left | marker_bottom_left"
        ),
    )

    rotation_reference: str = Field(
        default="marker_rotation",
        description=(
            "Rotation reference: "
            "marker_rotation (0° = marker at 0°) | image_axes (0° = horizontal right)"
        ),
    )

    @validator("origin")
    def validate_origin(cls, v):
        valid_origins = ["marker_center", "marker_top_left", "marker_bottom_left"]
        if v not in valid_origins:
            raise ValueError(f"origin must be one of {valid_origins}")
        return v

    @validator("rotation_reference")
    def validate_rotation_reference(cls, v):
        valid_refs = ["marker_rotation", "image_axes"]
        if v not in valid_refs:
            raise ValueError(f"rotation_reference must be one of {valid_refs}")
        return v


class PlaneConfig(BaseModel):
    """
    Configuration for 4-marker plane reference mode.

    Uses four ArUco markers at known positions to establish a reference plane
    with perspective transformation. Robust to camera angle and position changes.
    """

    marker_ids: Dict[str, int] = Field(
        description=(
            "Marker IDs for plane corners. Must contain keys: "
            "top_left, top_right, bottom_right, bottom_left"
        ),
    )

    width_mm: float = Field(
        description="Plane width in millimeters (measured between marker centers)",
        gt=0,
    )

    height_mm: float = Field(
        description="Plane height in millimeters (measured between marker centers)",
        gt=0,
    )

    origin: str = Field(
        default="top_left",
        description=(
            "Which corner is the origin (0,0): " "top_left | top_right | bottom_left | bottom_right"
        ),
    )

    x_direction: str = Field(
        default="right",
        description="Direction of X axis: right | left | up | down",
    )

    y_direction: str = Field(
        default="down",
        description="Direction of Y axis: right | left | up | down",
    )

    @validator("marker_ids")
    def validate_marker_ids(cls, v):
        required_keys = {"top_left", "top_right", "bottom_right", "bottom_left"}
        if set(v.keys()) != required_keys:
            raise ValueError(f"marker_ids must contain exactly these keys: {required_keys}")

        # Check all IDs are unique
        ids = list(v.values())
        if len(ids) != len(set(ids)):
            raise ValueError("All marker IDs must be unique")

        return v

    @validator("origin")
    def validate_origin(cls, v):
        valid_origins = ["top_left", "top_right", "bottom_left", "bottom_right"]
        if v not in valid_origins:
            raise ValueError(f"origin must be one of {valid_origins}")
        return v

    @validator("x_direction", "y_direction")
    def validate_direction(cls, v):
        valid_directions = ["right", "left", "up", "down"]
        if v not in valid_directions:
            raise ValueError(f"direction must be one of {valid_directions}")
        return v


class ReferenceObject(BaseModel):
    """
    Universal reference object for coordinate transformation.

    Contains homography matrix and metadata for transforming image coordinates
    to reference frame coordinates. Supports multiple reference types (single
    marker, plane, checkerboard, etc.).
    """

    type: str = Field(
        description="Reference type: single_marker | plane | checkerboard | custom",
    )

    units: str = Field(
        description="Units of transformed coordinates: pixels | mm | cm | m",
    )

    homography_matrix: List[List[float]] = Field(
        description="3x3 homography transformation matrix",
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Type-specific metadata (marker IDs, dimensions, etc.)",
    )

    thumbnail: str = Field(
        description=(
            "Base64-encoded JPEG visualization of reference plane showing: "
            "detected markers with IDs, coordinate axes (X/Y), grid in physical units, "
            "and origin point [0,0]. Format: data:image/jpeg;base64,..."
        ),
    )

    @validator("homography_matrix")
    def validate_homography(cls, v):
        if len(v) != 3:
            raise ValueError("Homography matrix must have 3 rows")
        for row in v:
            if len(row) != 3:
                raise ValueError("Each row of homography matrix must have 3 columns")
        return v

    @validator("type")
    def validate_type(cls, v):
        valid_types = ["single_marker", "plane", "checkerboard", "custom"]
        if v not in valid_types:
            raise ValueError(f"type must be one of {valid_types}")
        return v

    @validator("units")
    def validate_units(cls, v):
        valid_units = ["pixels", "mm", "cm", "m"]
        if v not in valid_units:
            raise ValueError(f"units must be one of {valid_units}")
        return v
