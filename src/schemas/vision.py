"""
Vision processing API models.

This module contains request models for various vision detection operations:
- Template matching
- Edge detection
- Color detection
- ArUco marker detection
- Rotation detection
"""

from typing import List, Optional

from pydantic import BaseModel, Field, validator

from schemas.params import (
    ArucoDetectionParams,
    ColorDetectionParams,
    EdgeDetectionParams,
    RotationDetectionParams,
    TemplateMatchParams,
)

from .common import ROI


class TemplateMatchRequest(BaseModel):
    """Request for template matching"""

    image_id: str
    roi: Optional[ROI] = Field(None, description="Region of interest to limit search area")
    params: TemplateMatchParams = Field(
        description="Template matching parameters (template_id is required)"
    )


class EdgeDetectRequest(BaseModel):
    """Request for edge detection"""

    image_id: str
    roi: Optional[ROI] = Field(None, description="Region of interest to limit search area")
    params: Optional[EdgeDetectionParams] = Field(
        None,
        description=("Edge detection parameters (method, filtering, preprocessing)"),
    )


class ColorDetectRequest(BaseModel):
    """Request for color detection"""

    image_id: str = Field(..., description="ID of the image to analyze")
    roi: Optional[ROI] = Field(None, description="Region of interest (if None, analyze full image)")
    expected_color: Optional[str] = Field(
        None,
        description=("Expected color name (red, blue, green, etc.) " "or None to just detect"),
    )
    contour: Optional[List] = Field(
        None, description="Contour points for masking (from edge detection)"
    )
    params: Optional[ColorDetectionParams] = Field(
        None,
        description=(
            "Color detection parameters (method, thresholds, "
            "kmeans settings, defaults applied if None)"
        ),
    )


class ArucoDetectRequest(BaseModel):
    """Request for ArUco marker detection"""

    image_id: str = Field(..., description="ID of the image to analyze")
    roi: Optional[ROI] = Field(None, description="Region of interest to search in")
    params: Optional[ArucoDetectionParams] = Field(
        None,
        description=(
            "ArUco detection parameters (dictionary type, "
            "detector settings, defaults applied if None)"
        ),
    )


class RotationDetectRequest(BaseModel):
    """Request for rotation detection"""

    image_id: str = Field(..., description="ID of the image for visualization")
    contour: List = Field(
        ...,
        description="Contour points [[x1,y1], [x2,y2], ...] (minimum 5 points required)",
    )
    roi: Optional[ROI] = Field(None, description="Optional ROI for visualization context")
    params: Optional[RotationDetectionParams] = Field(
        None,
        description=(
            "Rotation detection parameters " "(method, angle range, defaults applied if None)"
        ),
    )

    @validator("contour")
    def validate_contour(cls, v):
        """Validate contour has minimum required points."""
        if len(v) < 5:
            raise ValueError(
                f"Contour must have at least 5 points for rotation detection, got {len(v)}"
            )
        return v
