"""
Image processing API models.

This module contains models for image operations:
- ROI extraction requests and responses
- Image import from file system
"""

from datetime import datetime
from typing import Any, Dict

from pydantic import BaseModel, Field

from .common import ROI


class ROIExtractRequest(BaseModel):
    """Request to extract ROI from image"""

    image_id: str
    roi: ROI = Field(..., description="Region of interest to extract")


class ROIExtractResponse(BaseModel):
    """Response from ROI extraction"""

    success: bool
    thumbnail: str
    bounding_box: ROI


class ImageImportRequest(BaseModel):
    """Request to import image from file system"""

    file_path: str = Field(..., description="Path to image file (JPG, PNG, BMP, etc.)")


class ImageImportResponse(BaseModel):
    """Response from image import (same structure as CameraCaptureResponse)"""

    success: bool
    image_id: str
    timestamp: datetime
    thumbnail_base64: str
    metadata: Dict[str, Any]
