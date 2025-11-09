"""
Image processing API models.

This module contains models for image operations:
- ROI extraction requests and responses
- Image import from file system
"""

from pydantic import BaseModel, Field

from .common import ROI


class ROIExtractRequest(BaseModel):
    """Request to extract ROI from image"""

    image_id: str
    roi: ROI = Field(..., description="Region of interest to extract")


class ImageImportRequest(BaseModel):
    """Request to import image from file system"""

    file_path: str = Field(..., description="Path to image file (JPG, PNG, BMP, etc.)")
