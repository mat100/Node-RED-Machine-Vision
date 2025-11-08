"""
Template-related API models.

This module contains request and response models for template operations:
- Template information
- Template upload and learning
"""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel

from .common import ROI, Size


class TemplateInfo(BaseModel):
    """Template information"""

    id: str
    name: str
    description: Optional[str] = None
    size: Size
    created_at: datetime

    @classmethod
    def from_manager_dict(cls, data: Dict[str, Any]) -> "TemplateInfo":
        """
        Create TemplateInfo from template manager dict.

        Args:
            data: Dictionary from template manager with template details

        Returns:
            TemplateInfo instance

        Example:
            >>> tmpl_dict = {
            ...     "id": "tmpl_123",
            ...     "name": "Part A",
            ...     "description": "Reference template",
            ...     "size": {"width": 100, "height": 50},
            ...     "created_at": "2025-01-15T10:30:00"
            ... }
            >>> info = TemplateInfo.from_manager_dict(tmpl_dict)
        """
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description"),
            size=Size(width=data["size"]["width"], height=data["size"]["height"]),
            created_at=datetime.fromisoformat(data["created_at"]),
        )


class TemplateUploadResponse(BaseModel):
    """Response from template upload"""

    success: bool
    template_id: str
    name: str
    size: Size


class TemplateLearnRequest(BaseModel):
    """Request to learn template from image"""

    image_id: str
    name: str
    roi: ROI
    description: Optional[str] = None
