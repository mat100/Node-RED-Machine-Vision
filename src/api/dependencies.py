"""
Shared FastAPI dependencies for the Machine Vision Flow system.
Centralizes common dependencies to eliminate code duplication.
"""

import logging
from functools import lru_cache
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer

from core.camera_manager import CameraManager
from core.image_manager import ImageManager
from core.template_manager import TemplateManager
from schemas import ROI

# Import services
from services.camera_service import CameraService
from services.image_service import ImageService
from services.vision_service import VisionService

logger = logging.getLogger(__name__)

# Optional security bearer for future use
security = HTTPBearer(auto_error=False)


class Managers:
    """Container for all manager instances."""

    def __init__(
        self,
        image_manager: ImageManager,
        camera_manager: CameraManager,
        template_manager: TemplateManager,
    ):
        self.image_manager = image_manager
        self.camera_manager = camera_manager
        self.template_manager = template_manager


def get_managers(request: Request) -> Managers:
    """
    Get all manager instances from app state.

    Args:
        request: FastAPI request object

    Returns:
        Managers container with all manager instances

    Raises:
        HTTPException: If managers not initialized
    """
    try:
        return Managers(
            image_manager=request.app.state.image_manager,
            camera_manager=request.app.state.camera_manager,
            template_manager=request.app.state.template_manager,
        )
    except AttributeError as e:
        logger.error(f"Managers not initialized in app state: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error: Managers not initialized"
        )


def get_image_manager(managers: Managers = Depends(get_managers)) -> ImageManager:
    """Get ImageManager instance."""
    return managers.image_manager


def get_camera_manager(managers: Managers = Depends(get_managers)) -> CameraManager:
    """Get CameraManager instance."""
    return managers.camera_manager


def get_template_manager(managers: Managers = Depends(get_managers)) -> TemplateManager:
    """Get TemplateManager instance."""
    return managers.template_manager


def validate_image_exists(
    image_id: str,
    image_manager: ImageManager,
) -> str:
    """
    Validate that image exists in shared memory.

    Args:
        image_id: Image identifier
        image_manager: ImageManager instance

    Returns:
        Validated image_id

    Raises:
        HTTPException: If image not found
    """
    # ImageManager.get() returns None if image doesn't exist
    if image_manager.get(image_id) is None:
        raise HTTPException(status_code=404, detail=f"Image {image_id} not found in cache")
    return image_id


def validate_roi_bounds(
    roi: ROI,
    image_id: str,
    image_manager: ImageManager,
) -> ROI:
    """
    Validate that ROI fits within image bounds.

    Args:
        roi: ROI to validate
        image_id: Image identifier
        image_manager: ImageManager instance

    Returns:
        Validated ROI

    Raises:
        HTTPException: If ROI exceeds image bounds or image not found
    """
    # Get image to check dimensions
    image_data = image_manager.get(image_id)
    if image_data is None:
        raise HTTPException(status_code=404, detail=f"Image {image_id} not found")

    img_height, img_width = image_data.shape[:2]

    # Validate ROI against image dimensions
    if not roi.is_valid(img_width, img_height):
        raise HTTPException(
            status_code=400,
            detail=f"ROI {roi.to_dict()} exceeds image bounds ({img_width}x{img_height})",
        )

    return roi


def roi_to_dict(roi: Optional[ROI]) -> Optional[Dict[str, int]]:
    """
    Convert Pydantic ROI model to dict for service layer compatibility.

    This helper eliminates the repeated manual dict construction pattern
    found throughout the vision routers.

    Args:
        roi: Optional ROI model

    Returns:
        ROI as dict or None if roi is None
    """
    if roi is None:
        return None
    return roi.to_dict()


def validate_vision_request(
    image_id: str,
    roi: Optional[ROI],
    image_manager: ImageManager,
):
    """
    Unified validation for vision detection endpoints.

    Eliminates the repeated pattern of:
    - validate_image_exists()
    - validate_roi_bounds()
    - roi_to_dict()

    Args:
        image_id: Image identifier to validate
        roi: Optional ROI to validate
        image_manager: ImageManager instance

    Returns:
        ROI as dict, or None if roi is None

    Raises:
        HTTPException: If image not found or ROI invalid
    """
    # Validate image exists
    validate_image_exists(image_id, image_manager)

    # Validate ROI if provided
    if roi:
        validate_roi_bounds(roi, image_id, image_manager)

    # Convert ROI to dict
    return roi_to_dict(roi)


# Authentication and rate limiting removed - implement when needed
# For future implementation, consider using:
# - FastAPI Security utilities for OAuth2/JWT
# - slowapi or fastapi-limiter for rate limiting
# - Redis for distributed rate limiting state


# Configuration access
@lru_cache()
def get_config(request: Request) -> Dict[str, Any]:
    """
    Get application configuration.

    Args:
        request: FastAPI request object

    Returns:
        Configuration dictionary
    """
    try:
        return request.app.state.config
    except AttributeError:
        logger.warning("Config not found in app state, using defaults")
        return {}


# Service layer dependencies
def get_camera_service(
    camera_manager: CameraManager = Depends(get_camera_manager),
    image_manager: ImageManager = Depends(get_image_manager),
) -> CameraService:
    """
    Get camera service instance.

    Args:
        camera_manager: Camera manager dependency
        image_manager: Image manager dependency

    Returns:
        CameraService instance
    """
    return CameraService(camera_manager=camera_manager, image_manager=image_manager)


def get_image_service(image_manager: ImageManager = Depends(get_image_manager)) -> ImageService:
    """
    Get image service instance.

    Args:
        image_manager: Image manager dependency

    Returns:
        ImageService instance
    """
    return ImageService(image_manager=image_manager)


def get_vision_service(
    image_manager: ImageManager = Depends(get_image_manager),
    template_manager: TemplateManager = Depends(get_template_manager),
) -> VisionService:
    """
    Get vision service instance.

    Args:
        image_manager: Image manager dependency
        template_manager: Template manager dependency

    Returns:
        VisionService instance
    """
    return VisionService(
        image_manager=image_manager,
        template_manager=template_manager,
    )
