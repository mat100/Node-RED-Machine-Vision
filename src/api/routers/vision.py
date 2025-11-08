"""
Vision API Router - Vision processing endpoints

All vision detection endpoints follow a unified pattern:
1. Validate request (image exists, ROI valid)
2. Call vision service method (returns List[VisionObject], thumbnail, timing)
3. Return VisionResponse

This consistency reduces code duplication and makes the API predictable.
"""

import logging
from typing import Callable, List, Optional

from fastapi import APIRouter, Depends

from api.dependencies import get_image_manager, get_vision_service, validate_vision_request
from api.exceptions import safe_endpoint
from core.image_manager import ImageManager
from schemas import (
    ROI,
    ArucoDetectRequest,
    ColorDetectRequest,
    EdgeDetectRequest,
    RotationDetectRequest,
    TemplateMatchRequest,
    VisionObject,
    VisionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def execute_vision_detection(
    image_id: str,
    roi: Optional[ROI],
    image_manager: ImageManager,
    detection_callable: Callable[[Optional[dict]], tuple[List[VisionObject], str, int]],
) -> VisionResponse:
    """
    Unified helper for executing vision detection endpoints.

    Eliminates duplicated validation and response construction code across
    all vision endpoints (~15 lines per endpoint → ~3 lines with this helper).

    Args:
        image_id: Image ID to process
        roi: Optional ROI from request
        image_manager: ImageManager instance for validation
        detection_callable: Service method to call, receives roi_dict and returns
                           (objects, thumbnail_base64, processing_time_ms)

    Returns:
        VisionResponse with detection results
    """
    # Unified validation and ROI conversion
    roi_dict = validate_vision_request(image_id, roi, image_manager)

    # Execute detection (service handles all logic)
    detected_objects, thumbnail_base64, processing_time = detection_callable(roi_dict)

    # Unified response construction
    return VisionResponse(
        objects=detected_objects,
        thumbnail_base64=thumbnail_base64,
        processing_time_ms=processing_time,
    )


@router.post("/template-match")
@safe_endpoint
async def template_match(
    request: TemplateMatchRequest,
    vision_service=Depends(get_vision_service),
    image_manager=Depends(get_image_manager),
) -> VisionResponse:
    """
    Perform template matching on an image.

    INPUT constraints:
    - roi: Optional region to limit template search area

    OUTPUT results:
    - bounding_box: Location where template was found
    """
    return execute_vision_detection(
        request.image_id,
        request.roi,
        image_manager,
        lambda roi: vision_service.template_match(
            image_id=request.image_id,
            template_id=request.params.template_id,
            roi=roi,
            params=request.params,
        ),
    )


@router.post("/edge-detect")
@safe_endpoint
async def edge_detect(
    request: EdgeDetectRequest,
    vision_service=Depends(get_vision_service),
    image_manager=Depends(get_image_manager),
) -> VisionResponse:
    """
    Perform edge detection with multiple methods.

    INPUT constraints:
    - roi: Optional region to limit edge detection area

    OUTPUT results:
    - bounding_box: Bounding box of detected contour
    - contour: Actual contour points for precise shape representation
    """
    return execute_vision_detection(
        request.image_id,
        request.roi,
        image_manager,
        lambda roi: vision_service.edge_detect(
            image_id=request.image_id,
            method=request.params.method,
            params=request.params,
            roi=roi,
        ),
    )


@router.post("/color-detect")
@safe_endpoint
async def color_detect(
    request: ColorDetectRequest,
    vision_service=Depends(get_vision_service),
    image_manager=Depends(get_image_manager),
) -> VisionResponse:
    """
    Perform color detection with automatic dominant color recognition.

    INPUT constraints:
    - roi: Optional region for color analysis
    - contour: Optional contour points for precise masking (auto-used from msg.payload)

    OUTPUT results:
    - bounding_box: Region where color was analyzed
    """
    return execute_vision_detection(
        request.image_id,
        request.roi,
        image_manager,
        lambda roi: vision_service.color_detect(
            image_id=request.image_id,
            roi=roi,
            contour=request.contour,
            expected_color=request.expected_color,
            params=request.params,
        ),
    )


@router.post("/aruco-detect")
@safe_endpoint
async def aruco_detect(
    request: ArucoDetectRequest,
    vision_service=Depends(get_vision_service),
    image_manager=Depends(get_image_manager),
) -> VisionResponse:
    """
    Detect ArUco fiducial markers in image.

    INPUT constraints:
    - roi: Optional region to limit marker search area

    OUTPUT results:
    - bounding_box: Bounding box of detected marker
    - properties.marker_id: Unique marker ID
    - properties.corners: 4 corner points
    - rotation: Marker rotation in degrees (0-360)
    """
    return execute_vision_detection(
        request.image_id,
        request.roi,
        image_manager,
        lambda roi: vision_service.aruco_detect(
            image_id=request.image_id,
            roi=roi,
            params=request.params,
        ),
    )


@router.post("/rotation-detect")
@safe_endpoint
async def rotation_detect(
    request: RotationDetectRequest,
    vision_service=Depends(get_vision_service),
    image_manager=Depends(get_image_manager),
) -> VisionResponse:
    """
    Detect rotation angle from contour points.

    INPUT constraints:
    - roi: Optional ROI for visualization context only (not a constraint)
    - contour: Required contour points from edge detection

    OUTPUT results:
    - rotation: Calculated rotation in degrees
    """
    return execute_vision_detection(
        request.image_id,
        request.roi,
        image_manager,
        lambda roi: vision_service.rotation_detect(
            image_id=request.image_id,
            contour=request.contour,
            roi=roi,
            params=request.params,
        ),
    )
