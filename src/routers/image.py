"""
Image API Router - Image processing operations
"""

import logging
import os
import time

import cv2
from fastapi import APIRouter, Depends
from fastapi.responses import Response

from dependencies import get_image_manager
from domain_types import ROI, Point
from exceptions import ImageNotFoundException, safe_endpoint
from image.roi import extract_roi
from managers.image_manager import ImageManager
from models import ImageImportRequest, ROIExtractRequest, VisionObject, VisionResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/{image_id}")
@safe_endpoint
async def get_image(
    image_id: str, image_manager: ImageManager = Depends(get_image_manager)
) -> Response:
    """
    Get full quality image by ID.

    Returns the complete image stored in ImageManager as JPEG.
    Useful for downloading processed images at full resolution.

    Args:
        image_id: UUID of the image in ImageManager
        image_manager: Image manager dependency

    Returns:
        JPEG image as Response

    Raises:
        ImageNotFoundException: If image_id not found in cache
    """
    # Get image from manager
    image = image_manager.get(image_id)
    if image is None:
        logger.error(f"Image {image_id} not found in cache")
        raise ImageNotFoundException(image_id)

    # Encode as JPEG (quality 95 for download)
    success, encoded_image = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        logger.error(f"Failed to encode image {image_id}")
        raise ValueError(f"Failed to encode image {image_id}")

    logger.info(f"Returning full image {image_id} ({image.shape[1]}x{image.shape[0]})")

    # Return as JPEG response
    return Response(content=encoded_image.tobytes(), media_type="image/jpeg")


@router.post("/extract-roi")
@safe_endpoint
async def extract_roi_endpoint(
    request: ROIExtractRequest, image_manager: ImageManager = Depends(get_image_manager)
) -> VisionResponse:
    """
    Extract Region of Interest thumbnail from an image.

    This endpoint extracts a rectangular region from an existing image
    and returns it as a VisionObject. The original image_id remains
    unchanged, only the roi and thumbnail are updated.

    Args:
        request: ROI extraction request with image_id and roi coordinates
        image_manager: Image manager dependency

    Returns:
        VisionResponse with single VisionObject representing the extracted ROI
    """
    start_time = time.time()

    # Get image from manager
    original_image = image_manager.get(request.image_id)
    if original_image is None:
        logger.error(f"Image {request.image_id} not found in cache")
        raise ImageNotFoundException(request.image_id)

    # Extract ROI from source image (clipped to image bounds)
    roi_image = extract_roi(original_image, request.roi, safe_mode=True)
    if roi_image is None:
        raise ValueError(
            f"Invalid ROI parameters: {request.roi.dict()} for image {request.image_id}"
        )

    # Generate thumbnail using ImageManager
    _, thumbnail = image_manager.create_thumbnail(roi_image, width=None, image_id=None)

    # Get actual clipped bounding box
    img_height, img_width = original_image.shape[:2]
    clipped_bbox = request.roi.clip(img_width, img_height)

    logger.info(
        f"Extracted ROI thumbnail from {request.image_id}: "
        f"{clipped_bbox.width}x{clipped_bbox.height} at ({clipped_bbox.x},{clipped_bbox.y})"
    )

    # Create VisionObject representing the extracted ROI
    vision_object = VisionObject(
        object_id=f"roi_{request.image_id[:8]}",
        object_type="roi_extract",
        bounding_box=clipped_bbox,
        center=Point(
            x=clipped_bbox.x + clipped_bbox.width / 2,
            y=clipped_bbox.y + clipped_bbox.height / 2,
        ),
        confidence=1.0,
        area=float(clipped_bbox.width * clipped_bbox.height),
        properties={
            "source_image_id": request.image_id,
            "original_roi": request.roi.dict(),
            "clipped": (
                clipped_bbox.x != request.roi.x
                or clipped_bbox.y != request.roi.y
                or clipped_bbox.width != request.roi.width
                or clipped_bbox.height != request.roi.height
            ),
        },
    )

    # Calculate processing time
    processing_time_ms = int((time.time() - start_time) * 1000)

    return VisionResponse(
        objects=[vision_object],
        thumbnail_base64=thumbnail,
        processing_time_ms=processing_time_ms,
    )


@router.post("/import")
@safe_endpoint
async def import_image(
    request: ImageImportRequest, image_manager: ImageManager = Depends(get_image_manager)
) -> VisionResponse:
    """
    Import image from file system.

    Loads an image from the file system (JPG, PNG, BMP, etc.), stores it in
    ImageManager, and generates a thumbnail. This allows external applications
    to register their images for processing by vision nodes.

    The response structure uses VisionResponse for consistency with all other
    vision processing endpoints.

    Args:
        request: Import request with file path
        image_manager: Image manager dependency

    Returns:
        VisionResponse with single VisionObject representing the imported image

    Raises:
        HTTPException 404: If file not found
        HTTPException 400: If file cannot be loaded as image
    """
    start_time = time.time()

    # Validate file exists
    if not os.path.exists(request.file_path):
        logger.error(f"Image file not found: {request.file_path}")
        raise FileNotFoundError(f"Image file not found: {request.file_path}")

    # Load image with OpenCV
    image = cv2.imread(request.file_path, cv2.IMREAD_COLOR)
    if image is None:
        logger.error(f"Failed to load image from: {request.file_path}")
        raise ValueError(f"Failed to load image from: {request.file_path}")

    # Extract metadata
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) > 2 else 1
    file_size = os.path.getsize(request.file_path)

    metadata = {
        "source": "file_import",
        "file_path": request.file_path,
        "width": width,
        "height": height,
        "channels": channels,
        "file_size_bytes": file_size,
    }

    # Store in ImageManager
    image_id = image_manager.store(image, metadata)

    # Generate thumbnail
    _, thumbnail_base64 = image_manager.create_thumbnail(image, width=None, image_id=image_id)

    logger.info(
        f"Imported image from {request.file_path}: {image_id} ({width}x{height}, {file_size} bytes)"
    )

    # Create VisionObject representing the imported image
    vision_object = VisionObject(
        object_id=f"img_{image_id[:8]}",
        object_type="image_import",
        bounding_box=ROI(x=0, y=0, width=width, height=height),
        center=Point(x=width / 2, y=height / 2),
        confidence=1.0,
        properties={
            "source": metadata["source"],
            "file_path": metadata["file_path"],
            "file_size_bytes": metadata["file_size_bytes"],
            "resolution": [width, height],
            "image_id": image_id,
        },
    )

    # Calculate processing time
    processing_time_ms = int((time.time() - start_time) * 1000)

    return VisionResponse(
        objects=[vision_object],
        thumbnail_base64=thumbnail_base64,
        processing_time_ms=processing_time_ms,
    )
