"""
Test Image API Router - Test image management for testing without cameras
"""

import logging
import time
from typing import List

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from dependencies import get_image_manager, get_test_image_manager
from domain_types import ROI, Point
from exceptions import safe_endpoint
from managers.image_manager import ImageManager
from managers.test_image_manager import TestImageManager
from models import (
    Size,
    TestImageCaptureResponse,
    TestImageInfo,
    TestImageUploadResponse,
    VisionObject,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class TestImageNotFoundException(HTTPException):
    """Exception for test image not found"""

    def __init__(self, test_id: str):
        super().__init__(status_code=404, detail=f"Test image not found: {test_id}")


@router.get("/list")
@safe_endpoint
async def list_test_images(
    test_image_manager: TestImageManager = Depends(get_test_image_manager),
) -> List[TestImageInfo]:
    """List all test images"""
    test_images = test_image_manager.list()
    return [TestImageInfo.from_manager_dict(t) for t in test_images]


@router.post("/upload")
@safe_endpoint
async def upload_test_image(
    file: UploadFile = File(...),
    test_image_manager: TestImageManager = Depends(get_test_image_manager),
) -> TestImageUploadResponse:
    """Upload new test image"""
    # Read and decode image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Upload test image
    test_id = test_image_manager.upload(file.filename or "test_image.png", image)

    return TestImageUploadResponse(
        success=True,
        test_id=test_id,
        filename=file.filename or "test_image.png",
        size=Size(width=image.shape[1], height=image.shape[0]),
    )


@router.post("/{test_id}/capture")
@safe_endpoint
async def capture_test_image(
    test_id: str,
    test_image_manager: TestImageManager = Depends(get_test_image_manager),
    image_manager: ImageManager = Depends(get_image_manager),
) -> TestImageCaptureResponse:
    """
    Capture (retrieve) test image and store in ImageManager.

    This endpoint mimics camera capture behavior but returns the same test image
    every time. Perfect for testing vision algorithms without real cameras.
    """
    start_time = time.time()

    # Get test image
    test_image = test_image_manager.get(test_id)
    if test_image is None:
        raise TestImageNotFoundException(test_id)

    # Store in ImageManager (creates new image_id each time, like camera capture)
    metadata = {"source": "test_image", "test_id": test_id}
    image_id = image_manager.store(test_image, metadata)

    # Create thumbnail
    _, thumbnail_base64 = image_manager.create_thumbnail(test_image)

    # Get image dimensions
    width = test_image.shape[1]
    height = test_image.shape[0]

    # Create VisionObject representing the captured test image (similar to camera capture)
    vision_object = VisionObject(
        object_id=f"test_{test_id[:8]}",
        object_type="test_image_capture",
        bounding_box=ROI(x=0, y=0, width=width, height=height),
        center=Point(x=width / 2, y=height / 2),
        confidence=1.0,
        properties={
            "test_id": test_id,
            "resolution": [width, height],
            "image_id": image_id,
        },
    )

    # Calculate processing time
    processing_time_ms = int((time.time() - start_time) * 1000)

    logger.info(f"Test image captured: {test_id} → image_id: {image_id}")

    return TestImageCaptureResponse(
        test_id=test_id,
        objects=[vision_object],
        thumbnail_base64=thumbnail_base64,
        processing_time_ms=processing_time_ms,
    )


@router.get("/{test_id}")
@safe_endpoint
async def get_test_image_info(
    test_id: str,
    test_image_manager: TestImageManager = Depends(get_test_image_manager),
) -> TestImageInfo:
    """Get test image metadata"""
    info = test_image_manager.get_info(test_id)
    if info is None:
        raise TestImageNotFoundException(test_id)

    return TestImageInfo.from_manager_dict(info)


@router.get("/{test_id}/image")
@safe_endpoint
async def get_test_image_thumbnail(
    test_id: str,
    test_image_manager: TestImageManager = Depends(get_test_image_manager),
) -> dict:
    """Get test image thumbnail"""
    thumbnail = test_image_manager.create_thumbnail(test_id, max_width=200)
    if thumbnail is None:
        raise TestImageNotFoundException(test_id)

    return {"success": True, "test_id": test_id, "image_base64": thumbnail}


@router.delete("/{test_id}")
@safe_endpoint
async def delete_test_image(
    test_id: str,
    test_image_manager: TestImageManager = Depends(get_test_image_manager),
) -> dict:
    """Delete test image"""
    success = test_image_manager.delete(test_id)
    if not success:
        raise TestImageNotFoundException(test_id)

    return {"success": True, "message": f"Test image {test_id} deleted"}
