"""
Camera API Router
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import List

import cv2
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from dependencies import get_camera_manager, get_image_manager
from domain_types import ROI, Point
from exceptions import CameraConnectionException, CameraNotFoundException, safe_endpoint
from image.roi import extract_roi
from managers.camera_manager import CameraManager
from managers.image_manager import ImageManager
from models import CameraConnectRequest, CameraInfo, CaptureRequest, VisionObject, VisionResponse
from utils import parse_camera_id

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/list")
@safe_endpoint
async def list_cameras(
    camera_manager: CameraManager = Depends(get_camera_manager),
) -> List[CameraInfo]:
    """List available cameras"""
    cameras = camera_manager.list_available_cameras()
    return [CameraInfo.from_manager_dict(cam) for cam in cameras]


@router.post("/connect")
@safe_endpoint
async def connect_camera(
    request: CameraConnectRequest, camera_manager: CameraManager = Depends(get_camera_manager)
) -> dict:
    """Connect to a camera"""
    # Parse camera ID using unified utility
    camera_type, source = parse_camera_id(request.camera_id)

    resolution = None
    if request.resolution:
        resolution = (request.resolution.width, request.resolution.height)

    # Connect camera
    success = camera_manager.connect_camera(
        camera_id=request.camera_id, camera_type=camera_type, source=source, resolution=resolution
    )

    if not success:
        raise CameraConnectionException(camera_id=request.camera_id, reason="Connection failed")

    logger.info(f"Camera {request.camera_id} connected successfully")
    return {"success": True, "message": f"Camera {request.camera_id} connected"}


@router.post("/capture")
@safe_endpoint
async def capture_image(
    request: CaptureRequest,
    camera_manager: CameraManager = Depends(get_camera_manager),
    image_manager: ImageManager = Depends(get_image_manager),
) -> VisionResponse:
    """Capture image from camera"""
    start_time = time.time()

    # Extract ROI from params if provided
    roi_obj = None
    if request.params and request.params.roi:
        roi_obj = request.params.roi

    # Auto-connect camera if not already connected
    cameras = camera_manager.list_available_cameras()
    is_connected = any(
        cam["id"] == request.camera_id and cam.get("connected", False) for cam in cameras
    )

    if not is_connected:
        logger.info(f"Camera {request.camera_id} not connected, attempting auto-connect")
        try:
            # Parse camera ID using unified utility
            camera_type, source = parse_camera_id(request.camera_id)
            success = camera_manager.connect_camera(
                camera_id=request.camera_id, camera_type=camera_type, source=source
            )
            if not success:
                raise CameraConnectionException(
                    camera_id=request.camera_id, reason="Connection failed"
                )
            logger.info(f"Camera {request.camera_id} auto-connected successfully")
        except Exception as e:
            logger.warning(f"Failed to auto-connect camera {request.camera_id}: {e}")

    # Capture image
    image = camera_manager.capture(request.camera_id)

    if image is None:
        # Try test image for development
        logger.warning(f"Camera {request.camera_id} not found, using test image")
        image = camera_manager.create_test_image(f"Camera: {request.camera_id}")

    # Apply ROI if specified
    if roi_obj:
        image = extract_roi(image, roi_obj, safe_mode=True)
        if image is None:
            raise ValueError("Invalid ROI parameters")

    # Prepare metadata
    metadata = {
        "camera_id": request.camera_id,
        "timestamp": datetime.now().isoformat(),
        "roi": roi_obj.to_dict() if roi_obj else None,
    }

    # Store image
    image_id = image_manager.store(image, metadata)

    # Create thumbnail (uses config width)
    _, thumbnail_base64 = image_manager.create_thumbnail(image)

    # Get image dimensions
    width = image.shape[1]
    height = image.shape[0]

    # Create VisionObject representing the captured image
    vision_object = VisionObject(
        object_id=f"img_{image_id[:8]}",
        object_type="camera_capture",
        bounding_box=ROI(x=0, y=0, width=width, height=height),
        center=Point(x=width / 2, y=height / 2),
        confidence=1.0,
        properties={
            "camera_id": request.camera_id,
            "resolution": [width, height],
            "image_id": image_id,
        },
    )

    # Calculate processing time
    processing_time_ms = int((time.time() - start_time) * 1000)

    logger.debug(f"Image captured and stored: {image_id}")

    return VisionResponse(
        objects=[vision_object],
        thumbnail_base64=thumbnail_base64,
        processing_time_ms=processing_time_ms,
    )


@router.get("/preview/{camera_id}")
@safe_endpoint
async def get_preview(
    camera_id: str,
    camera_manager: CameraManager = Depends(get_camera_manager),
    image_manager: ImageManager = Depends(get_image_manager),
) -> dict:
    """Get preview image from camera"""
    # Get preview frame
    image = camera_manager.get_preview(camera_id)

    if image is None:
        # Use test image
        image = camera_manager.create_test_image(f"Preview: {camera_id}")

    # Create thumbnail (uses config width)
    _, thumbnail_base64 = image_manager.create_thumbnail(image)

    return {
        "success": True,
        "thumbnail_base64": thumbnail_base64,
        "timestamp": datetime.now().isoformat(),
    }


@router.delete("/disconnect/{camera_id}")
@safe_endpoint
async def disconnect_camera(
    camera_id: str, camera_manager: CameraManager = Depends(get_camera_manager)
) -> dict:
    """Disconnect camera"""
    # Disconnect camera
    success = camera_manager.disconnect_camera(camera_id)

    if not success:
        raise CameraNotFoundException(camera_id)

    logger.info(f"Camera {camera_id} disconnected successfully")
    return {"success": True, "message": f"Camera {camera_id} disconnected"}


@router.get("/stream/{camera_id:path}")
async def stream_mjpeg(
    camera_id: str, camera_manager=Depends(get_camera_manager), request: Request = None
):
    """
    Stream MJPEG video from camera for live preview.

    Returns a multipart/x-mixed-replace stream of JPEG frames.
    Resolution: 1280x720
    FPS: 15
    """
    config = request.app.state.config if request else {}
    active_streams = request.app.state.active_streams

    # Check if stream already exists for another camera (single stream limitation)
    if active_streams and camera_id not in active_streams:
        # Stop other streams
        for stream_id in list(active_streams.keys()):
            active_streams[stream_id] = False
        active_streams.clear()

    # Mark this stream as active
    active_streams[camera_id] = True

    # Get preview settings from config or use defaults
    preview_resolution = config.get("preview", {}).get("resolution", [1280, 720])
    preview_fps = config.get("preview", {}).get("fps", 15)
    preview_quality = config.get("preview", {}).get("quality", 85)

    frame_interval = 1.0 / preview_fps  # Time between frames

    async def generate():
        """Generate MJPEG frames"""
        last_frame_time = 0

        try:
            while active_streams.get(camera_id, False):
                current_time = time.time()

                # Limit frame rate
                if current_time - last_frame_time < frame_interval:
                    await asyncio.sleep(frame_interval - (current_time - last_frame_time))

                # Get frame from camera
                frame = camera_manager.get_preview(camera_id)

                if frame is None:
                    # Use test image if camera not available
                    frame = camera_manager.create_test_image(
                        f"Live Preview: {camera_id}\nTime: {datetime.now().strftime('%H:%M:%S')}"
                    )

                # Resize frame to target resolution
                if frame.shape[:2] != (preview_resolution[1], preview_resolution[0]):
                    frame = cv2.resize(frame, tuple(preview_resolution))

                # Encode frame as JPEG
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), preview_quality]
                _, buffer = cv2.imencode(".jpg", frame, encode_param)

                # Yield frame in MJPEG format
                yield (
                    b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )

                last_frame_time = time.time()

        except Exception as e:
            logger.error(f"Error in MJPEG stream: {e}")
        finally:
            # Clean up stream
            if camera_id in active_streams:
                del active_streams[camera_id]
            logger.info(f"MJPEG stream ended for camera {camera_id}")

    # Return streaming response
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "close",
        },
    )


@router.post("/stream/stop/{camera_id:path}")
async def stop_stream(camera_id: str, request: Request) -> dict:
    """Stop MJPEG stream for a camera"""
    active_streams = request.app.state.active_streams
    if camera_id in active_streams:
        active_streams[camera_id] = False
        return {"success": True, "message": f"Stream stopped for camera {camera_id}"}
    return {"success": False, "message": "Stream not found"}
