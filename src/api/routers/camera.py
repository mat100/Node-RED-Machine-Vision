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

from api.dependencies import get_camera_manager  # Still needed for stream endpoint
from api.dependencies import get_camera_service
from api.exceptions import safe_endpoint
from core.utils.camera_identifier import parse as parse_camera_id
from schemas import CameraCaptureResponse, CameraConnectRequest, CameraInfo, CaptureRequest

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/list")
@safe_endpoint
async def list_cameras(camera_service=Depends(get_camera_service)) -> List[CameraInfo]:
    """List available cameras"""
    cameras = camera_service.list_available_cameras()
    return [CameraInfo.from_manager_dict(cam) for cam in cameras]


@router.post("/connect")
@safe_endpoint
async def connect_camera(
    request: CameraConnectRequest, camera_service=Depends(get_camera_service)
) -> dict:
    """Connect to a camera"""
    # Parse camera ID using unified utility
    camera_type, source = parse_camera_id(request.camera_id)

    resolution = None
    if request.resolution:
        resolution = (request.resolution.width, request.resolution.height)

    # Service handles connection and raises exception on failure
    camera_service.connect_camera(
        camera_id=request.camera_id, camera_type=camera_type, source=source, resolution=resolution
    )

    return {"success": True, "message": f"Camera {request.camera_id} connected"}


@router.post("/capture")
@safe_endpoint
async def capture_image(
    request: CaptureRequest,
    camera_service=Depends(get_camera_service),
) -> CameraCaptureResponse:
    """Capture image from camera"""
    # Extract ROI from params if provided
    roi_obj = None
    if request.params and request.params.roi:
        roi_obj = request.params.roi

    # Service handles capture, ROI extraction, storage, and thumbnail creation
    image_id, thumbnail_base64, metadata = camera_service.capture_and_store(
        camera_id=request.camera_id, roi=roi_obj
    )

    return CameraCaptureResponse(
        success=True,
        image_id=image_id,
        timestamp=datetime.now(),
        thumbnail_base64=thumbnail_base64,
        metadata=metadata,
    )


@router.get("/preview/{camera_id}")
@safe_endpoint
async def get_preview(camera_id: str, camera_service=Depends(get_camera_service)) -> dict:
    """Get preview image from camera"""
    # Service handles preview and thumbnail creation
    _, thumbnail_base64 = camera_service.get_preview(camera_id=camera_id, create_thumbnail=True)

    return {
        "success": True,
        "thumbnail_base64": thumbnail_base64,
        "timestamp": datetime.now().isoformat(),
    }


@router.delete("/disconnect/{camera_id}")
@safe_endpoint
async def disconnect_camera(camera_id: str, camera_service=Depends(get_camera_service)) -> dict:
    """Disconnect camera"""
    # Service handles disconnection and raises exception on failure
    camera_service.disconnect_camera(camera_id)

    return {"success": True, "message": f"Camera {camera_id} disconnected"}


# Store active streams to limit concurrent connections
active_streams = {}


@router.get("/stream/{camera_id}")
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


@router.post("/stream/stop/{camera_id}")
async def stop_stream(camera_id: str) -> dict:
    """Stop MJPEG stream for a camera"""
    if camera_id in active_streams:
        active_streams[camera_id] = False
        return {"success": True, "message": f"Stream stopped for camera {camera_id}"}
    return {"success": False, "message": "Stream not found"}
