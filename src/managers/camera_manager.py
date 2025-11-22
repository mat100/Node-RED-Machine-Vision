"""
Camera Manager - Handles multiple camera types (USB, IP, etc.)
"""

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from queue import Queue
from threading import Lock, Thread
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from domain_types import CameraType

# Suppress H.264 decoder warnings from ffmpeg/libav
# These occur when starting H.264 streams mid-stream before receiving keyframe
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

# Set OpenCV log level to ERROR to suppress warnings (if available)
try:
    # OpenCV 4.5+ has setLogLevel
    if hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(
            3
        )  # 3 = ERROR level (0=SILENT, 1=FATAL, 2=ERROR, 3=WARNING, 4=INFO, 5=DEBUG)
except Exception:
    pass  # Silently ignore if not supported

logger = logging.getLogger(__name__)


@dataclass
class CameraSettings:
    """Camera settings and configuration"""

    id: str
    name: str
    type: CameraType
    source: Any  # int for USB, str for IP/file
    resolution: tuple = (1920, 1080)
    fps: int = 30
    capture_timeout_ms: int = 5000  # Timeout for capture operations in milliseconds

    def __post_init__(self):
        """Adjust timeout for IP cameras to account for buffer flushing"""
        if self.type == CameraType.IP:
            # IP cameras need longer timeout due to buffer flushing
            self.capture_timeout_ms = max(self.capture_timeout_ms, 10000)


class Camera:
    """Base camera class"""

    def __init__(self, config: CameraSettings):
        self.config = config
        self.cap = None
        self.connected = False
        self.lock = Lock()
        self.last_frame = None
        self.last_capture_time = 0
        # Thread pool for timeout handling (single thread per camera)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"cam_{config.id}")

    def connect(self) -> bool:
        """Connect to camera"""
        try:
            with self.lock:
                # TEST camera doesn't use VideoCapture
                if self.config.type == CameraType.TEST:
                    self.connected = True
                    logger.info(f"Test camera {self.config.id} connected")
                    return True

                if self.config.type == CameraType.USB:
                    self.cap = cv2.VideoCapture(self.config.source)
                elif self.config.type == CameraType.IP:
                    self.cap = cv2.VideoCapture(self.config.source, cv2.CAP_FFMPEG)
                elif self.config.type == CameraType.FILE:
                    self.cap = cv2.VideoCapture(self.config.source)
                else:
                    return False

                if self.cap and self.cap.isOpened():
                    # Set resolution
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution[0])
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution[1])
                    self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)

                    # For H.264 streams: discard initial frames until we get a valid keyframe
                    # This prevents "non-existing PPS" errors
                    if self.config.type in (CameraType.IP, CameraType.FILE):
                        logger.debug(f"Flushing initial frames for {self.config.id}...")
                        for _ in range(10):
                            ret, _ = self.cap.read()
                            if ret:
                                break
                        logger.debug(f"Initial frames flushed for {self.config.id}")

                    self.connected = True
                    logger.info(f"Camera {self.config.id} connected")
                    return True

                return False

        except Exception as e:
            logger.error(f"Failed to connect camera {self.config.id}: {e}")
            return False

    def disconnect(self):
        """Disconnect camera"""
        with self.lock:
            if self.cap:
                try:
                    # Ensure all buffers are cleared before release
                    if self.cap.isOpened():
                        # Read and discard any remaining frames in buffer
                        for _ in range(5):
                            self.cap.grab()
                    self.cap.release()
                    # Explicitly delete the VideoCapture object
                    del self.cap
                except Exception as e:
                    logger.warning(f"Error during camera release: {e}")
                finally:
                    self.cap = None
            self.connected = False
            self.last_frame = None

        # Shutdown executor outside of lock to avoid deadlock
        try:
            self._executor.shutdown(wait=False)
        except Exception as e:
            logger.warning(f"Error shutting down executor: {e}")

        logger.info(f"Camera {self.config.id} disconnected")

    def _create_test_frame(self) -> np.ndarray:
        """Create a test frame for TEST camera type"""
        from managers.image.test_patterns import create_simple_test_image

        width, height = self.config.resolution
        return create_simple_test_image(
            width=width, height=height, text=f"Test Camera: {self.config.id}"
        )

    def _capture_frame_blocking(self) -> tuple:
        """
        Internal method to capture a frame (blocking).
        This is run in a separate thread to enable timeout.
        """
        with self.lock:
            if self.cap and self.cap.isOpened():
                # For IP cameras (RTSP), flush buffer to get fresh frame
                # RTSP streams buffer many frames - read and discard until we hit live stream
                if self.config.type == CameraType.IP:
                    flush_start = time.time()
                    max_flush_time = 5.0  # Max 5 seconds for flush to avoid timeout
                    discarded = 0
                    expected_frame_time = 1.0 / max(self.config.fps, 1)

                    for i in range(50):
                        # Stop if flush takes too long (prevent timeout)
                        if (time.time() - flush_start) > max_flush_time:
                            logger.warning(f"RTSP flush time limit reached ({discarded} frames)")
                            break

                        frame_start = time.time()
                        ret, _ = self.cap.read()
                        frame_time = time.time() - frame_start

                        if not ret:
                            logger.warning(f"RTSP buffer flush failed at frame {i}")
                            break

                        discarded += 1

                        # Buffered frames return instantly, live frames wait
                        # If read took significant time, we've reached live stream
                        if frame_time > (expected_frame_time * 0.5):
                            break

                    logger.debug(
                        f"RTSP: flushed {discarded} frames in {time.time() - flush_start:.1f}s"
                    )

                return self.cap.read()
        return False, None

    def capture(self) -> Optional[np.ndarray]:
        """
        Capture single frame with timeout protection.

        Returns:
            Frame as numpy array, or None if capture fails or times out
        """
        if not self.connected:
            return None

        # TEST camera generates frames directly
        if self.config.type == CameraType.TEST:
            try:
                frame = self._create_test_frame()
                self.last_frame = frame
                self.last_capture_time = time.time()
                return frame
            except Exception as e:
                logger.error(f"Failed to create test frame: {e}")
                return None

        # Convert timeout from milliseconds to seconds
        timeout_seconds = self.config.capture_timeout_ms / 1000.0

        try:
            # Submit capture task to executor with timeout
            future = self._executor.submit(self._capture_frame_blocking)
            ret, frame = future.result(timeout=timeout_seconds)

            if ret and frame is not None:
                self.last_frame = frame
                self.last_capture_time = time.time()
                return frame
            else:
                logger.warning(f"Camera {self.config.id}: capture failed (no frame returned)")
                return None

        except FuturesTimeoutError:
            logger.error(
                f"Camera {self.config.id}: capture timeout after {timeout_seconds}s "
                f"- camera may be disconnected or unresponsive"
            )
            return None
        except Exception as e:
            logger.error(f"Camera {self.config.id}: capture error: {e}")
            return None

    def get_preview(self) -> Optional[np.ndarray]:
        """Get preview frame (can be cached)"""
        # For IP cameras (RTSP), always get fresh frame - don't cache
        if self.config.type == CameraType.IP:
            return self.capture()

        # For other cameras, keep a very short cache based on target FPS to avoid stale frames
        cache_ttl = 1.0 / max(self.config.fps, 1)
        if self.last_frame is not None and (time.time() - self.last_capture_time) < cache_ttl:
            return self.last_frame

        # Otherwise capture new frame
        return self.capture()


class CameraManager:
    """Manages multiple cameras"""

    def __init__(self, default_resolution: Dict[str, int] = None, capture_timeout_ms: int = 5000):
        self.cameras: Dict[str, Camera] = {}
        self.lock = Lock()
        self.default_resolution = (
            (default_resolution["width"], default_resolution["height"])
            if default_resolution
            else (1920, 1080)
        )
        self.capture_timeout_ms = capture_timeout_ms

        # Preview thread
        self.preview_thread = None
        self.preview_queue = Queue()
        self.preview_running = False

        logger.info(
            f"Camera Manager initialized (timeout: {capture_timeout_ms}ms, "
            f"resolution: {self.default_resolution})"
        )

    def list_available_cameras(self) -> List[Dict[str, Any]]:
        """List available cameras"""
        cameras = []

        # Always add test camera option
        cameras.append(
            {
                "id": "test",
                "name": "Test Image Generator",
                "type": "test",
                "connected": True,
                "resolution": {"width": 1920, "height": 1080},
            }
        )

        # Check USB cameras (typically 0-4 for better performance)
        for i in range(5):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    # Try to read actual resolution
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    cameras.append(
                        {
                            "id": f"usb_{i}",
                            "name": f"USB Camera {i}",
                            "type": "usb",
                            "source": i,
                            "connected": False,  # Not yet connected (just available)
                            "resolution": {
                                "width": width if width > 0 else 1920,
                                "height": height if height > 0 else 1080,
                            },
                        }
                    )
                    # Properly release the camera
                    cap.release()
                    # Explicitly delete to free resources
                    del cap
            except Exception as e:
                logger.debug(f"Camera index {i} not available: {e}")
                continue

        # Add currently connected cameras
        with self.lock:
            for cam_id, camera in self.cameras.items():
                # Skip if already in list (e.g., USB cameras)
                if not any(c["id"] == cam_id for c in cameras):
                    cameras.append(
                        {
                            "id": cam_id,
                            "name": camera.config.name,
                            "type": camera.config.type.value,
                            "connected": camera.connected,
                            "resolution": {
                                "width": camera.config.resolution[0],
                                "height": camera.config.resolution[1],
                            },
                        }
                    )
                else:
                    # Update connection status for existing camera
                    for c in cameras:
                        if c["id"] == cam_id:
                            c["connected"] = camera.connected
                            break

        return cameras

    def connect_camera(
        self,
        camera_id: str,
        camera_type: str = "usb",
        source: Any = 0,
        name: Optional[str] = None,
        resolution: Optional[tuple] = None,
    ) -> bool:
        """Connect to a camera"""
        with self.lock:
            # Check if already connected
            if camera_id in self.cameras and self.cameras[camera_id].connected:
                logger.warning(f"Camera {camera_id} already connected")
                return True

            # Create camera configuration
            config = CameraSettings(
                id=camera_id,
                name=name or camera_id,
                type=CameraType(camera_type),
                source=source,
                resolution=resolution or self.default_resolution,
                capture_timeout_ms=self.capture_timeout_ms,
            )

            # Create and connect camera
            camera = Camera(config)
            if camera.connect():
                self.cameras[camera_id] = camera
                return True

            return False

    def disconnect_camera(self, camera_id: str) -> bool:
        """Disconnect a camera"""
        with self.lock:
            if camera_id in self.cameras:
                self.cameras[camera_id].disconnect()
                del self.cameras[camera_id]
                return True
            return False

    def capture(self, camera_id: str) -> Optional[np.ndarray]:
        """Capture frame from specific camera"""
        with self.lock:
            if camera_id in self.cameras:
                return self.cameras[camera_id].capture()

        logger.warning(f"Camera {camera_id} not found")
        return None

    def get_preview(self, camera_id: str) -> Optional[np.ndarray]:
        """Get preview frame from camera"""
        with self.lock:
            if camera_id in self.cameras:
                return self.cameras[camera_id].get_preview()

        return None

    def start_preview_stream(self, camera_id: str, interval_ms: int = 2000):
        """Start preview stream for a camera"""
        if self.preview_running:
            self.stop_preview_stream()

        self.preview_running = True
        self.preview_thread = Thread(
            target=self._preview_worker, args=(camera_id, interval_ms), daemon=True
        )
        self.preview_thread.start()
        logger.info(f"Preview stream started for camera {camera_id}")

    def stop_preview_stream(self):
        """Stop preview stream"""
        self.preview_running = False
        if self.preview_thread:
            self.preview_thread.join(timeout=2)
        logger.info("Preview stream stopped")

    def _preview_worker(self, camera_id: str, interval_ms: int):
        """Worker thread for preview streaming"""
        interval_sec = interval_ms / 1000.0

        while self.preview_running:
            frame = self.get_preview(camera_id)
            if frame is not None:
                self.preview_queue.put(frame)

            time.sleep(interval_sec)

    def get_camera_info(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """Get camera information"""
        with self.lock:
            if camera_id in self.cameras:
                camera = self.cameras[camera_id]
                return {
                    "id": camera.config.id,
                    "name": camera.config.name,
                    "type": camera.config.type.value,
                    "resolution": {
                        "width": camera.config.resolution[0],
                        "height": camera.config.resolution[1],
                    },
                    "fps": camera.config.fps,
                    "connected": camera.connected,
                }

        return None

    async def cleanup(self):
        """Clean up all cameras"""
        logger.info("Cleaning up Camera Manager...")

        # Stop preview stream first
        self.stop_preview_stream()

        # Small delay to ensure preview thread is fully stopped
        await asyncio.sleep(0.1)

        # Disconnect all cameras properly
        with self.lock:
            camera_ids = list(self.cameras.keys())

        # Disconnect cameras outside the lock to avoid deadlock
        for camera_id in camera_ids:
            try:
                logger.info(f"Disconnecting camera {camera_id}...")
                with self.lock:
                    if camera_id in self.cameras:
                        self.cameras[camera_id].disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting camera {camera_id}: {e}")

        # Clear the cameras dictionary
        with self.lock:
            self.cameras.clear()

        # Force garbage collection to ensure resources are freed
        import gc

        gc.collect()

        logger.info("Camera Manager cleanup complete")

    def create_test_image(self, text: str = "Test Image") -> np.ndarray:
        """Create a test image for development with ArUco markers"""
        from managers.image.test_patterns import create_test_image_with_markers

        return create_test_image_with_markers(
            width=1920,
            height=1080,
            text=text,
            add_timestamp=True,
            add_grid=True,
            add_shapes=True,
        )
