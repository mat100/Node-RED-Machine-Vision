"""
Camera Manager - Handles multiple camera types (USB, IP, etc.)
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from queue import Queue
from threading import Lock, Thread
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from core.enums import CameraType

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
                if self.config.type == CameraType.USB:
                    self.cap = cv2.VideoCapture(self.config.source)
                elif self.config.type == CameraType.IP:
                    self.cap = cv2.VideoCapture(self.config.source)
                elif self.config.type == CameraType.FILE:
                    self.cap = cv2.VideoCapture(self.config.source)
                else:
                    return False

                if self.cap and self.cap.isOpened():
                    # Set resolution
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution[0])
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution[1])
                    self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)

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

    def _capture_frame_blocking(self) -> tuple:
        """
        Internal method to capture a frame (blocking).
        This is run in a separate thread to enable timeout.
        """
        with self.lock:
            if self.cap and self.cap.isOpened():
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
        # Keep a very short cache based on target FPS to avoid stale frames
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
        # Create a 1920x1080 test image
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # Add gradient background
        for i in range(1080):
            img[i, :] = [i * 255 // 1080, 100, 255 - i * 255 // 1080]

        # Add ArUco markers (for testing rotation detection)
        try:
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            marker_size = 200  # Marker itself
            border_size = 50  # White border around marker
            total_size = marker_size + 2 * border_size  # Total with border

            # Generate first marker (ID 0)
            marker_image = cv2.aruco.generateImageMarker(aruco_dict, 0, marker_size)
            # Create white background and place marker in center
            marker_with_border = np.ones((total_size, total_size), dtype=np.uint8) * 255
            marker_with_border[
                border_size : border_size + marker_size, border_size : border_size + marker_size
            ] = marker_image

            # Place first marker in top-left
            y_pos, x_pos = 50, 50
            img[y_pos : y_pos + total_size, x_pos : x_pos + total_size] = cv2.cvtColor(
                marker_with_border, cv2.COLOR_GRAY2BGR
            )

            # Generate second marker (ID 5)
            marker_image2 = cv2.aruco.generateImageMarker(aruco_dict, 5, marker_size)
            marker_with_border2 = np.ones((total_size, total_size), dtype=np.uint8) * 255
            marker_with_border2[
                border_size : border_size + marker_size, border_size : border_size + marker_size
            ] = marker_image2

            # Place second marker in top-right
            x_pos2 = 1920 - total_size - 50
            img[y_pos : y_pos + total_size, x_pos2 : x_pos2 + total_size] = cv2.cvtColor(
                marker_with_border2, cv2.COLOR_GRAY2BGR
            )
        except Exception as e:
            logger.warning(f"Could not add ArUco markers to test image: {e}")

        # Add some test objects (rectangles) for edge/rotation detection
        cv2.rectangle(img, (800, 400), (1000, 600), (255, 100, 100), -1)  # Blue rectangle
        cv2.rectangle(img, (1100, 450), (1250, 550), (100, 255, 100), -1)  # Green rectangle
        cv2.circle(img, (950, 800), 80, (100, 100, 255), -1)  # Red circle

        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, (600, 950), font, 2, (255, 255, 255), 3)

        # Add timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(img, timestamp, (50, 1050), font, 1, (255, 255, 255), 2)

        # Add grid
        for x in range(0, 1920, 192):
            cv2.line(img, (x, 0), (x, 1080), (50, 50, 50), 1)
        for y in range(0, 1080, 108):
            cv2.line(img, (0, y), (1920, y), (50, 50, 50), 1)

        return img
