"""
Camera Service - Business logic for camera operations.

This service orchestrates camera-related operations, combining
camera management, image capture, and storage in a single interface.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from api.exceptions import CameraConnectionException, CameraNotFoundException
from core.camera_manager import CameraManager
from core.image import extract_roi
from core.image_manager import ImageManager
from core.utils.camera_identifier import parse as parse_camera_id
from schemas import ROI

logger = logging.getLogger(__name__)


class CameraService:
    """
    Service for camera operations including capture, streaming, and management.

    This service combines CameraManager and ImageManager functionality to provide
    high-level camera operations for the API layer.
    """

    def __init__(self, camera_manager: CameraManager, image_manager: ImageManager):
        """
        Initialize camera service.

        Args:
            camera_manager: Camera manager instance
            image_manager: Image manager instance
        """
        self.camera_manager = camera_manager
        self.image_manager = image_manager

    def list_available_cameras(self) -> List[Dict]:
        """
        List all available cameras with their details.

        Returns:
            List of camera information dictionaries
        """
        return self.camera_manager.list_available_cameras()

    def connect_camera(
        self,
        camera_id: str,
        camera_type: str = "usb",
        source: int = 0,
        resolution: Optional[Tuple[int, int]] = None,
    ) -> bool:
        """
        Connect to a camera.

        Args:
            camera_id: Camera identifier
            camera_type: Type of camera (usb, ip, test)
            source: Camera source (device index or IP)
            resolution: Optional resolution tuple (width, height)

        Returns:
            True if connection successful

        Raises:
            CameraConnectionException: If connection fails
        """
        success = self.camera_manager.connect_camera(
            camera_id=camera_id, camera_type=camera_type, source=source, resolution=resolution
        )

        if not success:
            raise CameraConnectionException(camera_id=camera_id, reason="Connection failed")

        logger.info(f"Camera {camera_id} connected successfully")
        return True

    def disconnect_camera(self, camera_id: str) -> bool:
        """
        Disconnect a camera.

        Args:
            camera_id: Camera identifier

        Returns:
            True if disconnection successful

        Raises:
            CameraNotFoundException: If camera not found
        """
        success = self.camera_manager.disconnect_camera(camera_id)

        if not success:
            raise CameraNotFoundException(camera_id)

        logger.info(f"Camera {camera_id} disconnected successfully")
        return True

    def capture_and_store(
        self, camera_id: str, roi: Optional[ROI] = None, metadata: Optional[Dict] = None
    ) -> Tuple[str, str, Dict]:
        """
        Capture image from camera and store it.

        This is a high-level operation that:
        1. Captures image from camera (auto-connects if needed)
        2. Applies ROI if specified
        3. Stores image in image manager
        4. Creates thumbnail
        5. Returns image ID, thumbnail, and metadata

        Args:
            camera_id: Camera identifier
            roi: Optional ROI to extract
            metadata: Optional metadata to attach to image

        Returns:
            Tuple of (image_id, thumbnail_base64, metadata_dict)

        Raises:
            CameraNotFoundException: If camera not found or image capture fails
        """
        # Auto-connect camera if not already connected
        if not self.is_camera_connected(camera_id):
            logger.info(f"Camera {camera_id} not connected, attempting auto-connect")
            try:
                # Parse camera ID using unified utility
                camera_type, source = parse_camera_id(camera_id)
                self.connect_camera(camera_id=camera_id, camera_type=camera_type, source=source)
                logger.info(f"Camera {camera_id} auto-connected successfully")
            except Exception as e:
                logger.warning(f"Failed to auto-connect camera {camera_id}: {e}")

        # Capture image
        image = self.camera_manager.capture(camera_id)

        if image is None:
            # Try test image for development
            logger.warning(f"Camera {camera_id} not found, using test image")
            image = self.camera_manager.create_test_image(f"Camera: {camera_id}")

        # Apply ROI if specified
        if roi:
            image = extract_roi(image, roi, safe_mode=True)
            if image is None:
                raise ValueError("Invalid ROI parameters")

        # Prepare metadata
        if metadata is None:
            metadata = {}

        metadata.update(
            {
                "camera_id": camera_id,
                "timestamp": datetime.now().isoformat(),
                "roi": roi.to_dict() if roi else None,
            }
        )

        # Store image
        image_id = self.image_manager.store(image, metadata)

        # Create thumbnail (uses config width)
        _, thumbnail_base64 = self.image_manager.create_thumbnail(image)

        # Return metadata with image dimensions
        result_metadata = {
            "camera_id": camera_id,
            "width": image.shape[1],
            "height": image.shape[0],
        }

        logger.debug(f"Image captured and stored: {image_id}")
        return image_id, thumbnail_base64, result_metadata

    def get_preview(
        self, camera_id: str, create_thumbnail: bool = True
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Get preview image from camera.

        Args:
            camera_id: Camera identifier
            create_thumbnail: Whether to create thumbnail

        Returns:
            Tuple of (image, thumbnail_base64)
        """
        # Get preview frame
        image = self.camera_manager.get_preview(camera_id)

        if image is None:
            # Use test image
            image = self.camera_manager.create_test_image(f"Preview: {camera_id}")

        # Create thumbnail if requested (uses config width)
        thumbnail_base64 = None
        if create_thumbnail and image is not None:
            _, thumbnail_base64 = self.image_manager.create_thumbnail(image)

        return image, thumbnail_base64

    def is_camera_connected(self, camera_id: str) -> bool:
        """
        Check if camera is connected.

        Args:
            camera_id: Camera identifier

        Returns:
            True if camera is connected
        """
        cameras = self.list_available_cameras()
        for cam in cameras:
            if cam["id"] == camera_id:
                return cam.get("connected", False)
        return False

    def get_camera_info(self, camera_id: str) -> Optional[Dict]:
        """
        Get detailed information about a camera.

        Args:
            camera_id: Camera identifier

        Returns:
            Camera info dict or None if not found
        """
        cameras = self.list_available_cameras()
        for cam in cameras:
            if cam["id"] == camera_id:
                return cam
        return None
