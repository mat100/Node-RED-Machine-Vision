"""
Image Service - Business logic for image operations.

This service provides high-level image operations including
retrieval, processing, and metadata management.
"""

import logging
import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from api.exceptions import ImageNotFoundException
from core.constants import ImageConstants
from core.image import extract_roi
from core.image_manager import ImageManager
from schemas import ROI

logger = logging.getLogger(__name__)


class ImageService:
    """
    Service for image operations including storage, retrieval, and processing.

    This service provides high-level image operations for the API layer.
    """

    def __init__(self, image_manager: ImageManager):
        """
        Initialize image service.

        Args:
            image_manager: Image manager instance
        """
        self.image_manager = image_manager

    def get_image(self, image_id: str) -> np.ndarray:
        """
        Get image by ID.

        Args:
            image_id: Image identifier

        Returns:
            Image as numpy array

        Raises:
            ImageNotFoundException: If image not found
        """
        image = self.image_manager.get(image_id)
        if image is None:
            raise ImageNotFoundException(image_id)
        return image

    def store_image(self, image: np.ndarray, metadata: Optional[Dict] = None) -> str:
        """
        Store image with optional metadata.

        Args:
            image: Image as numpy array
            metadata: Optional metadata dictionary

        Returns:
            Image ID
        """
        image_id = self.image_manager.store(image, metadata or {})
        logger.debug(f"Image stored: {image_id}")
        return image_id

    def import_from_file(self, file_path: str) -> Tuple[str, str, Dict]:
        """
        Import image from file system.

        Loads an image from the file system (JPG, PNG, BMP, etc.),
        stores it in ImageManager, and generates a thumbnail.

        Args:
            file_path: Path to image file (can be in /dev/shm or regular filesystem)

        Returns:
            Tuple of (image_id, thumbnail_base64, metadata)

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file cannot be loaded as image
        """
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")

        # Load image with OpenCV
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image from: {file_path}")

        # Extract metadata
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1
        file_size = os.path.getsize(file_path)

        metadata = {
            "source": "file_import",
            "file_path": file_path,
            "width": width,
            "height": height,
            "channels": channels,
            "file_size_bytes": file_size,
        }

        # Store in ImageManager
        image_id = self.image_manager.store(image, metadata)

        # Generate thumbnail
        _, thumbnail_base64 = self.image_manager.create_thumbnail(
            image, width=None, image_id=image_id
        )

        logger.info(
            f"Imported image from {file_path}: {image_id} ({width}x{height}, {file_size} bytes)"
        )

        return image_id, thumbnail_base64, metadata

    def get_image_with_thumbnail(
        self, image_id: str, thumbnail_width: int = ImageConstants.DEFAULT_THUMBNAIL_WIDTH
    ) -> Tuple[np.ndarray, str]:
        """
        Get image and its thumbnail.

        Args:
            image_id: Image identifier
            thumbnail_width: Thumbnail width in pixels

        Returns:
            Tuple of (image, thumbnail_base64)

        Raises:
            ImageNotFoundException: If image not found
        """
        image = self.get_image(image_id)

        _, thumbnail_base64 = self.image_manager.create_thumbnail(image, thumbnail_width)

        return image, thumbnail_base64

    def get_image_with_roi(self, image_id: str, roi: ROI, safe_mode: bool = True) -> np.ndarray:
        """
        Get image with ROI extracted.

        Args:
            image_id: Image identifier
            roi: ROI to extract
            safe_mode: Whether to clip ROI to image bounds

        Returns:
            Extracted ROI as numpy array

        Raises:
            ImageNotFoundException: If image not found
            ValueError: If ROI is invalid
        """
        image = self.get_image(image_id)

        roi_image = extract_roi(image, roi, safe_mode=safe_mode)
        if roi_image is None:
            raise ValueError("Invalid ROI parameters")

        return roi_image

    def create_thumbnail(
        self, image_id: str, width: int = ImageConstants.DEFAULT_THUMBNAIL_WIDTH
    ) -> str:
        """
        Create thumbnail for an image.

        Args:
            image_id: Image identifier
            width: Thumbnail width in pixels

        Returns:
            Thumbnail as base64 string

        Raises:
            ImageNotFoundException: If image not found
        """
        image = self.get_image(image_id)

        _, thumbnail_base64 = self.image_manager.create_thumbnail(image, width)
        return thumbnail_base64

    def get_image_metadata(self, image_id: str) -> Optional[Dict]:
        """
        Get metadata for an image.

        Args:
            image_id: Image identifier

        Returns:
            Metadata dictionary or None if not found
        """
        return self.image_manager.get_metadata(image_id)

    def get_stats(self) -> Dict:
        """
        Get image manager statistics.

        Returns:
            Statistics dictionary with:
            - total_images: Number of images stored
            - total_size_mb: Total size in MB
            - max_size_mb: Maximum size in MB
            - usage_percent: Usage percentage
            - referenced_images: Number of referenced images
        """
        return self.image_manager.get_stats()

    def cleanup_old_images(self) -> int:
        """
        Clean up old images to free memory.

        Returns:
            Number of images removed
        """
        stats_before = self.get_stats()
        total_before = stats_before["total_images"]

        self.image_manager.cleanup()

        stats_after = self.get_stats()
        total_after = stats_after["total_images"]

        removed = total_before - total_after
        if removed > 0:
            logger.info(f"Cleaned up {removed} old images")

        return removed

    def delete_image(self, image_id: str) -> bool:
        """
        Delete a specific image.

        Args:
            image_id: Image identifier

        Returns:
            True if image was deleted

        Raises:
            ImageNotFoundException: If image not found
        """
        if not self.image_manager.has_image(image_id):
            raise ImageNotFoundException(image_id)

        # Note: Current ImageManager doesn't have delete method
        # This would need to be implemented in ImageManager
        logger.warning("Image deletion not yet implemented in ImageManager")
        return False
