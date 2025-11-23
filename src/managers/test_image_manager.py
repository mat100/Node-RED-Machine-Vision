"""
Test Image Manager - Handles test image storage and management for testing without cameras
"""

import base64
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class TestImageManager:
    """Manages test images for testing vision algorithms without cameras"""

    def __init__(self, storage_path: str = "./test_images"):
        """
        Initialize Test Image Manager

        Args:
            storage_path: Path to store test image files
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Test image cache
        self.test_images: Dict[str, Dict] = {}
        self.test_image_data: Dict[str, np.ndarray] = {}

        # Thread safety
        self.lock = Lock()

        # Load existing test images
        self._load_test_images()

        logger.info(f"Test Image Manager initialized with path: {self.storage_path}")

    def _load_test_images(self):
        """Load test images from storage"""
        metadata_file = self.storage_path / "test_images.json"

        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    self.test_images = json.load(f)

                # Load test image data
                for test_id, info in self.test_images.items():
                    image_path = self.storage_path / info["filename"]
                    if image_path.exists():
                        img = cv2.imread(str(image_path))
                        if img is not None:
                            self.test_image_data[test_id] = img
                        else:
                            logger.warning(f"Failed to load test image: {image_path}")

                logger.info(f"Loaded {len(self.test_images)} test images")

            except Exception as e:
                logger.error(f"Failed to load test images: {e}")
                self.test_images = {}

    def _save_metadata(self):
        """Save test image metadata"""
        metadata_file = self.storage_path / "test_images.json"

        try:
            with open(metadata_file, "w") as f:
                json.dump(self.test_images, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save test image metadata: {e}")

    def upload(self, filename: str, image: np.ndarray) -> str:
        """
        Upload a new test image

        Args:
            filename: Original filename
            image: Test image data

        Returns:
            Test image ID
        """
        with self.lock:
            # Generate unique ID
            test_id = f"test_{uuid.uuid4().hex[:8]}"

            # Save image file (keep original extension if valid, otherwise use .png)
            original_ext = Path(filename).suffix.lower()
            valid_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"]
            ext = original_ext if original_ext in valid_extensions else ".png"
            save_filename = f"{test_id}{ext}"
            file_path = self.storage_path / save_filename

            try:
                # Save image
                cv2.imwrite(str(file_path), image)

                # Store metadata
                self.test_images[test_id] = {
                    "id": test_id,
                    "filename": save_filename,
                    "original_filename": filename,
                    "size": {"width": image.shape[1], "height": image.shape[0]},
                    "created_at": datetime.now().isoformat(),
                }

                # Cache image
                self.test_image_data[test_id] = image

                # Save metadata
                self._save_metadata()

                logger.info(f"Test image uploaded: {test_id} - {filename}")
                return test_id

            except Exception as e:
                logger.error(f"Failed to upload test image: {e}")
                if file_path.exists():
                    file_path.unlink()
                raise

    def get(self, test_id: str) -> Optional[np.ndarray]:
        """
        Get test image data

        Args:
            test_id: Test image identifier

        Returns:
            Test image or None if not found
        """
        with self.lock:
            if test_id in self.test_image_data:
                return self.test_image_data[test_id].copy()

            # Try to load from file
            if test_id in self.test_images:
                file_path = self.storage_path / self.test_images[test_id]["filename"]
                if file_path.exists():
                    img = cv2.imread(str(file_path))
                    if img is not None:
                        self.test_image_data[test_id] = img
                        return img.copy()

        logger.warning(f"Test image not found: {test_id}")
        return None

    def get_info(self, test_id: str) -> Optional[Dict]:
        """Get test image metadata"""
        with self.lock:
            if test_id in self.test_images:
                return self.test_images[test_id].copy()
        return None

    def list(self) -> List[Dict]:
        """List all test images"""
        with self.lock:
            return list(self.test_images.values())

    def delete(self, test_id: str) -> bool:
        """
        Delete a test image

        Args:
            test_id: Test image identifier

        Returns:
            True if deleted, False if not found
        """
        with self.lock:
            if test_id not in self.test_images:
                return False

            try:
                # Delete file
                file_path = self.storage_path / self.test_images[test_id]["filename"]
                if file_path.exists():
                    file_path.unlink()

                # Remove from cache
                if test_id in self.test_image_data:
                    del self.test_image_data[test_id]

                # Remove metadata
                del self.test_images[test_id]

                # Save metadata
                self._save_metadata()

                logger.info(f"Test image deleted: {test_id}")
                return True

            except Exception as e:
                logger.error(f"Failed to delete test image {test_id}: {e}")
                return False

    def create_thumbnail(self, test_id: str, max_width: int = 100) -> Optional[str]:
        """
        Create thumbnail for test image using OpenCV.

        Args:
            test_id: Test image identifier
            max_width: Maximum thumbnail width

        Returns:
            Base64 encoded thumbnail with data URI prefix or None
        """
        test_image = self.get(test_id)
        if test_image is None:
            return None

        # Calculate size maintaining aspect ratio
        aspect = test_image.shape[0] / test_image.shape[1]
        width = min(max_width, test_image.shape[1])
        height = int(width * aspect)

        # Resize using Lanczos interpolation for quality
        thumbnail = cv2.resize(test_image, (width, height), interpolation=cv2.INTER_LANCZOS4)

        # Encode to PNG
        success, buffer = cv2.imencode(".png", thumbnail)

        if not success:
            return None

        base64_str = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/png;base64,{base64_str}"
