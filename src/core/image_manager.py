"""
Image Manager - Handles image storage with shared memory and LRU cache
"""

import logging
import time
import uuid
from collections import OrderedDict
from multiprocessing import shared_memory
from threading import Lock
from typing import Dict, Optional, Tuple

import numpy as np

from core.constants import ImageConstants
from core.image.processors import create_thumbnail

logger = logging.getLogger(__name__)


class ImageManager:
    """
    Manages image storage using shared memory for zero-copy access
    and LRU cache for memory management
    """

    def __init__(
        self,
        max_size_mb: int = ImageConstants.DEFAULT_MAX_MEMORY_MB,
        max_images: int = ImageConstants.DEFAULT_MAX_IMAGES,
        thumbnail_width: int = ImageConstants.DEFAULT_THUMBNAIL_WIDTH,
    ):
        """
        Initialize Image Manager

        Args:
            max_size_mb: Maximum total size in megabytes
            max_images: Maximum number of images to store
            thumbnail_width: Width for thumbnail generation (from config)
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_images = max_images
        self.current_size = 0
        self.thumbnail_width = thumbnail_width

        # LRU cache for image metadata
        self.cache: OrderedDict = OrderedDict()

        # Shared memory segments
        self.shared_memories: Dict[str, shared_memory.SharedMemory] = {}

        # Thread safety
        self.lock = Lock()

        # Reference counting
        self.ref_counts: Dict[str, int] = {}

        # Thumbnail cache - stores base64 thumbnails to avoid regeneration
        # Key: image_id, Value: base64 thumbnail string
        self.thumbnail_cache: Dict[str, str] = {}

        logger.info(
            f"Image Manager initialized: {max_size_mb}MB, "
            f"max {max_images} images, thumbnail_width={thumbnail_width}px"
        )

    def store(self, image: np.ndarray, metadata: Optional[Dict] = None) -> str:
        """
        Store image in shared memory

        Args:
            image: NumPy array (OpenCV image)
            metadata: Optional metadata

        Returns:
            Image ID for retrieval
        """
        with self.lock:
            # Generate unique ID
            image_id = str(uuid.uuid4())

            # Calculate size
            image_size = image.nbytes

            # Check if we need to free space
            self._ensure_space(image_size)

            try:
                # Create shared memory
                shm = shared_memory.SharedMemory(create=True, size=image_size)

                # Copy image to shared memory
                shared_array = np.ndarray(image.shape, dtype=image.dtype, buffer=shm.buf)
                shared_array[:] = image[:]

                # Store in cache
                self.cache[image_id] = {
                    "shape": image.shape,
                    "dtype": image.dtype,
                    "size": image_size,
                    "shm_name": shm.name,
                    "timestamp": time.time(),
                    "metadata": metadata or {},
                }

                # Store shared memory reference
                self.shared_memories[image_id] = shm

                # Initialize reference count
                self.ref_counts[image_id] = 0

                # Update current size
                self.current_size += image_size

                # Move to end (most recent)
                self.cache.move_to_end(image_id)

                logger.debug(f"Stored image {image_id}: {image.shape}, {image_size} bytes")
                return image_id

            except Exception as e:
                logger.error(f"Failed to store image: {e}")
                if image_id in self.shared_memories:
                    self.shared_memories[image_id].close()
                    self.shared_memories[image_id].unlink()
                raise

    def get(self, image_id: str) -> Optional[np.ndarray]:
        """
        Retrieve image from shared memory

        Args:
            image_id: Image identifier

        Returns:
            NumPy array or None if not found
        """
        with self.lock:
            if image_id not in self.cache:
                logger.warning(f"Image {image_id} not found")
                return None

            try:
                # Get metadata
                info = self.cache[image_id]

                # Get shared memory
                shm = self.shared_memories[image_id]

                # Create array view
                image = np.ndarray(
                    info["shape"], dtype=info["dtype"], buffer=shm.buf
                ).copy()  # Copy to avoid issues when shm is released

                # Update access time and move to end
                info["last_access"] = time.time()
                self.cache.move_to_end(image_id)

                # Increment reference count
                self.ref_counts[image_id] += 1

                logger.debug(f"Retrieved image {image_id}")
                return image

            except Exception as e:
                logger.error(f"Failed to retrieve image {image_id}: {e}")
                return None

    def get_metadata(self, image_id: str) -> Optional[Dict]:
        """Get image metadata without loading the image"""
        with self.lock:
            if image_id in self.cache:
                return self.cache[image_id].copy()
            return None

    def delete(self, image_id: str) -> bool:
        """
        Delete image from cache

        Args:
            image_id: Image identifier

        Returns:
            True if deleted, False if not found or still referenced
        """
        with self.lock:
            if image_id not in self.cache:
                return False

            # Check reference count
            if self.ref_counts.get(image_id, 0) > 0:
                logger.warning(f"Cannot delete image {image_id}: still referenced")
                return False

            return self._delete_image(image_id)

    def _delete_image(self, image_id: str) -> bool:
        """Internal method to delete image"""
        try:
            # Get info
            info = self.cache[image_id]

            # Close and unlink shared memory
            if image_id in self.shared_memories:
                shm = self.shared_memories[image_id]
                shm.close()
                shm.unlink()
                del self.shared_memories[image_id]

            # Update size
            self.current_size -= info["size"]

            # Remove from cache
            del self.cache[image_id]
            del self.ref_counts[image_id]

            # Remove from thumbnail cache
            self.thumbnail_cache.pop(image_id, None)

            logger.debug(f"Deleted image {image_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete image {image_id}: {e}")
            return False

    def _ensure_space(self, required_size: int):
        """Ensure enough space by removing old images if necessary"""
        # Check image count
        while len(self.cache) >= self.max_images:
            self._evict_oldest()

        # Check total size
        while self.current_size + required_size > self.max_size_bytes:
            if not self._evict_oldest():
                raise MemoryError("Cannot free enough space")

    def _evict_oldest(self) -> bool:
        """Evict oldest unreferenced image"""
        for image_id in list(self.cache.keys()):
            if self.ref_counts.get(image_id, 0) == 0:
                self._delete_image(image_id)
                return True
        return False

    def release_reference(self, image_id: str):
        """Decrement reference count"""
        with self.lock:
            if image_id in self.ref_counts:
                self.ref_counts[image_id] = max(0, self.ref_counts[image_id] - 1)

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with self.lock:
            return {
                "total_images": len(self.cache),
                "total_size_mb": self.current_size / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "usage_percent": (self.current_size / self.max_size_bytes) * 100,
                "referenced_images": sum(1 for c in self.ref_counts.values() if c > 0),
                "cached_thumbnails": len(self.thumbnail_cache),
            }

    def cleanup(self):
        """Clean up all shared memory"""
        with self.lock:
            logger.info("Cleaning up Image Manager...")

            for image_id in list(self.cache.keys()):
                self._delete_image(image_id)

            self.cache.clear()
            self.shared_memories.clear()
            self.ref_counts.clear()
            self.thumbnail_cache.clear()
            self.current_size = 0

            logger.info("Image Manager cleanup complete")

    def create_thumbnail(
        self, image: np.ndarray, width: Optional[int] = None, image_id: Optional[str] = None
    ) -> Tuple[np.ndarray, str]:
        """
        Create thumbnail from image with caching support.

        Args:
            image: Source image
            width: Target width (uses config value if None)
            image_id: Optional image ID for caching (if provided, enables cache)

        Returns:
            Thumbnail array and base64 string
        """
        # Use config width if not specified
        if width is None:
            width = self.thumbnail_width

        # Check cache if image_id provided and using default width
        if image_id and width == self.thumbnail_width:
            cached_thumbnail = self.thumbnail_cache.get(image_id)
            if cached_thumbnail:
                # Cache hit - return cached thumbnail
                # Note: We don't have the array, so regenerate only if needed
                # For now, return None for array when using cache
                logger.debug(f"Thumbnail cache hit for {image_id}")
                return None, cached_thumbnail

        # Use centralized create_thumbnail function
        thumbnail_array, thumbnail_base64 = create_thumbnail(
            image=image, width=width, maintain_aspect=True
        )

        # Add data URI prefix for compatibility
        thumbnail_base64_with_prefix = f"data:image/jpeg;base64,{thumbnail_base64}"

        # Cache thumbnail if image_id provided and using default width
        if image_id and width == self.thumbnail_width:
            with self.lock:
                self.thumbnail_cache[image_id] = thumbnail_base64_with_prefix
                logger.debug(f"Cached thumbnail for {image_id}")

        return thumbnail_array, thumbnail_base64_with_prefix
