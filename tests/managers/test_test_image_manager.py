"""
Tests for managers.test_image_manager module.

Tests test image storage, retrieval, and management operations for testing without cameras.
"""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from managers.test_image_manager import TestImageManager


class TestTestImageManager:
    """Tests for TestImageManager class."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def manager(self, temp_storage):
        """Create TestImageManager instance with temp storage."""
        return TestImageManager(storage_path=temp_storage)

    @pytest.fixture
    def test_image(self):
        """Create a test image."""
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (250, 150), (255, 128, 64), -1)
        cv2.circle(image, (150, 100), 30, (0, 255, 255), -1)
        return image

    def test_test_image_manager_creation(self, temp_storage):
        """Test creating TestImageManager."""
        manager = TestImageManager(storage_path=temp_storage)

        assert manager.storage_path == Path(temp_storage)
        assert manager.storage_path.exists()
        assert len(manager.test_images) == 0
        assert len(manager.test_image_data) == 0

    def test_upload_test_image_basic(self, manager, test_image):
        """Test uploading a test image."""
        test_id = manager.upload("test_image.png", test_image)

        assert test_id is not None
        assert isinstance(test_id, str)
        assert test_id.startswith("test_")
        assert test_id in manager.test_images
        assert test_id in manager.test_image_data

    def test_upload_test_image_filename_stored(self, manager, test_image):
        """Test that original filename is stored."""
        test_id = manager.upload("my_test_image.jpg", test_image)

        assert manager.test_images[test_id]["original_filename"] == "my_test_image.jpg"

    def test_upload_multiple_test_images(self, manager, test_image):
        """Test uploading multiple test images."""
        id1 = manager.upload("test1.png", test_image)
        id2 = manager.upload("test2.png", test_image)

        assert len(manager.test_images) == 2
        assert id1 != id2
        assert id1 in manager.test_image_data
        assert id2 in manager.test_image_data

    def test_get_test_image_existing(self, manager, test_image):
        """Test retrieving existing test image."""
        test_id = manager.upload("test.png", test_image)

        retrieved = manager.get(test_id)

        assert retrieved is not None
        assert isinstance(retrieved, np.ndarray)
        assert retrieved.shape == test_image.shape
        # Ensure we get a copy, not the original
        assert retrieved is not manager.test_image_data[test_id]

    def test_get_test_image_nonexistent(self, manager):
        """Test retrieving non-existent test image."""
        result = manager.get("nonexistent-id")

        assert result is None

    def test_get_info_existing(self, manager, test_image):
        """Test retrieving test image metadata."""
        test_id = manager.upload("test.png", test_image)

        info = manager.get_info(test_id)

        assert info is not None
        assert info["id"] == test_id
        assert info["original_filename"] == "test.png"
        assert info["size"]["width"] == test_image.shape[1]
        assert info["size"]["height"] == test_image.shape[0]
        assert "created_at" in info

    def test_get_info_nonexistent(self, manager):
        """Test retrieving info for non-existent test image."""
        result = manager.get_info("nonexistent-id")

        assert result is None

    def test_list_test_images_empty(self, manager):
        """Test listing test images when empty."""
        test_images = manager.list()

        assert isinstance(test_images, list)
        assert len(test_images) == 0

    def test_list_test_images_with_data(self, manager, test_image):
        """Test listing test images with data."""
        manager.upload("test1.png", test_image)
        manager.upload("test2.png", test_image)

        test_images = manager.list()

        assert len(test_images) == 2
        assert all("id" in img for img in test_images)
        assert all("original_filename" in img for img in test_images)

    def test_delete_test_image_existing(self, manager, test_image):
        """Test deleting existing test image."""
        test_id = manager.upload("test.png", test_image)

        success = manager.delete(test_id)

        assert success is True
        assert test_id not in manager.test_images
        assert test_id not in manager.test_image_data

    def test_delete_test_image_nonexistent(self, manager):
        """Test deleting non-existent test image."""
        success = manager.delete("nonexistent-id")

        assert success is False

    def test_create_thumbnail(self, manager, test_image):
        """Test creating thumbnail."""
        test_id = manager.upload("test.png", test_image)

        thumbnail = manager.create_thumbnail(test_id, max_width=100)

        assert thumbnail is not None
        assert isinstance(thumbnail, str)
        assert thumbnail.startswith("data:image/png;base64,")

    def test_create_thumbnail_nonexistent(self, manager):
        """Test creating thumbnail for non-existent test image."""
        thumbnail = manager.create_thumbnail("nonexistent-id")

        assert thumbnail is None

    def test_persistence(self, temp_storage, test_image):
        """Test that test images persist across manager instances."""
        # Create manager and upload test image
        manager1 = TestImageManager(storage_path=temp_storage)
        test_id = manager1.upload("test.png", test_image)

        # Create new manager instance with same storage
        manager2 = TestImageManager(storage_path=temp_storage)

        # Verify test image was loaded
        assert test_id in manager2.test_images
        retrieved = manager2.get(test_id)
        assert retrieved is not None
        assert retrieved.shape == test_image.shape

    def test_metadata_file_created(self, manager, test_image):
        """Test that metadata file is created."""
        manager.upload("test.png", test_image)

        metadata_file = manager.storage_path / "test_images.json"
        assert metadata_file.exists()

        # Verify metadata is valid JSON
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        assert isinstance(metadata, dict)
        assert len(metadata) == 1

    def test_file_extension_preserved(self, manager, test_image):
        """Test that valid file extensions are preserved."""
        for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
            test_id = manager.upload(f"test{ext}", test_image)
            assert manager.test_images[test_id]["filename"].endswith(ext)

    def test_invalid_extension_defaults_to_png(self, manager, test_image):
        """Test that invalid extensions default to .png."""
        test_id = manager.upload("test.invalid", test_image)
        assert manager.test_images[test_id]["filename"].endswith(".png")

    def test_thread_safety_upload(self, manager, test_image):
        """Test thread-safe upload operations."""
        import threading

        results = []

        def upload_image(name):
            test_id = manager.upload(name, test_image)
            results.append(test_id)

        # Create multiple threads
        threads = [threading.Thread(target=upload_image, args=(f"test{i}.png",)) for i in range(5)]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all to complete
        for thread in threads:
            thread.join()

        # Verify all uploads succeeded with unique IDs
        assert len(results) == 5
        assert len(set(results)) == 5  # All IDs are unique
        assert all(tid in manager.test_images for tid in results)
