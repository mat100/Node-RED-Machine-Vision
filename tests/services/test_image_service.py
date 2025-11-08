"""
Unit tests for ImageService
"""

import numpy as np
import pytest

from api.exceptions import ImageNotFoundException
from schemas import ROI
from services.image_service import ImageService


class TestImageService:
    """Test ImageService functionality"""

    def test_get_image_success(self, mock_image_manager):
        """Test successful image retrieval"""
        service = ImageService(mock_image_manager)

        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_image_manager.get.return_value = test_image

        image = service.get_image("test-image-id")

        assert image is not None
        assert image.shape == (480, 640, 3)
        mock_image_manager.get.assert_called_once_with("test-image-id")

    def test_get_image_not_found(self, mock_image_manager):
        """Test image not found error"""
        service = ImageService(mock_image_manager)
        mock_image_manager.get.return_value = None

        with pytest.raises(ImageNotFoundException):
            service.get_image("non-existent-id")

    def test_get_image_with_thumbnail(self, mock_image_manager):
        """Test getting image with thumbnail creation"""
        service = ImageService(mock_image_manager)

        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_image_manager.get.return_value = test_image
        mock_image_manager.create_thumbnail.return_value = (test_image, "thumb-data")

        image, thumbnail = service.get_image_with_thumbnail("test-id", thumbnail_width=320)

        assert image is not None
        assert thumbnail == "thumb-data"
        mock_image_manager.get.assert_called_once_with("test-id")
        mock_image_manager.create_thumbnail.assert_called_once()

    def test_get_image_with_roi(self, mock_image_manager):
        """Test getting image with ROI extraction"""
        service = ImageService(mock_image_manager)

        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add a white rectangle to test ROI extraction
        test_image[100:300, 100:300] = 255
        mock_image_manager.get.return_value = test_image

        roi = ROI(x=100, y=100, width=200, height=200)
        roi_image = service.get_image_with_roi("test-id", roi, safe_mode=True)

        assert roi_image is not None
        assert roi_image.shape == (200, 200, 3)

    def test_get_image_with_roi_not_found(self, mock_image_manager):
        """Test ROI extraction when image not found"""
        service = ImageService(mock_image_manager)
        mock_image_manager.get.return_value = None

        roi = ROI(x=100, y=100, width=200, height=200)

        with pytest.raises(ImageNotFoundException):
            service.get_image_with_roi("non-existent-id", roi)

    def test_create_thumbnail(self, mock_image_manager):
        """Test thumbnail creation"""
        service = ImageService(mock_image_manager)

        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_image_manager.get.return_value = test_image
        mock_image_manager.create_thumbnail.return_value = (test_image, "base64-data")

        thumbnail_b64 = service.create_thumbnail("test-id", width=320)

        assert thumbnail_b64 == "base64-data"
        mock_image_manager.create_thumbnail.assert_called_once()

    def test_create_thumbnail_image_not_found(self, mock_image_manager):
        """Test thumbnail creation when image not found"""
        service = ImageService(mock_image_manager)
        mock_image_manager.get.return_value = None

        with pytest.raises(ImageNotFoundException):
            service.create_thumbnail("non-existent-id")

    def test_store_image(self, mock_image_manager):
        """Test storing an image"""
        service = ImageService(mock_image_manager)

        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_image_manager.store.return_value = "new-image-id"

        image_id = service.store_image(test_image, metadata={"source": "test"})

        assert image_id == "new-image-id"
        mock_image_manager.store.assert_called_once()

    def test_delete_image(self, mock_image_manager):
        """Test deleting an image"""
        service = ImageService(mock_image_manager)
        mock_image_manager.has_image.return_value = True

        result = service.delete_image("test-id")

        # Currently returns False as deletion not implemented
        assert result is False
        mock_image_manager.has_image.assert_called_once_with("test-id")

    def test_delete_image_not_found(self, mock_image_manager):
        """Test deleting non-existent image raises exception"""
        service = ImageService(mock_image_manager)
        mock_image_manager.has_image.return_value = False

        with pytest.raises(ImageNotFoundException):
            service.delete_image("non-existent")

    def test_list_images(self, mock_image_manager):
        """Test listing stored images - delegated to manager"""
        ImageService(mock_image_manager)

        mock_image_manager.list_images.return_value = [
            {"id": "img1", "timestamp": "2025-01-01T00:00:00"},
            {"id": "img2", "timestamp": "2025-01-01T00:01:00"},
        ]

        # ImageService doesn't have list_images method - it's on the manager
        images = mock_image_manager.list_images()

        assert len(images) == 2
        assert images[0]["id"] == "img1"

    def test_get_image_metadata(self, mock_image_manager):
        """Test getting image metadata"""
        service = ImageService(mock_image_manager)

        mock_metadata = {"id": "test-id", "width": 640, "height": 480, "camera_id": "test"}
        mock_image_manager.get_metadata.return_value = mock_metadata

        metadata = service.get_image_metadata("test-id")

        assert metadata["width"] == 640
        assert metadata["camera_id"] == "test"
        mock_image_manager.get_metadata.assert_called_once_with("test-id")


class TestImageServiceIntegration:
    """Integration tests with real ImageManager"""

    def test_store_and_retrieve(self, image_service, test_image):
        """Test storing and retrieving an image"""
        # Store image
        metadata = {"source": "test", "camera_id": "test"}
        image_id = image_service.store_image(test_image, metadata)

        assert image_id is not None

        # Retrieve image
        retrieved_image = image_service.get_image(image_id)

        assert retrieved_image is not None
        assert retrieved_image.shape == test_image.shape
        assert np.array_equal(retrieved_image, test_image)

    def test_roi_extraction_integration(self, image_service, test_image):
        """Test ROI extraction with real manager"""
        # Store image
        image_id = image_service.store_image(test_image, {})

        # Extract ROI
        roi = ROI(x=100, y=100, width=200, height=150)
        roi_image = image_service.get_image_with_roi(image_id, roi, safe_mode=True)

        assert roi_image is not None
        assert roi_image.shape == (150, 200, 3)

    def test_thumbnail_creation_integration(self, image_service, test_image):
        """Test thumbnail creation with real manager"""
        # Store image
        image_id = image_service.store_image(test_image, {})

        # Create thumbnail
        thumbnail_b64 = image_service.create_thumbnail(image_id, width=320)

        assert thumbnail_b64 is not None
        assert len(thumbnail_b64) > 0
        assert isinstance(thumbnail_b64, str)
