"""
Unit tests for CameraService
"""

from unittest.mock import patch

import numpy as np
import pytest

from api.exceptions import CameraConnectionException, CameraNotFoundException
from schemas import ROI
from services.camera_service import CameraService


class TestCameraService:
    """Test CameraService functionality"""

    def test_list_available_cameras(self, camera_service):
        """Test listing available cameras"""
        cameras = camera_service.list_available_cameras()
        assert isinstance(cameras, list)
        # Test camera should be in the list
        test_cam = next((c for c in cameras if c["id"] == "test"), None)
        assert test_cam is not None
        assert test_cam["name"] == "Test Image Generator"

    def test_connect_camera_success(self, mock_camera_manager, mock_image_manager):
        """Test successful camera connection"""
        service = CameraService(mock_camera_manager, mock_image_manager)
        mock_camera_manager.connect_camera.return_value = True

        result = service.connect_camera(camera_id="test", camera_type="test")

        assert result is True
        mock_camera_manager.connect_camera.assert_called_once()

    def test_connect_camera_failure(self, mock_camera_manager, mock_image_manager):
        """Test camera connection failure"""
        service = CameraService(mock_camera_manager, mock_image_manager)
        mock_camera_manager.connect_camera.return_value = False

        with pytest.raises(CameraConnectionException):
            service.connect_camera(camera_id="test", camera_type="test")

    def test_disconnect_camera(self, mock_camera_manager, mock_image_manager):
        """Test camera disconnection"""
        service = CameraService(mock_camera_manager, mock_image_manager)
        mock_camera_manager.disconnect_camera.return_value = True

        result = service.disconnect_camera("test")

        assert result is True
        mock_camera_manager.disconnect_camera.assert_called_once_with("test")

    def test_disconnect_camera_not_found(self, mock_camera_manager, mock_image_manager):
        """Test disconnection of non-existent camera"""
        service = CameraService(mock_camera_manager, mock_image_manager)
        mock_camera_manager.disconnect_camera.return_value = False

        with pytest.raises(CameraNotFoundException):
            service.disconnect_camera("non-existent")

    def test_capture_and_store_basic(self, mock_camera_manager, mock_image_manager):
        """Test basic image capture and storage"""
        service = CameraService(mock_camera_manager, mock_image_manager)

        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_camera_manager.capture.return_value = test_image
        mock_image_manager.store.return_value = "test-image-id"
        mock_image_manager.create_thumbnail.return_value = (test_image, "base64-thumb")

        image_id, thumbnail, metadata = service.capture_and_store(camera_id="test")

        assert image_id == "test-image-id"
        assert thumbnail == "base64-thumb"
        assert metadata["camera_id"] == "test"
        assert metadata["width"] == 640
        assert metadata["height"] == 480
        mock_camera_manager.capture.assert_called_once()
        mock_image_manager.store.assert_called_once()

    def test_capture_and_store_with_roi(self, mock_camera_manager, mock_image_manager):
        """Test image capture with ROI extraction"""
        service = CameraService(mock_camera_manager, mock_image_manager)

        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        # After ROI extraction, image will be 200x150
        roi_image = np.zeros((150, 200, 3), dtype=np.uint8)
        mock_camera_manager.capture.return_value = test_image
        mock_image_manager.store.return_value = "test-image-id"
        mock_image_manager.create_thumbnail.return_value = (roi_image, "base64-thumb")

        roi = ROI(x=100, y=100, width=200, height=150)

        # Mock extract_roi to return the ROI image
        with patch("services.camera_service.extract_roi", return_value=roi_image):
            image_id, thumbnail, metadata = service.capture_and_store(camera_id="test", roi=roi)

        assert image_id == "test-image-id"
        # Metadata should reflect ROI dimensions
        assert metadata["width"] == 200
        assert metadata["height"] == 150

    def test_capture_and_store_invalid_roi(self, mock_camera_manager, mock_image_manager):
        """Test capture with invalid ROI that returns None"""
        service = CameraService(mock_camera_manager, mock_image_manager)

        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_camera_manager.capture.return_value = test_image

        roi = ROI(x=1000, y=1000, width=200, height=150)  # Out of bounds

        # Mock extract_roi to return None for invalid ROI
        with patch("services.camera_service.extract_roi", return_value=None):
            with pytest.raises(ValueError, match="Invalid ROI parameters"):
                service.capture_and_store(camera_id="test", roi=roi)

    def test_capture_and_store_fallback_to_test(self, mock_camera_manager, mock_image_manager):
        """Test fallback to test image when capture fails"""
        service = CameraService(mock_camera_manager, mock_image_manager)

        # Simulate capture returning None
        mock_camera_manager.capture.return_value = None
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_camera_manager.create_test_image.return_value = test_image
        mock_image_manager.store.return_value = "test-image-id"
        mock_image_manager.create_thumbnail.return_value = (test_image, "base64-thumb")

        image_id, thumbnail, metadata = service.capture_and_store(camera_id="test")

        assert image_id == "test-image-id"
        mock_camera_manager.create_test_image.assert_called_once()

    def test_get_preview(self, mock_camera_manager, mock_image_manager):
        """Test getting camera preview"""
        service = CameraService(mock_camera_manager, mock_image_manager)

        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_camera_manager.get_preview.return_value = test_image
        mock_image_manager.create_thumbnail.return_value = (test_image, "base64-thumb")

        image, thumbnail = service.get_preview(camera_id="test", create_thumbnail=True)

        assert image is not None
        assert thumbnail == "base64-thumb"
        mock_camera_manager.get_preview.assert_called_once_with("test")

    def test_get_preview_no_thumbnail(self, mock_camera_manager, mock_image_manager):
        """Test getting preview without thumbnail"""
        service = CameraService(mock_camera_manager, mock_image_manager)

        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_camera_manager.get_preview.return_value = test_image

        image, thumbnail = service.get_preview(camera_id="test", create_thumbnail=False)

        assert image is not None
        assert thumbnail is None
        mock_image_manager.create_thumbnail.assert_not_called()

    def test_capture_with_metadata(self, mock_camera_manager, mock_image_manager):
        """Test capture with custom metadata"""
        service = CameraService(mock_camera_manager, mock_image_manager)

        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_camera_manager.capture.return_value = test_image
        mock_image_manager.store.return_value = "test-image-id"
        mock_image_manager.create_thumbnail.return_value = (test_image, "base64-thumb")

        custom_metadata = {"custom_field": "custom_value", "batch_id": "123"}
        image_id, thumbnail, metadata = service.capture_and_store(
            camera_id="test", metadata=custom_metadata
        )

        # Returned metadata only contains camera_id, width, height
        # Custom metadata is stored with image but not returned in result
        assert metadata["camera_id"] == "test"
        assert metadata["width"] == 640
        assert metadata["height"] == 480


class TestCameraServiceIntegration:
    """Integration tests with real managers"""

    def test_full_capture_workflow(self, camera_service, test_image):
        """Test complete capture workflow with real managers"""
        # Note: This uses real CameraManager with test camera
        image_id, thumbnail, metadata = camera_service.capture_and_store(camera_id="test")

        assert image_id is not None
        assert len(image_id) > 0
        assert thumbnail is not None
        assert metadata["camera_id"] == "test"

    def test_capture_with_roi_integration(self, camera_service):
        """Test capture with ROI using real managers"""
        roi = ROI(x=100, y=100, width=300, height=200)

        image_id, thumbnail, metadata = camera_service.capture_and_store(camera_id="test", roi=roi)

        assert image_id is not None
        # After ROI applied, metadata should reflect ROI dimensions
        assert metadata["width"] == 300
        assert metadata["height"] == 200
        assert metadata["camera_id"] == "test"
