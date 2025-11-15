"""
Tests for core.camera_manager module.

Tests camera management including connection, capture, and test image generation.
"""

import numpy as np
import pytest

from common.enums import CameraType
from core.camera_manager import Camera, CameraManager, CameraSettings


class TestCameraSettings:
    """Tests for CameraSettings dataclass."""

    def test_camera_settings_creation(self):
        """Test creating camera settings."""
        settings = CameraSettings(
            id="test_cam",
            name="Test Camera",
            type=CameraType.USB,
            source=0,
            resolution=(1920, 1080),
            fps=30,
        )

        assert settings.id == "test_cam"
        assert settings.name == "Test Camera"
        assert settings.type == CameraType.USB
        assert settings.source == 0
        assert settings.resolution == (1920, 1080)
        assert settings.fps == 30

    def test_camera_settings_defaults(self):
        """Test camera settings with defaults."""
        settings = CameraSettings(id="test", name="Test", type=CameraType.TEST, source="test")

        # Should have default values
        assert settings.resolution == (1920, 1080)
        assert settings.fps == 30
        assert settings.capture_timeout_ms == 5000


class TestCamera:
    """Tests for Camera class."""

    def test_camera_creation(self):
        """Test creating a Camera instance."""
        config = CameraSettings(id="test", name="Test", type=CameraType.TEST, source="test")
        camera = Camera(config)

        assert camera.config == config
        assert camera.connected is False
        assert camera.cap is None
        assert camera.last_frame is None

    def test_test_camera_connect(self):
        """Test connecting a TEST camera."""
        config = CameraSettings(id="test", name="Test", type=CameraType.TEST, source="test")
        camera = Camera(config)

        result = camera.connect()

        assert result is True
        assert camera.connected is True
        # TEST camera doesn't use VideoCapture
        assert camera.cap is None

    def test_test_camera_capture(self):
        """Test capturing from TEST camera."""
        config = CameraSettings(id="test", name="Test", type=CameraType.TEST, source="test")
        camera = Camera(config)
        camera.connect()

        frame = camera.capture()

        assert frame is not None
        assert isinstance(frame, np.ndarray)
        # Should be BGR image
        assert len(frame.shape) == 3
        assert frame.shape[2] == 3
        # Should have some resolution
        assert frame.shape[0] > 0
        assert frame.shape[1] > 0

    def test_test_camera_disconnect(self):
        """Test disconnecting TEST camera."""
        config = CameraSettings(id="test", name="Test", type=CameraType.TEST, source="test")
        camera = Camera(config)
        camera.connect()

        camera.disconnect()

        assert camera.connected is False

    def test_test_camera_is_connected(self):
        """Test checking if camera is connected."""
        config = CameraSettings(id="test", name="Test", type=CameraType.TEST, source="test")
        camera = Camera(config)

        assert camera.connected is False

        camera.connect()
        assert camera.connected is True

        camera.disconnect()
        assert camera.connected is False

    def test_create_test_frame_basic(self):
        """Test creating test frame."""
        config = CameraSettings(id="test", name="Test", type=CameraType.TEST, source="test")
        camera = Camera(config)

        frame = camera._create_test_frame()

        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (1080, 1920, 3)  # Default resolution
        assert frame.dtype == np.uint8


class TestCameraManager:
    """Tests for CameraManager class."""

    def test_camera_manager_creation(self):
        """Test creating CameraManager instance."""
        manager = CameraManager(capture_timeout_ms=5000)

        assert manager.capture_timeout_ms == 5000
        assert len(manager.cameras) == 0

    def test_connect_test_camera(self):
        """Test connecting a TEST camera through manager."""
        manager = CameraManager()

        result = manager.connect_camera(camera_id="test1", camera_type="test", source="test")

        assert result is True
        assert "test1" in manager.cameras
        assert manager.cameras["test1"].connected is True

    def test_connect_multiple_cameras(self):
        """Test connecting multiple cameras."""
        manager = CameraManager()

        manager.connect_camera("cam1", "test", "test1")
        manager.connect_camera("cam2", "test", "test2")

        assert len(manager.cameras) == 2
        assert "cam1" in manager.cameras
        assert "cam2" in manager.cameras

    def test_disconnect_camera(self):
        """Test disconnecting a camera."""
        manager = CameraManager()
        manager.connect_camera("test1", "test", "test")

        result = manager.disconnect_camera("test1")

        assert result is True
        assert "test1" not in manager.cameras

    def test_disconnect_nonexistent_camera(self):
        """Test disconnecting non-existent camera."""
        manager = CameraManager()

        result = manager.disconnect_camera("nonexistent")

        assert result is False

    def test_capture_from_camera(self):
        """Test capturing image from camera."""
        manager = CameraManager()
        manager.connect_camera("test1", "test", "test")

        frame = manager.capture("test1")

        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert len(frame.shape) == 3

    def test_capture_from_nonexistent_camera(self):
        """Test capturing from non-existent camera."""
        manager = CameraManager()

        frame = manager.capture("nonexistent")

        assert frame is None

    def test_get_preview(self):
        """Test getting preview from camera."""
        manager = CameraManager()
        manager.connect_camera("test1", "test", "test")

        preview = manager.get_preview("test1")

        assert preview is not None
        assert isinstance(preview, np.ndarray)

    def test_get_preview_nonexistent_camera(self):
        """Test getting preview from non-existent camera."""
        manager = CameraManager()

        preview = manager.get_preview("nonexistent")

        assert preview is None

    def test_create_test_image_basic(self):
        """Test creating test image."""
        manager = CameraManager()

        image = manager.create_test_image("Test Message")

        assert image is not None
        assert isinstance(image, np.ndarray)
        assert image.shape[2] == 3  # BGR
        assert image.dtype == np.uint8

    def test_create_test_image_default_message(self):
        """Test creating test image with default message."""
        manager = CameraManager()

        image = manager.create_test_image()

        assert image is not None
        assert isinstance(image, np.ndarray)

    def test_create_test_image_with_aruco(self):
        """Test that test image contains ArUco markers."""
        manager = CameraManager()

        image = manager.create_test_image("Test")

        # Image should not be empty (contains markers)
        assert image.mean() > 0
        # Should have some variation (not all same color)
        assert image.std() > 0

    def test_list_available_cameras_empty(self):
        """Test listing cameras when none connected."""
        manager = CameraManager()

        cameras = manager.list_available_cameras()

        # Should return list (may include USB cameras if available)
        assert isinstance(cameras, list)

    def test_list_available_cameras_with_connected(self):
        """Test listing cameras with connected camera."""
        manager = CameraManager()
        manager.connect_camera("test1", "test", "test")

        cameras = manager.list_available_cameras()

        # Should include the connected test camera
        test_cameras = [c for c in cameras if c["id"] == "test1"]
        assert len(test_cameras) == 1
        assert test_cameras[0]["connected"] is True
        assert test_cameras[0]["type"] == "test"

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test async cleanup of camera manager."""
        manager = CameraManager()
        manager.connect_camera("test1", "test", "test")
        manager.connect_camera("test2", "test", "test")

        await manager.cleanup()

        # All cameras should be disconnected
        assert len(manager.cameras) == 0

    def test_camera_with_custom_resolution(self):
        """Test camera with custom resolution."""
        manager = CameraManager()

        manager.connect_camera("test1", "test", "test", resolution=(1280, 720))

        # Camera should be connected
        assert "test1" in manager.cameras
        assert manager.cameras["test1"].config.resolution == (1280, 720)

    def test_reconnect_camera(self):
        """Test reconnecting a camera with same ID."""
        manager = CameraManager()

        manager.connect_camera("test1", "test", "test")
        manager.disconnect_camera("test1")
        result = manager.connect_camera("test1", "test", "test")

        assert result is True
        assert "test1" in manager.cameras

    def test_camera_type_enum_conversion(self):
        """Test that camera type strings are properly converted."""
        manager = CameraManager()

        # Should accept string
        manager.connect_camera("test1", "test", "test")

        camera = manager.cameras["test1"]
        assert camera.config.type == CameraType.TEST

    def test_capture_timeout_configuration(self):
        """Test that capture timeout is configurable."""
        manager = CameraManager(capture_timeout_ms=10000)

        assert manager.capture_timeout_ms == 10000

    def test_multiple_captures_from_same_camera(self):
        """Test multiple captures from same camera."""
        manager = CameraManager()
        manager.connect_camera("test1", "test", "test")

        frame1 = manager.capture("test1")
        frame2 = manager.capture("test1")

        assert frame1 is not None
        assert frame2 is not None
        # Both should be valid images
        assert frame1.shape == frame2.shape
