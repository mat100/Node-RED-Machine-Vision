"""
Pytest configuration and fixtures for Machine Vision Flow tests
"""

from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from core.camera_manager import CameraManager
from core.image_manager import ImageManager
from core.template_manager import TemplateManager
from services.camera_service import CameraService
from services.image_service import ImageService
from services.vision_service import VisionService


@pytest.fixture
def test_image():
    """Create a test image for testing"""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add some content
    cv2.rectangle(image, (100, 100), (300, 300), (255, 255, 255), -1)
    cv2.circle(image, (450, 350), 50, (128, 128, 128), -1)
    return image


@pytest.fixture
def test_template():
    """Create a test template image"""
    template = np.zeros((50, 50, 3), dtype=np.uint8)
    cv2.rectangle(template, (10, 10), (40, 40), (255, 255, 255), -1)
    return template


@pytest.fixture
def image_manager():
    """Create ImageManager instance for testing"""
    manager = ImageManager(max_size_mb=100, max_images=10)
    yield manager
    # Cleanup
    manager.cleanup()


@pytest.fixture
def camera_manager():
    """Create CameraManager instance for testing"""
    import asyncio

    manager = CameraManager()
    yield manager
    # Cleanup - run async cleanup in event loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, create task
            asyncio.create_task(manager.cleanup())
        else:
            # If no loop or not running, run synchronously
            loop.run_until_complete(manager.cleanup())
    except RuntimeError:
        # No event loop, create new one
        asyncio.run(manager.cleanup())


@pytest.fixture
def template_manager(tmp_path):
    """Create TemplateManager instance with temporary directory"""
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    manager = TemplateManager(str(template_dir))
    yield manager
    # Cleanup happens automatically with tmp_path


@pytest.fixture
def camera_service(camera_manager, image_manager):
    """Create CameraService instance for testing"""
    return CameraService(camera_manager=camera_manager, image_manager=image_manager)


@pytest.fixture
def image_service(image_manager):
    """Create ImageService instance for testing"""
    return ImageService(image_manager=image_manager)


@pytest.fixture
def vision_service(image_manager, template_manager):
    """Create VisionService instance for testing"""
    return VisionService(
        image_manager=image_manager,
        template_manager=template_manager,
    )


@pytest.fixture
def mock_camera_manager():
    """Create mock CameraManager for unit testing"""
    mock = MagicMock()
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    mock.list_available_cameras.return_value = [
        {"id": "test", "name": "Test Camera", "type": "test"}
    ]
    mock.connect_camera.return_value = True
    mock.disconnect_camera.return_value = True
    mock.capture.return_value = test_image
    mock.get_preview.return_value = test_image
    mock.create_test_image.return_value = test_image
    return mock


@pytest.fixture
def mock_image_manager():
    """Create mock ImageManager for unit testing"""
    mock = MagicMock()
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    mock.get.return_value = test_image
    mock.store.return_value = "test-image-id"
    mock.create_thumbnail.return_value = (test_image, "base64-thumbnail")
    mock.delete.return_value = True
    mock.has_image.return_value = True
    mock.list_images.return_value = []
    mock.get_metadata.return_value = {}
    return mock


@pytest.fixture
def mock_template_manager():
    """Create mock TemplateManager for unit testing"""
    mock = MagicMock()
    test_template = np.zeros((50, 50, 3), dtype=np.uint8)
    mock.get_template.return_value = test_template
    mock.list_templates.return_value = []
    mock.learn_template.return_value = "new-template-id"
    mock.create_template_thumbnail.return_value = "base64-template-thumb"
    return mock
