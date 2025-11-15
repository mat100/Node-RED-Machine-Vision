"""
Pytest configuration for API integration tests
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="function")
def client():
    """
    Create a test client with properly initialized app state.
    Each test gets a fresh client to avoid state contamination.
    """
    import shutil
    import tempfile

    from core.camera_manager import CameraManager
    from core.image_manager import ImageManager
    from core.template_manager import TemplateManager
    from main import app
    from services.camera_service import CameraService
    from services.image_service import ImageService
    from services.vision_service import VisionService

    # Create temporary template directory
    temp_dir = tempfile.mkdtemp()

    # Initialize managers (lightweight for testing)
    image_manager = ImageManager(max_size_mb=100, max_images=10)
    camera_manager = CameraManager()
    template_manager = TemplateManager(temp_dir)

    # Initialize services as singletons (matching production behavior)
    vision_service = VisionService(
        image_manager=image_manager,
        template_manager=template_manager,
    )
    camera_service = CameraService(
        camera_manager=camera_manager,
        image_manager=image_manager,
    )
    image_service = ImageService(image_manager=image_manager)

    # Create test config
    test_config = {
        "debug": {
            "save_debug_images": False,
            "show_overlays": False,
        }
    }

    # Set in app state (managers + services)
    app.state.image_manager = image_manager
    app.state.camera_manager = camera_manager
    app.state.template_manager = template_manager
    app.state.vision_service = vision_service
    app.state.camera_service = camera_service
    app.state.image_service = image_service
    app.state.config = test_config
    app.state.active_streams = {}

    # Create test client (no context manager to avoid blocking)
    test_client = TestClient(app, raise_server_exceptions=False)

    yield test_client

    # Quick cleanup (don't wait for async)
    try:
        image_manager.cleanup()
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:  # noqa: E722
        pass
