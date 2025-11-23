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

    from managers.camera_manager import CameraManager
    from managers.image_manager import ImageManager
    from managers.template_manager import TemplateManager
    from managers.test_image_manager import TestImageManager
    from main import app

    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    test_image_dir = tempfile.mkdtemp()

    # Initialize managers (lightweight for testing)
    image_manager = ImageManager(max_size_mb=100, max_images=10)
    camera_manager = CameraManager()
    template_manager = TemplateManager(temp_dir)
    test_image_manager = TestImageManager(test_image_dir)

    # Create test config
    test_config = {
        "debug": {
            "save_debug_images": False,
            "show_overlays": False,
        }
    }

    # Set in app state (managers only - no services after refactoring)
    app.state.image_manager = image_manager
    app.state.camera_manager = camera_manager
    app.state.template_manager = template_manager
    app.state.test_image_manager = test_image_manager
    app.state.config = test_config
    app.state.active_streams = {}

    # Create test client (no context manager to avoid blocking)
    test_client = TestClient(app, raise_server_exceptions=False)

    yield test_client

    # Quick cleanup (don't wait for async)
    try:
        image_manager.cleanup()
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(test_image_dir, ignore_errors=True)
    except Exception:  # noqa: E722
        pass
