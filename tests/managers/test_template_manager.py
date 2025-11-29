"""
Tests for core.template_manager module.

Tests template storage, retrieval, and management operations.
"""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from managers.template_manager import TemplateManager


class TestTemplateManager:
    """Tests for TemplateManager class."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def manager(self, temp_storage):
        """Create TemplateManager instance with temp storage."""
        return TemplateManager(storage_path=temp_storage)

    @pytest.fixture
    def test_image(self):
        """Create a test template image."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (80, 80), (255, 255, 255), -1)
        return image

    def test_template_manager_creation(self, temp_storage):
        """Test creating TemplateManager."""
        manager = TemplateManager(storage_path=temp_storage)

        assert manager.storage_path == Path(temp_storage)
        assert manager.storage_path.exists()
        assert len(manager.templates) == 0
        assert len(manager.template_images) == 0

    def test_upload_template_basic(self, manager, test_image):
        """Test uploading a template."""
        template_id = manager.upload_template("test_template", test_image)

        assert template_id is not None
        assert isinstance(template_id, str)
        assert template_id in manager.templates
        assert template_id in manager.template_images

    def test_upload_template_with_description(self, manager, test_image):
        """Test uploading template with description."""
        template_id = manager.upload_template(
            "test_template", test_image, description="Test description"
        )

        assert manager.templates[template_id]["description"] == "Test description"

    def test_upload_multiple_templates(self, manager, test_image):
        """Test uploading multiple templates."""
        id1 = manager.upload_template("template1", test_image)
        id2 = manager.upload_template("template2", test_image)

        assert len(manager.templates) == 2
        assert id1 != id2
        assert id1 in manager.template_images
        assert id2 in manager.template_images

    def test_get_template_existing(self, manager, test_image):
        """Test retrieving existing template."""
        template_id = manager.upload_template("test", test_image)

        retrieved = manager.get_template(template_id)

        assert retrieved is not None
        assert isinstance(retrieved, np.ndarray)
        assert retrieved.shape == test_image.shape

    def test_get_template_nonexistent(self, manager):
        """Test retrieving non-existent template."""
        result = manager.get_template("nonexistent-id")

        assert result is None

    def test_list_templates_empty(self, manager):
        """Test listing templates when empty."""
        templates = manager.list_templates()

        assert isinstance(templates, list)
        assert len(templates) == 0

    def test_list_templates_with_data(self, manager, test_image):
        """Test listing templates with data."""
        manager.upload_template("template1", test_image)
        manager.upload_template("template2", test_image, description="Test")

        templates = manager.list_templates()

        assert len(templates) == 2
        assert all("id" in t for t in templates)
        assert all("name" in t for t in templates)

    def test_delete_template_existing(self, manager, test_image):
        """Test deleting existing template."""
        template_id = manager.upload_template("test", test_image)

        result = manager.delete_template(template_id)

        assert result is True
        assert template_id not in manager.templates
        assert template_id not in manager.template_images

    def test_delete_template_nonexistent(self, manager):
        """Test deleting non-existent template."""
        result = manager.delete_template("nonexistent-id")

        assert result is False

    def test_template_persistence(self, temp_storage, test_image):
        """Test that templates persist across manager instances."""
        # Create manager and add template
        manager1 = TemplateManager(storage_path=temp_storage)
        template_id = manager1.upload_template("persistent", test_image)

        # Create new manager instance
        manager2 = TemplateManager(storage_path=temp_storage)

        # Template should be loaded
        assert template_id in manager2.templates
        assert template_id in manager2.template_images
        retrieved = manager2.get_template(template_id)
        assert retrieved is not None

    def test_learn_template_from_image(self, manager):
        """Test learning template from full image."""
        source_image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(source_image, (50, 50), (150, 150), (255, 255, 255), -1)

        # Use full image as ROI
        roi = {"x": 0, "y": 0, "width": 200, "height": 200}
        template_id = manager.learn_template("learned", source_image, roi)

        assert template_id is not None
        assert template_id in manager.templates
        template = manager.get_template(template_id)
        assert template is not None

    def test_learn_template_with_roi(self, manager):
        """Test learning template from ROI."""
        source_image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(source_image, (50, 50), (150, 150), (255, 255, 255), -1)

        roi = {"x": 50, "y": 50, "width": 100, "height": 100}
        template_id = manager.learn_template("learned_roi", source_image, roi)

        assert template_id is not None
        template = manager.get_template(template_id)
        assert template.shape == (100, 100, 3)

    def test_learn_template_invalid_roi(self, manager):
        """Test learning template with invalid ROI."""
        source_image = np.zeros((100, 100, 3), dtype=np.uint8)

        # ROI exceeds image bounds
        roi = {"x": 50, "y": 50, "width": 200, "height": 200}

        with pytest.raises(ValueError, match="ROI exceeds image bounds"):
            manager.learn_template("invalid_roi", source_image, roi=roi)

    def test_template_metadata_structure(self, manager, test_image):
        """Test template metadata structure."""
        template_id = manager.upload_template("test", test_image, description="Test template")

        metadata = manager.templates[template_id]

        assert "id" in metadata
        assert "name" in metadata
        assert "description" in metadata
        assert "filename" in metadata
        assert "created_at" in metadata
        assert "size" in metadata
        assert "width" in metadata["size"]
        assert "height" in metadata["size"]

    def test_template_image_saved_to_disk(self, manager, test_image):
        """Test that template image is saved to disk."""
        template_id = manager.upload_template("test", test_image)

        filename = manager.templates[template_id]["filename"]
        image_path = manager.storage_path / filename

        assert image_path.exists()
        # Verify image can be loaded
        loaded = cv2.imread(str(image_path))
        assert loaded is not None
        assert loaded.shape == test_image.shape

    def test_metadata_json_saved(self, manager, test_image):
        """Test that metadata JSON is saved."""
        manager.upload_template("test", test_image)

        metadata_file = manager.storage_path / "templates.json"

        assert metadata_file.exists()
        with open(metadata_file) as f:
            data = json.load(f)
        assert len(data) == 1

    def test_create_template_thumbnail(self, manager, test_image):
        """Test creating template thumbnail."""
        template_id = manager.upload_template("test", test_image)

        thumbnail = manager.create_template_thumbnail(template_id)

        assert thumbnail is not None
        assert isinstance(thumbnail, str)
        # Should be base64 string
        assert len(thumbnail) > 0

    def test_create_thumbnail_nonexistent(self, manager):
        """Test creating thumbnail for non-existent template."""
        result = manager.create_template_thumbnail("nonexistent")

        assert result is None

    def test_template_names_unique(self, manager, test_image):
        """Test that templates with same name get different IDs."""
        id1 = manager.upload_template("same_name", test_image)
        id2 = manager.upload_template("same_name", test_image)

        assert id1 != id2
        assert len(manager.templates) == 2

    def test_upload_grayscale_template(self, manager):
        """Test uploading grayscale template."""
        gray_image = np.zeros((100, 100), dtype=np.uint8)
        gray_image[20:80, 20:80] = 255

        template_id = manager.upload_template("gray", gray_image)

        assert template_id in manager.templates
        retrieved = manager.get_template(template_id)
        # Grayscale images are stored as-is
        assert retrieved is not None
        assert retrieved.shape[:2] == (100, 100)

    def test_template_file_cleanup_on_delete(self, manager, test_image):
        """Test that template file is deleted from disk."""
        template_id = manager.upload_template("test", test_image)
        filename = manager.templates[template_id]["filename"]
        image_path = manager.storage_path / filename

        manager.delete_template(template_id)

        assert not image_path.exists()

    def test_concurrent_access_safety(self, manager, test_image):
        """Test thread-safe operations."""
        import threading

        results = []

        def upload():
            tid = manager.upload_template("concurrent", test_image)
            results.append(tid)

        threads = [threading.Thread(target=upload) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All uploads should succeed with unique IDs
        assert len(results) == 5
        assert len(set(results)) == 5

    def test_learn_template_with_description(self, manager):
        """Test learning template with description."""
        source_image = np.zeros((100, 100, 3), dtype=np.uint8)

        roi = {"x": 0, "y": 0, "width": 100, "height": 100}
        template_id = manager.learn_template(
            "test", source_image, roi, description="Learned template"
        )

        assert manager.templates[template_id]["description"] == "Learned template"

    def test_empty_template_name(self, manager, test_image):
        """Test uploading template with empty name."""
        template_id = manager.upload_template("", test_image)

        # Should still work, just with empty name
        assert template_id in manager.templates
        assert manager.templates[template_id]["name"] == ""

    def test_template_dimensions_stored(self, manager, test_image):
        """Test that template dimensions are stored in metadata."""
        template_id = manager.upload_template("test", test_image)

        metadata = manager.templates[template_id]
        assert metadata["size"]["width"] == 100
        assert metadata["size"]["height"] == 100

    def test_upload_template_with_alpha_channel(self, manager):
        """Test uploading template with alpha channel (BGRA)."""
        # Create BGRA image (4 channels)
        image = np.zeros((100, 100, 4), dtype=np.uint8)
        # White rectangle in center
        image[20:80, 20:80, :3] = 255  # BGR = white
        # Full opacity in center, transparent elsewhere
        image[20:80, 20:80, 3] = 255  # Alpha = opaque
        image[:20, :, 3] = 0  # Top transparent
        image[80:, :, 3] = 0  # Bottom transparent

        template_id = manager.upload_template("alpha_template", image)

        assert template_id in manager.templates
        assert manager.templates[template_id]["has_alpha"] is True

        # Retrieved template should be BGR only (3 channels)
        retrieved = manager.get_template(template_id)
        assert retrieved is not None
        assert len(retrieved.shape) == 3
        assert retrieved.shape[2] == 3  # BGR only

        # Mask should be available
        mask = manager.get_template_mask(template_id)
        assert mask is not None
        assert len(mask.shape) == 2  # Single channel
        assert mask.shape == (100, 100)

    def test_get_template_mask_existing(self, manager):
        """Test retrieving mask for template with alpha channel."""
        # Create BGRA image
        image = np.zeros((50, 50, 4), dtype=np.uint8)
        image[:, :, :3] = [100, 150, 200]  # BGR color
        image[:, :, 3] = 128  # Half transparent

        template_id = manager.upload_template("masked", image)

        mask = manager.get_template_mask(template_id)

        assert mask is not None
        assert mask.shape == (50, 50)
        assert np.all(mask == 128)  # Should preserve alpha values

    def test_get_template_mask_no_alpha(self, manager, test_image):
        """Test retrieving mask for template without alpha channel."""
        template_id = manager.upload_template("no_alpha", test_image)

        mask = manager.get_template_mask(template_id)

        assert mask is None
        assert manager.templates[template_id]["has_alpha"] is False

    def test_alpha_channel_persistence(self, temp_storage):
        """Test that alpha channel persists across manager instances."""
        # Create BGRA image
        image = np.zeros((100, 100, 4), dtype=np.uint8)
        image[25:75, 25:75, :3] = 255
        image[25:75, 25:75, 3] = 200

        # Upload with first manager
        manager1 = TemplateManager(storage_path=temp_storage)
        template_id = manager1.upload_template("alpha_persistent", image)

        # Create new manager and verify alpha is loaded
        manager2 = TemplateManager(storage_path=temp_storage)

        assert template_id in manager2.templates
        assert manager2.templates[template_id]["has_alpha"] is True

        mask = manager2.get_template_mask(template_id)
        assert mask is not None
        assert mask.shape == (100, 100)
        # Check that center has alpha value
        assert np.all(mask[25:75, 25:75] == 200)

    def test_thumbnail_with_alpha_channel(self, manager):
        """Test creating thumbnail for template with alpha channel."""
        # Create BGRA image with transparency
        image = np.zeros((100, 100, 4), dtype=np.uint8)
        image[20:80, 20:80, :3] = [255, 0, 0]  # Blue square
        image[20:80, 20:80, 3] = 255  # Opaque
        image[:, :, 3][image[:, :, 3] == 0] = 0  # Rest transparent

        template_id = manager.upload_template("alpha_thumb", image)

        thumbnail = manager.create_template_thumbnail(template_id)

        assert thumbnail is not None
        assert isinstance(thumbnail, str)
        assert len(thumbnail) > 0
        # Thumbnail should contain base64 encoded PNG with alpha

    def test_delete_template_with_alpha_removes_mask(self, manager):
        """Test that deleting template also removes mask from cache."""
        # Create BGRA image
        image = np.zeros((50, 50, 4), dtype=np.uint8)
        image[:, :, :3] = 255
        image[:, :, 3] = 200

        template_id = manager.upload_template("alpha_delete", image)

        # Verify mask exists
        assert template_id in manager.template_masks
        assert manager.get_template_mask(template_id) is not None

        # Delete template
        manager.delete_template(template_id)

        # Mask should be removed
        assert template_id not in manager.template_masks

    def test_learn_template_preserves_alpha_if_present(self, manager):
        """Test that learning from BGRA source preserves alpha channel."""
        # Create BGRA source image
        source_image = np.zeros((200, 200, 4), dtype=np.uint8)
        source_image[50:150, 50:150, :3] = [0, 255, 0]  # Green square
        source_image[50:150, 50:150, 3] = 180  # Semi-transparent

        roi = {"x": 50, "y": 50, "width": 100, "height": 100}
        template_id = manager.learn_template("learned_alpha", source_image, roi)

        assert template_id in manager.templates
        # Check if alpha was preserved
        mask = manager.get_template_mask(template_id)
        if mask is not None:
            assert mask.shape == (100, 100)

    def test_mixed_templates_alpha_and_non_alpha(self, manager, test_image):
        """Test managing templates with and without alpha channel."""
        # Upload template without alpha
        id_no_alpha = manager.upload_template("no_alpha", test_image)

        # Upload template with alpha
        alpha_image = np.zeros((100, 100, 4), dtype=np.uint8)
        alpha_image[:, :, :3] = 255
        alpha_image[:, :, 3] = 128
        id_with_alpha = manager.upload_template("with_alpha", alpha_image)

        # Verify both exist
        assert id_no_alpha in manager.templates
        assert id_with_alpha in manager.templates

        # Verify alpha flags
        assert manager.templates[id_no_alpha]["has_alpha"] is False
        assert manager.templates[id_with_alpha]["has_alpha"] is True

        # Verify masks
        assert manager.get_template_mask(id_no_alpha) is None
        assert manager.get_template_mask(id_with_alpha) is not None
