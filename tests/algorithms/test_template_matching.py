"""
Tests for vision.template_matching module.

Tests template matching algorithms for pattern recognition.
"""

import cv2
import numpy as np
import pytest

from algorithms.template_matching import TemplateDetector


class TestTemplateDetector:
    """Tests for TemplateDetector class."""

    @pytest.fixture
    def detector(self):
        """Create TemplateDetector instance."""
        return TemplateDetector()

    @pytest.fixture
    def test_scene(self):
        """Create test scene image."""
        scene = np.zeros((300, 400, 3), dtype=np.uint8)
        # Add template pattern at known location
        cv2.rectangle(scene, (150, 100), (200, 150), (255, 255, 255), -1)
        return scene

    @pytest.fixture
    def test_template(self):
        """Create test template image."""
        template = np.zeros((50, 50, 3), dtype=np.uint8)
        cv2.rectangle(template, (0, 0), (50, 50), (255, 255, 255), -1)
        return template

    def test_detector_creation(self, detector):
        """Test creating TemplateDetector instance."""
        assert detector is not None
        assert isinstance(detector, TemplateDetector)

    def test_detect_basic(self, detector, test_scene, test_template):
        """Test basic template detection."""
        params = {"threshold": 0.7}
        result = detector.detect(test_scene, test_template, "test_tmpl", params)

        assert result is not None
        assert "success" in result
        assert result["success"] is True
        assert "objects" in result

    def test_detect_with_threshold(self, detector, test_scene, test_template):
        """Test detection with custom threshold."""
        params = {"threshold": 0.5}
        result = detector.detect(test_scene, test_template, "test", params)

        assert result["success"] is True

    def test_detect_with_method(self, detector, test_scene, test_template):
        """Test detection with specific method."""
        params = {"method": "TM_CCOEFF_NORMED", "threshold": 0.7}
        result = detector.detect(test_scene, test_template, "test", params)

        assert result["success"] is True

    def test_detect_result_structure(self, detector, test_scene, test_template):
        """Test structure of detection result."""
        params = {"threshold": 0.7}
        result = detector.detect(test_scene, test_template, "test", params)

        assert "success" in result
        assert "objects" in result
        assert "image" in result
        assert isinstance(result["objects"], list)
        assert isinstance(result["image"], np.ndarray)

    def test_detect_grayscale_images(self, detector):
        """Test detection with grayscale images."""
        scene = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(scene, (50, 50), (100, 100), 255, -1)
        template = np.ones((50, 50), dtype=np.uint8) * 255

        params = {"threshold": 0.7}
        result = detector.detect(scene, template, "test", params)

        assert result["success"] is True

    def test_detect_template_larger_than_scene(self, detector):
        """Test detection when template is larger than scene."""
        scene = np.zeros((50, 50, 3), dtype=np.uint8)
        template = np.zeros((100, 100, 3), dtype=np.uint8)

        params = {"threshold": 0.7}
        result = detector.detect(scene, template, "test", params)

        # Should handle gracefully
        assert result["success"] is True

    def test_detect_identical_images(self, detector, test_template):
        """Test detection with identical scene and template."""
        params = {"threshold": 0.9}
        result = detector.detect(test_template, test_template, "test", params)

        # Should find exact match
        assert result["success"] is True
        assert len(result["objects"]) >= 1

    def test_detect_completely_different_images(self, detector):
        """Test detection with completely different images."""
        scene = np.zeros((200, 200, 3), dtype=np.uint8)
        template = np.ones((50, 50, 3), dtype=np.uint8) * 255

        params = {"threshold": 0.9}
        result = detector.detect(scene, template, "test", params)

        # Should find no matches
        assert result["success"] is True

    def test_detect_vision_object_properties(self, detector, test_scene, test_template):
        """Test VisionObject properties in detection results."""
        params = {"threshold": 0.7}
        result = detector.detect(test_scene, test_template, "test", params)

        if len(result["objects"]) > 0:
            obj = result["objects"][0]
            assert hasattr(obj, "object_id")
            assert hasattr(obj, "object_type")
            assert hasattr(obj, "bounding_box")
            assert hasattr(obj, "center")
            assert hasattr(obj, "confidence")

    def test_detect_different_methods(self, detector, test_scene, test_template):
        """Test different template matching methods."""
        methods = [
            "TM_CCOEFF_NORMED",
            "TM_CCORR_NORMED",
            "TM_SQDIFF_NORMED",
        ]

        for method in methods:
            params = {"method": method, "threshold": 0.5}
            result = detector.detect(test_scene, test_template, "test", params)
            assert result["success"] is True, f"Method {method} failed"

    def test_detect_empty_scene(self, detector, test_template):
        """Test detection on empty scene."""
        empty_scene = np.zeros((200, 200, 3), dtype=np.uint8)

        params = {"threshold": 0.9}
        result = detector.detect(empty_scene, test_template, "test", params)

        assert result["success"] is True

    def test_detect_with_high_threshold(self, detector, test_scene, test_template):
        """Test detection with very high threshold."""
        params = {"threshold": 0.99}
        result = detector.detect(test_scene, test_template, "test", params)

        # Should succeed even if no matches found
        assert result["success"] is True
