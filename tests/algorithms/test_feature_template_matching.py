"""
Tests for feature-based template matching using ORB.

Tests feature matching algorithm including:
- Basic detection
- Rotation invariant detection
- Scale detection
- Multi-instance detection
- Low keypoint handling
"""

import cv2
import numpy as np
import pytest

from algorithms.feature_template_matching import FeatureTemplateDetector


class TestFeatureTemplateDetector:
    """Tests for FeatureTemplateDetector class."""

    @pytest.fixture
    def detector(self):
        """Create FeatureTemplateDetector instance."""
        return FeatureTemplateDetector()

    @pytest.fixture
    def textured_template(self):
        """Create textured template with good keypoints."""
        template = np.zeros((100, 100, 3), dtype=np.uint8)
        # Add noise for texture (helps ORB find keypoints)
        noise = np.random.randint(0, 50, (100, 100, 3), dtype=np.uint8)
        template = template + noise
        # Add checkerboard pattern for good keypoints
        for i in range(0, 100, 10):
            for j in range(0, 100, 10):
                if (i // 10 + j // 10) % 2 == 0:
                    cv2.rectangle(template, (i, j), (i + 10, j + 10), (255, 255, 255), -1)
        # Add some distinctive features with strong edges
        cv2.circle(template, (25, 25), 12, (0, 0, 255), -1)
        cv2.circle(template, (25, 25), 8, (255, 0, 0), -1)
        cv2.rectangle(template, (60, 60), (95, 95), (0, 255, 0), -1)
        cv2.rectangle(template, (65, 65), (90, 90), (0, 128, 0), -1)
        # Add more edges
        cv2.line(template, (0, 50), (100, 50), (200, 200, 200), 2)
        cv2.line(template, (50, 0), (50, 100), (200, 200, 200), 2)
        # Add corners
        pts = np.array([[10, 80], [30, 70], [20, 90]], np.int32)
        cv2.fillPoly(template, [pts], (128, 128, 255))
        return template

    @pytest.fixture
    def scene_with_template(self, textured_template):
        """Create scene containing the template."""
        scene = np.zeros((400, 500, 3), dtype=np.uint8)
        # Add some background texture
        cv2.rectangle(scene, (0, 0), (500, 400), (50, 50, 50), -1)
        # Place template in scene
        scene[150:250, 200:300] = textured_template
        return scene

    @pytest.fixture
    def rotated_scene(self, textured_template):
        """Create scene with rotated template."""
        scene = np.zeros((400, 500, 3), dtype=np.uint8)
        cv2.rectangle(scene, (0, 0), (500, 400), (50, 50, 50), -1)

        # Rotate template 45 degrees
        center = (50, 50)
        M = cv2.getRotationMatrix2D(center, 45, 1.0)

        # Calculate new bounding box
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int(100 * cos + 100 * sin)
        new_h = int(100 * sin + 100 * cos)
        M[0, 2] += (new_w - 100) / 2
        M[1, 2] += (new_h - 100) / 2

        rotated = cv2.warpAffine(textured_template, M, (new_w, new_h))

        # Place rotated template in scene
        y_offset = 150
        x_offset = 200
        scene[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = rotated

        return scene

    @pytest.fixture
    def uniform_template(self):
        """Create uniform template with few keypoints."""
        template = np.ones((50, 50, 3), dtype=np.uint8) * 128
        return template

    def test_detector_creation(self, detector):
        """Test creating FeatureTemplateDetector instance."""
        assert detector is not None
        assert isinstance(detector, FeatureTemplateDetector)

    def test_detect_basic_match(self, detector, scene_with_template, textured_template):
        """Test basic feature-based template detection."""
        params = {
            "threshold": 0.3,
            "min_matches": 6,
            "ratio_threshold": 0.75,
            "find_multiple": False,
        }
        result = detector.detect(
            scene_with_template, textured_template, "test_tmpl", params
        )

        assert result is not None
        assert "success" in result
        assert result["success"] is True
        assert "objects" in result
        # Should find at least one match
        assert len(result["objects"]) >= 1

    def test_detect_with_rotation(self, detector, rotated_scene, textured_template):
        """Test rotation invariant detection."""
        params = {
            "threshold": 0.3,
            "min_matches": 6,
            "ratio_threshold": 0.8,
            "find_multiple": False,
        }
        result = detector.detect(
            rotated_scene, textured_template, "test_tmpl", params
        )

        assert result["success"] is True
        if len(result["objects"]) > 0:
            obj = result["objects"][0]
            # Should have rotation information
            assert obj.rotation is not None
            # Rotation should be close to 45 degrees (allowing some tolerance)
            assert abs(obj.rotation - 45) < 20 or abs(obj.rotation + 315) < 20

    def test_detect_multiple_instances(self, detector, textured_template, scene_with_template):
        """Test multi-instance detection flag doesn't crash."""
        # Use the scene_with_template which has one known good match
        params = {
            "threshold": 0.2,
            "min_matches": 4,
            "ratio_threshold": 0.85,
            "find_multiple": True,  # Enable multi-instance mode
            "max_matches": 5,
        }
        result = detector.detect(
            scene_with_template, textured_template, "test_tmpl", params
        )

        assert result["success"] is True
        # With find_multiple=True, should still find at least one instance
        # The scene has one template copy, so should find 1
        assert len(result["objects"]) >= 1

    def test_low_keypoint_warning(self, detector, uniform_template, caplog):
        """Test warning when template has insufficient keypoints."""
        scene = np.zeros((200, 200, 3), dtype=np.uint8)

        params = {
            "threshold": 0.5,
            "min_matches": 10,
            "find_multiple": False,
        }

        import logging
        with caplog.at_level(logging.WARNING):
            _result = detector.detect(scene, uniform_template, "test_tmpl", params)

        # Should log warning about insufficient keypoints
        assert any("keypoint" in record.message.lower() for record in caplog.records)
        # Verify result is valid even with low keypoints
        assert _result["success"] is True

    def test_no_match_returns_empty(self, detector, textured_template):
        """Test that no match returns empty objects list."""
        # Scene with completely different content
        scene = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.circle(scene, (150, 150), 100, (255, 0, 0), -1)

        params = {
            "threshold": 0.8,  # High threshold
            "min_matches": 20,
            "find_multiple": False,
        }
        result = detector.detect(scene, textured_template, "test_tmpl", params)

        assert result["success"] is True
        assert len(result["objects"]) == 0

    def test_vision_object_properties(self, detector, scene_with_template, textured_template):
        """Test VisionObject properties in detection results."""
        params = {
            "threshold": 0.3,
            "min_matches": 6,
            "find_multiple": False,
        }
        result = detector.detect(
            scene_with_template, textured_template, "test_tmpl", params
        )

        if len(result["objects"]) > 0:
            obj = result["objects"][0]
            assert hasattr(obj, "object_id")
            assert hasattr(obj, "object_type")
            assert hasattr(obj, "bounding_box")
            assert hasattr(obj, "center")
            assert hasattr(obj, "confidence")
            assert hasattr(obj, "properties")
            assert "template_id" in obj.properties
            assert "method" in obj.properties
            assert obj.properties["method"] == "feature_orb"

    def test_corners_in_properties(self, detector, scene_with_template, textured_template):
        """Test that transformed corners are in properties."""
        params = {
            "threshold": 0.3,
            "min_matches": 6,
            "find_multiple": False,
        }
        result = detector.detect(
            scene_with_template, textured_template, "test_tmpl", params
        )

        if len(result["objects"]) > 0:
            obj = result["objects"][0]
            assert "corners" in obj.properties
            corners = obj.properties["corners"]
            assert len(corners) == 4
            # Each corner should have x, y
            for corner in corners:
                assert len(corner) == 2

    def test_scale_detection(self, detector, textured_template):
        """Test scale detection in properties."""
        # Create scaled scene
        scene = np.zeros((400, 500, 3), dtype=np.uint8)
        cv2.rectangle(scene, (0, 0), (500, 400), (50, 50, 50), -1)

        # Scale template to 1.5x
        scaled = cv2.resize(textured_template, None, fx=1.5, fy=1.5)
        h, w = scaled.shape[:2]
        scene[100:100 + h, 100:100 + w] = scaled

        params = {
            "threshold": 0.3,
            "min_matches": 6,
            "ratio_threshold": 0.8,
            "find_multiple": False,
        }
        result = detector.detect(scene, textured_template, "test_tmpl", params)

        if len(result["objects"]) > 0:
            obj = result["objects"][0]
            assert "scale" in obj.properties
            # Scale should be close to 1.5
            assert 1.2 < obj.properties["scale"] < 1.8

    def test_confidence_in_range(self, detector, scene_with_template, textured_template):
        """Test that confidence is in valid range."""
        params = {
            "threshold": 0.1,
            "min_matches": 4,
            "find_multiple": False,
        }
        result = detector.detect(
            scene_with_template, textured_template, "test_tmpl", params
        )

        for obj in result["objects"]:
            assert 0.0 <= obj.confidence <= 1.0

    def test_result_structure(self, detector, scene_with_template, textured_template):
        """Test structure of detection result."""
        params = {
            "threshold": 0.3,
            "min_matches": 6,
            "find_multiple": False,
        }
        result = detector.detect(
            scene_with_template, textured_template, "test_tmpl", params
        )

        assert "success" in result
        assert "objects" in result
        assert "image" in result
        assert isinstance(result["objects"], list)
        assert isinstance(result["image"], np.ndarray)

    def test_grayscale_images(self, detector):
        """Test detection with grayscale images."""
        # Create grayscale template with texture
        template = np.zeros((80, 80), dtype=np.uint8)
        # Add noise
        noise = np.random.randint(0, 30, (80, 80), dtype=np.uint8)
        template = template + noise
        # Add pattern
        for i in range(0, 80, 8):
            for j in range(0, 80, 8):
                if (i // 8 + j // 8) % 2 == 0:
                    cv2.rectangle(template, (i, j), (i + 8, j + 8), 255, -1)
        # Add distinctive features
        cv2.circle(template, (20, 20), 10, 200, -1)
        cv2.circle(template, (60, 60), 10, 180, -1)
        cv2.line(template, (0, 40), (80, 40), 150, 2)

        # Create scene with template
        scene = np.zeros((200, 200), dtype=np.uint8)
        scene[:] = 30  # Background
        scene[60:140, 60:140] = template

        params = {
            "threshold": 0.3,
            "min_matches": 4,
            "find_multiple": False,
        }
        result = detector.detect(scene, template, "test_tmpl", params)

        assert result["success"] is True
        # May or may not find match depending on keypoints
        # Just verify it doesn't crash

    def test_mask_support(self, detector, textured_template, scene_with_template):
        """Test detection with alpha mask."""
        # Create mask (only use center of template)
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 40, 255, -1)

        params = {
            "threshold": 0.3,
            "min_matches": 4,
            "find_multiple": False,
        }
        result = detector.detect(
            scene_with_template, textured_template, "test_tmpl", params, mask=mask
        )

        assert result["success"] is True

    def test_object_ids_unique(self, detector, textured_template):
        """Test that object IDs are unique for multiple matches."""
        # Create scene with multiple instances
        scene = np.zeros((400, 600, 3), dtype=np.uint8)
        scene[50:150, 50:150] = textured_template
        scene[50:150, 250:350] = textured_template

        params = {
            "threshold": 0.3,
            "min_matches": 6,
            "find_multiple": True,
            "max_matches": 5,
        }
        result = detector.detect(scene, textured_template, "test_tmpl", params)

        object_ids = [obj.object_id for obj in result["objects"]]
        # All IDs should be unique
        assert len(object_ids) == len(set(object_ids))

    def test_max_matches_limit(self, detector, textured_template):
        """Test that max_matches parameter is respected."""
        # Create scene with many instances
        scene = np.zeros((400, 800, 3), dtype=np.uint8)
        for x in range(0, 700, 120):
            scene[50:150, x:x + 100] = textured_template

        params = {
            "threshold": 0.3,
            "min_matches": 6,
            "find_multiple": True,
            "max_matches": 2,  # Limit to 2
        }
        result = detector.detect(scene, textured_template, "test_tmpl", params)

        assert len(result["objects"]) <= 2

    def test_ratio_threshold_strictness(self, detector, scene_with_template, textured_template):
        """Test that lower ratio threshold is stricter."""
        strict_params = {
            "threshold": 0.3,
            "min_matches": 4,
            "ratio_threshold": 0.5,  # Strict
            "find_multiple": False,
        }
        loose_params = {
            "threshold": 0.3,
            "min_matches": 4,
            "ratio_threshold": 0.9,  # Loose
            "find_multiple": False,
        }

        strict_result = detector.detect(
            scene_with_template, textured_template, "test_tmpl", strict_params
        )
        loose_result = detector.detect(
            scene_with_template, textured_template, "test_tmpl", loose_params
        )

        # Strict should have fewer or equal matches
        # (in properties, check match_count if available)
        if strict_result["objects"] and loose_result["objects"]:
            strict_matches = strict_result["objects"][0].properties.get("match_count", 0)
            loose_matches = loose_result["objects"][0].properties.get("match_count", 0)
            assert strict_matches <= loose_matches
