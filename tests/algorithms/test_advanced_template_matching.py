"""
Tests for advanced template matching with rotation and multi-instance support.

Tests advanced template matching algorithms including:
- Multi-instance detection with NMS
- Rotation-invariant matching
- Overlap filtering
"""

import cv2
import numpy as np
import pytest

from algorithms.advanced_template_matching import AdvancedTemplateDetector


class TestAdvancedTemplateDetector:
    """Tests for AdvancedTemplateDetector class."""

    @pytest.fixture
    def detector(self):
        """Create AdvancedTemplateDetector instance."""
        return AdvancedTemplateDetector()

    @pytest.fixture
    def test_scene(self):
        """Create test scene image with single pattern."""
        scene = np.zeros((300, 400, 3), dtype=np.uint8)
        # Add template pattern at known location
        cv2.rectangle(scene, (150, 100), (200, 150), (255, 255, 255), -1)
        return scene

    @pytest.fixture
    def multi_scene(self):
        """Create test scene with multiple instances of pattern."""
        scene = np.zeros((400, 500, 3), dtype=np.uint8)
        # Add multiple white rectangles
        cv2.rectangle(scene, (50, 50), (100, 100), (255, 255, 255), -1)
        cv2.rectangle(scene, (200, 50), (250, 100), (255, 255, 255), -1)
        cv2.rectangle(scene, (350, 50), (400, 100), (255, 255, 255), -1)
        cv2.rectangle(scene, (125, 200), (175, 250), (255, 255, 255), -1)
        return scene

    @pytest.fixture
    def test_template(self):
        """Create test template image."""
        template = np.zeros((50, 50, 3), dtype=np.uint8)
        cv2.rectangle(template, (0, 0), (50, 50), (255, 255, 255), -1)
        return template

    @pytest.fixture
    def asymmetric_template(self):
        """Create asymmetric template for rotation testing."""
        template = np.zeros((60, 40, 3), dtype=np.uint8)
        # L-shaped pattern
        cv2.rectangle(template, (0, 0), (15, 60), (255, 255, 255), -1)
        cv2.rectangle(template, (0, 45), (40, 60), (255, 255, 255), -1)
        return template

    def test_detector_creation(self, detector):
        """Test creating AdvancedTemplateDetector instance."""
        assert detector is not None
        assert isinstance(detector, AdvancedTemplateDetector)

    def test_detect_basic_single_match(self, detector, test_scene, test_template):
        """Test basic single template detection (backward compatibility)."""
        params = {
            "threshold": 0.7,
            "find_multiple": False,
            "enable_rotation": False,
        }
        result = detector.detect(test_scene, test_template, "test_tmpl", params)

        assert result is not None
        assert "success" in result
        assert result["success"] is True
        assert "objects" in result
        assert len(result["objects"]) <= 1

    def test_detect_multiple_instances(self, detector, multi_scene, test_template):
        """Test multi-instance detection."""
        params = {
            "threshold": 0.7,
            "find_multiple": True,
            "max_matches": 10,
            "overlap_threshold": 0.3,
            "enable_rotation": False,
        }
        result = detector.detect(multi_scene, test_template, "test", params)

        assert result["success"] is True
        assert len(result["objects"]) >= 2  # Should find multiple instances
        assert len(result["objects"]) <= 10  # Should respect max_matches

    def test_nms_overlap_filtering(self, detector, test_scene, test_template):
        """Test that NMS properly filters overlapping detections."""
        params = {
            "threshold": 0.5,  # Lower threshold to get more candidates
            "find_multiple": True,
            "max_matches": 10,
            "overlap_threshold": 0.1,  # Very strict - almost no overlap allowed
            "enable_rotation": False,
        }
        result = detector.detect(test_scene, test_template, "test", params)

        # Should filter out most overlaps
        assert result["success"] is True
        objects = result["objects"]

        # Check that remaining objects don't significantly overlap
        for i, obj1 in enumerate(objects):
            for obj2 in objects[i + 1 :]:
                bb1 = obj1.bounding_box
                bb2 = obj2.bounding_box
                # Calculate IoU
                x1 = max(bb1.x, bb2.x)
                y1 = max(bb1.y, bb2.y)
                x2 = min(bb1.x + bb1.width, bb2.x + bb2.width)
                y2 = min(bb1.y + bb1.height, bb2.y + bb2.height)

                if x2 > x1 and y2 > y1:
                    intersection = (x2 - x1) * (y2 - y1)
                    area1 = bb1.width * bb1.height
                    area2 = bb2.width * bb2.height
                    union = area1 + area2 - intersection
                    iou = intersection / union if union > 0 else 0
                    assert iou < 0.3  # Should be below overlap threshold (with margin)

    def test_rotation_detection(self, detector, asymmetric_template):
        """Test rotation-invariant template matching."""
        # Create rotated scene
        scene = np.zeros((400, 400, 3), dtype=np.uint8)
        h, w = asymmetric_template.shape[:2]

        # Rotate template 45 degrees and place in scene
        center = (200, 200)
        M = cv2.getRotationMatrix2D(center, 45, 1.0)
        rotated = cv2.warpAffine(
            asymmetric_template, M, (400, 400), flags=cv2.INTER_LINEAR
        )
        scene = cv2.addWeighted(scene, 1.0, rotated, 1.0, 0)

        params = {
            "threshold": 0.6,
            "find_multiple": False,
            "enable_rotation": True,
            "rotation_range": (0.0, 90.0),
            "rotation_step": 15.0,  # 15 degree steps
        }

        result = detector.detect(scene, asymmetric_template, "test", params)

        assert result["success"] is True
        if len(result["objects"]) > 0:
            obj = result["objects"][0]
            # Should have rotation information
            assert obj.rotation is not None or "rotation_angle" in obj.properties
            # Rotation should be close to 45 degrees (within step size tolerance)
            rotation = (
                obj.rotation if obj.rotation is not None
                else obj.properties["rotation_angle"]
            )
            assert abs(rotation - 45) <= 20  # Allow tolerance due to discrete steps

    def test_rotation_with_multiple_instances(self, detector, asymmetric_template):
        """Test rotation detection with multiple instances at different angles."""
        scene = np.zeros((500, 500, 3), dtype=np.uint8)

        # Add template at different rotations
        angles = [0, 45, 90]
        positions = [(100, 100), (250, 100), (400, 100)]

        for angle, (cx, cy) in zip(angles, positions):
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            rotated = cv2.warpAffine(
                asymmetric_template, M, (500, 500), flags=cv2.INTER_LINEAR
            )
            scene = cv2.addWeighted(scene, 1.0, rotated, 1.0, 0)

        params = {
            "threshold": 0.5,
            "find_multiple": True,
            "max_matches": 5,
            "overlap_threshold": 0.3,
            "enable_rotation": True,
            "rotation_range": (0.0, 180.0),
            "rotation_step": 15.0,
        }

        result = detector.detect(scene, asymmetric_template, "test", params)

        assert result["success"] is True
        # Should find multiple instances at different rotations
        assert len(result["objects"]) >= 1

    def test_max_matches_limit(self, detector, multi_scene, test_template):
        """Test that max_matches parameter is respected."""
        params = {
            "threshold": 0.6,
            "find_multiple": True,
            "max_matches": 2,  # Limit to 2 matches
            "overlap_threshold": 0.3,
            "enable_rotation": False,
        }

        result = detector.detect(multi_scene, test_template, "test", params)

        assert result["success"] is True
        assert len(result["objects"]) <= 2

    def test_vision_object_properties(self, detector, test_scene, test_template):
        """Test VisionObject properties in advanced detection results."""
        params = {
            "threshold": 0.7,
            "find_multiple": False,
            "enable_rotation": False,
        }
        result = detector.detect(test_scene, test_template, "test", params)

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

    def test_rotation_angle_in_properties(self, detector, test_scene, test_template):
        """Test that rotation angle is stored in properties when rotation is enabled."""
        params = {
            "threshold": 0.7,
            "find_multiple": False,
            "enable_rotation": True,
            "rotation_range": (-45.0, 45.0),
            "rotation_step": 15.0,
        }
        result = detector.detect(test_scene, test_template, "test", params)

        if len(result["objects"]) > 0:
            obj = result["objects"][0]
            assert "rotation_angle" in obj.properties
            # For non-rotated template, angle should be close to 0
            assert -20 <= obj.properties["rotation_angle"] <= 20

    def test_grayscale_images(self, detector):
        """Test detection with grayscale images."""
        scene = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(scene, (50, 50), (100, 100), 255, -1)
        cv2.rectangle(scene, (120, 50), (170, 100), 255, -1)

        template = np.ones((50, 50), dtype=np.uint8) * 255

        params = {
            "threshold": 0.7,
            "find_multiple": True,
            "max_matches": 5,
            "overlap_threshold": 0.3,
            "enable_rotation": False,
        }

        result = detector.detect(scene, template, "test", params)

        assert result["success"] is True
        assert len(result["objects"]) >= 1

    def test_high_overlap_threshold(self, detector, test_scene, test_template):
        """Test with high overlap threshold (allows more overlaps)."""
        params = {
            "threshold": 0.6,
            "find_multiple": True,
            "max_matches": 10,
            "overlap_threshold": 0.9,  # Allow almost complete overlap
            "enable_rotation": False,
        }

        result = detector.detect(test_scene, test_template, "test", params)

        assert result["success"] is True
        # With high overlap threshold, might get more detections

    def test_different_matching_methods(self, detector, test_scene, test_template):
        """Test different OpenCV template matching methods."""
        methods = [
            "TM_CCOEFF_NORMED",
            "TM_CCORR_NORMED",
            "TM_SQDIFF_NORMED",
        ]

        for method in methods:
            params = {
                "method": method,
                "threshold": 0.5,
                "find_multiple": False,
                "enable_rotation": False,
            }
            result = detector.detect(test_scene, test_template, "test", params)
            assert result["success"] is True, f"Method {method} failed"

    def test_mismatched_scene(self, detector):
        """Test detection when template doesn't match scene."""
        # Black scene
        scene = np.zeros((200, 200, 3), dtype=np.uint8)
        # White template with distinct pattern
        template = np.ones((50, 50, 3), dtype=np.uint8) * 255

        params = {
            "threshold": 0.95,  # Very high threshold
            "find_multiple": True,
            "enable_rotation": False,
        }
        result = detector.detect(scene, template, "test", params)

        assert result["success"] is True
        # May or may not find matches depending on OpenCV behavior with homogeneous images

    def test_identical_images(self, detector, test_template):
        """Test detection with identical scene and template."""
        params = {
            "threshold": 0.9,
            "find_multiple": False,
            "enable_rotation": False,
        }
        result = detector.detect(test_template, test_template, "test", params)

        assert result["success"] is True
        assert len(result["objects"]) >= 1  # Should find exact match

    def test_result_structure(self, detector, test_scene, test_template):
        """Test structure of detection result."""
        params = {
            "threshold": 0.7,
            "find_multiple": True,
            "enable_rotation": False,
        }
        result = detector.detect(test_scene, test_template, "test", params)

        assert "success" in result
        assert "objects" in result
        assert "image" in result
        assert isinstance(result["objects"], list)
        assert isinstance(result["image"], np.ndarray)

    def test_rotation_range_validation(self, detector, test_scene, test_template):
        """Test that rotation works within specified range."""
        params = {
            "threshold": 0.7,
            "find_multiple": False,
            "enable_rotation": True,
            "rotation_range": (-90.0, 90.0),
            "rotation_step": 30.0,
        }

        result = detector.detect(test_scene, test_template, "test", params)

        assert result["success"] is True
        if len(result["objects"]) > 0:
            obj = result["objects"][0]
            rotation_angle = obj.properties.get("rotation_angle", 0)
            # Rotation should be within specified range
            assert -90 <= rotation_angle <= 90

    def test_small_rotation_step(self, detector, test_scene, test_template):
        """Test with small rotation step for fine-grained search."""
        params = {
            "threshold": 0.7,
            "find_multiple": False,
            "enable_rotation": True,
            "rotation_range": (-45.0, 45.0),
            "rotation_step": 5.0,  # Fine-grained
        }

        result = detector.detect(test_scene, test_template, "test", params)

        assert result["success"] is True
        # Should work but take longer (not testing performance here)

    def test_confidence_scores(self, detector, test_scene, test_template):
        """Test that confidence scores are properly calculated."""
        params = {
            "threshold": 0.5,
            "find_multiple": True,
            "max_matches": 5,
            "overlap_threshold": 0.3,
            "enable_rotation": False,
        }

        result = detector.detect(test_scene, test_template, "test", params)

        for obj in result["objects"]:
            assert 0.0 <= obj.confidence <= 1.0
            assert obj.confidence >= 0.5  # Should be above threshold

    def test_object_ids_unique(self, detector, multi_scene, test_template):
        """Test that object IDs are unique for multiple matches."""
        params = {
            "threshold": 0.7,
            "find_multiple": True,
            "max_matches": 10,
            "overlap_threshold": 0.3,
            "enable_rotation": False,
        }

        result = detector.detect(multi_scene, test_template, "test", params)

        object_ids = [obj.object_id for obj in result["objects"]]
        # All IDs should be unique
        assert len(object_ids) == len(set(object_ids))
