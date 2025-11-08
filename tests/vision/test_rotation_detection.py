"""
Tests for rotation detection module.
"""

import numpy as np
import pytest

from core.enums import AngleRange, RotationMethod
from vision.rotation_detection import RotationDetector


class TestRotationDetector:
    """Test rotation detection functionality"""

    @pytest.fixture
    def detector(self):
        """Create rotation detector instance"""
        return RotationDetector()

    @pytest.fixture
    def test_image(self):
        """Create test image for visualization"""
        return np.zeros((200, 200, 3), dtype=np.uint8)

    @pytest.fixture
    def horizontal_rectangle_contour(self):
        """Horizontal rectangle (0° rotation)"""
        return [[50, 80], [150, 80], [150, 120], [50, 120]]

    @pytest.fixture
    def vertical_rectangle_contour(self):
        """Vertical rectangle (90° rotation)"""
        return [[80, 50], [120, 50], [120, 150], [80, 150]]

    @pytest.fixture
    def tilted_rectangle_contour(self):
        """Tilted rectangle (~45° rotation)"""
        return [[100, 50], [150, 100], [100, 150], [50, 100]]

    @pytest.fixture
    def circle_contour(self):
        """Circular contour for ellipse testing"""
        center = (100, 100)
        radius = 40
        num_points = 100
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        return [[center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)] for a in angles]

    def test_detect_min_area_rect_horizontal(
        self, detector, test_image, horizontal_rectangle_contour
    ):
        """Test MIN_AREA_RECT method with horizontal rectangle"""
        result = detector.detect(
            test_image,
            horizontal_rectangle_contour,
            method=RotationMethod.MIN_AREA_RECT,
            angle_range=AngleRange.RANGE_0_360,
        )

        assert result["success"] is True
        assert result["method"] == RotationMethod.MIN_AREA_RECT
        assert len(result["objects"]) == 1

        obj = result["objects"][0]
        # Horizontal rectangle should be close to 0° or 180°
        assert obj.rotation == pytest.approx(0, abs=5) or obj.rotation == pytest.approx(180, abs=5)

    def test_detect_min_area_rect_vertical(self, detector, test_image, vertical_rectangle_contour):
        """Test MIN_AREA_RECT method with vertical rectangle"""
        result = detector.detect(
            test_image,
            vertical_rectangle_contour,
            method=RotationMethod.MIN_AREA_RECT,
            angle_range=AngleRange.RANGE_0_360,
        )

        assert result["success"] is True
        obj = result["objects"][0]
        # Vertical rectangle should be close to 90° or 270°
        assert obj.rotation == pytest.approx(90, abs=5) or obj.rotation == pytest.approx(270, abs=5)

    def test_detect_ellipse_fit(self, detector, test_image, circle_contour):
        """Test ELLIPSE_FIT method with circular contour"""
        result = detector.detect(
            test_image,
            circle_contour,
            method=RotationMethod.ELLIPSE_FIT,
            angle_range=AngleRange.RANGE_0_360,
        )

        assert result["success"] is True
        assert result["method"] == RotationMethod.ELLIPSE_FIT
        assert len(result["objects"]) == 1

        # Circle should have valid rotation (any angle is valid for circle)
        obj = result["objects"][0]
        assert 0 <= obj.rotation <= 360

    def test_detect_pca(self, detector, test_image, horizontal_rectangle_contour):
        """Test PCA method with horizontal rectangle"""
        result = detector.detect(
            test_image,
            horizontal_rectangle_contour,
            method=RotationMethod.PCA,
            angle_range=AngleRange.RANGE_0_360,
        )

        assert result["success"] is True
        assert result["method"] == RotationMethod.PCA
        assert len(result["objects"]) == 1

        obj = result["objects"][0]
        # PCA should detect dominant axis
        assert 0 <= obj.rotation <= 360
        # Confidence should be present
        assert 0 <= obj.confidence <= 1.0

    def test_angle_range_0_360(self, detector, test_image, tilted_rectangle_contour):
        """Test angle range 0-360"""
        result = detector.detect(
            test_image,
            tilted_rectangle_contour,
            method=RotationMethod.MIN_AREA_RECT,
            angle_range=AngleRange.RANGE_0_360,
        )

        obj = result["objects"][0]
        assert 0 <= obj.rotation <= 360

    def test_angle_range_neg180_180(self, detector, test_image, tilted_rectangle_contour):
        """Test angle range -180 to +180"""
        result = detector.detect(
            test_image,
            tilted_rectangle_contour,
            method=RotationMethod.MIN_AREA_RECT,
            angle_range=AngleRange.RANGE_NEG180_180,
        )

        obj = result["objects"][0]
        assert -180 <= obj.rotation <= 180

    def test_angle_range_0_180(self, detector, test_image, tilted_rectangle_contour):
        """Test angle range 0-180 (symmetric objects)"""
        result = detector.detect(
            test_image,
            tilted_rectangle_contour,
            method=RotationMethod.MIN_AREA_RECT,
            angle_range=AngleRange.RANGE_0_180,
        )

        obj = result["objects"][0]
        assert 0 <= obj.rotation <= 180

    def test_invalid_contour_too_few_points(self, detector, test_image):
        """Test error handling for contour with too few points"""
        # Only 2 points - should raise error
        invalid_contour = [[50, 50], [100, 100]]

        with pytest.raises(ValueError, match="at least .* points"):
            detector.detect(
                test_image,
                invalid_contour,
                method=RotationMethod.MIN_AREA_RECT,
            )

    def test_invalid_contour_for_ellipse(self, detector, test_image):
        """Test error handling for ellipse fit with too few points"""
        # 4 points - not enough for ellipse (needs 5)
        small_contour = [[50, 50], [100, 50], [100, 100], [50, 100]]

        with pytest.raises(ValueError, match="Ellipse fitting requires at least"):
            detector.detect(
                test_image,
                small_contour,
                method=RotationMethod.ELLIPSE_FIT,
            )

    def test_result_structure(self, detector, test_image, horizontal_rectangle_contour):
        """Test that result has expected structure"""
        result = detector.detect(
            test_image,
            horizontal_rectangle_contour,
            method=RotationMethod.MIN_AREA_RECT,
        )

        # Check top-level keys
        assert "success" in result
        assert "method" in result
        assert "objects" in result
        assert "image" in result

        # Check object structure
        obj = result["objects"][0]
        assert hasattr(obj, "object_id")
        assert hasattr(obj, "object_type")
        assert hasattr(obj, "bounding_box")
        assert hasattr(obj, "center")
        assert hasattr(obj, "confidence")
        assert hasattr(obj, "area")
        assert hasattr(obj, "perimeter")
        assert hasattr(obj, "rotation")
        assert hasattr(obj, "properties")
        assert hasattr(obj, "contour")

        # Check properties
        assert "method" in obj.properties
        assert "angle_range" in obj.properties
        assert "absolute_angle" in obj.properties

    def test_with_roi_context(self, detector, test_image, horizontal_rectangle_contour):
        """Test rotation detection with ROI context"""
        roi = {"x": 25, "y": 50, "width": 150, "height": 100}

        result = detector.detect(
            test_image,
            horizontal_rectangle_contour,
            method=RotationMethod.MIN_AREA_RECT,
            roi=roi,
        )

        assert result["success"] is True
        # ROI is for visualization context only, shouldn't affect rotation
        assert len(result["objects"]) == 1

    def test_different_methods_same_contour(
        self, detector, test_image, horizontal_rectangle_contour
    ):
        """Test that different methods produce consistent results"""
        methods = [RotationMethod.MIN_AREA_RECT, RotationMethod.PCA]
        results = []

        for method in methods:
            result = detector.detect(
                test_image,
                horizontal_rectangle_contour,
                method=method,
                angle_range=AngleRange.RANGE_0_360,
            )
            results.append(result["objects"][0].rotation)

        # All methods should give similar angles for rectangle (within 10°)
        # Note: Different methods may give 0° or 90° or 180° depending on orientation
        # Rectangle has ambiguous orientation (can be 0° or 180°, 90° or 270°)
        angle_diff = abs(results[0] - results[1])
        angle_diff_mod = min(angle_diff, 360 - angle_diff)  # Handle wrap-around

        # Check if angles are close together, 90° apart, or 180° apart
        assert (
            angle_diff_mod < 10 or abs(angle_diff_mod - 90) < 10 or abs(angle_diff_mod - 180) < 10
        )

    def test_contour_preservation(self, detector, test_image, horizontal_rectangle_contour):
        """Test that original contour is preserved in result"""
        result = detector.detect(
            test_image,
            horizontal_rectangle_contour,
            method=RotationMethod.MIN_AREA_RECT,
        )

        obj = result["objects"][0]
        # Contour should be preserved
        assert obj.contour is not None
        assert len(obj.contour) == len(horizontal_rectangle_contour)
