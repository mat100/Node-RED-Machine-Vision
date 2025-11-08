"""
Tests for color detection module.
"""

import numpy as np
import pytest

from vision.color_detection import ColorDetector


class TestColorDetector:
    """Test color detection functionality"""

    @pytest.fixture
    def detector(self):
        """Create color detector instance"""
        return ColorDetector()

    @pytest.fixture
    def red_image(self):
        """Create image with dominant red color"""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        # BGR format: Red is [0, 0, 255]
        image[:, :] = [0, 0, 255]
        return image

    @pytest.fixture
    def blue_image(self):
        """Create image with dominant blue color"""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        # BGR format: Blue is [255, 0, 0]
        image[:, :] = [255, 0, 0]
        return image

    @pytest.fixture
    def green_image(self):
        """Create image with dominant green color"""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        # BGR format: Green is [0, 255, 0]
        image[:, :] = [0, 255, 0]
        return image

    @pytest.fixture
    def mixed_color_image(self):
        """Create image with multiple colors"""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Top half red, bottom half blue
        image[:50, :] = [0, 0, 255]  # Red
        image[50:, :] = [255, 0, 0]  # Blue
        return image

    def test_detect_histogram_red(self, detector, red_image):
        """Test histogram method detects red color"""
        result = detector.detect(red_image, method="histogram")

        assert result["success"] is True
        assert result["method"] == "histogram"
        assert len(result["objects"]) == 1

        obj = result["objects"][0]
        assert obj.properties["dominant_color"] == "red"
        # Red should have high percentage
        assert obj.properties["color_percentages"]["red"] > 80

    def test_detect_histogram_blue(self, detector, blue_image):
        """Test histogram method detects blue color"""
        result = detector.detect(blue_image, method="histogram")

        assert result["success"] is True
        obj = result["objects"][0]
        assert obj.properties["dominant_color"] == "blue"

    def test_detect_histogram_green(self, detector, green_image):
        """Test histogram method detects green color"""
        result = detector.detect(green_image, method="histogram")

        assert result["success"] is True
        obj = result["objects"][0]
        assert obj.properties["dominant_color"] == "green"

    def test_detect_with_roi(self, detector, mixed_color_image):
        """Test color detection with ROI"""
        # Detect only top half (should be red)
        roi = {"x": 0, "y": 0, "width": 100, "height": 50}

        result = detector.detect(mixed_color_image, roi=roi, method="histogram")

        assert result["success"] is True
        obj = result["objects"][0]
        assert obj.properties["dominant_color"] == "red"

    def test_detect_with_expected_color_match(self, detector, red_image):
        """Test color detection with expected color that matches"""
        result = detector.detect(
            red_image, expected_color="red", min_percentage=50.0, method="histogram"
        )

        assert result["success"] is True
        obj = result["objects"][0]
        assert obj.properties["match"] is True
        assert obj.properties["expected_color"] == "red"

    def test_detect_with_expected_color_no_match(self, detector, red_image):
        """Test color detection with expected color that doesn't match"""
        result = detector.detect(
            red_image, expected_color="blue", min_percentage=50.0, method="histogram"
        )

        assert result["success"] is True
        obj = result["objects"][0]
        assert obj.properties["match"] is False
        assert obj.properties["expected_color"] == "blue"
        assert obj.properties["dominant_color"] == "red"

    def test_detect_with_contour_mask(self, detector, mixed_color_image):
        """Test color detection with contour mask"""
        # Create contour for top-left quadrant
        contour = [[0, 0], [50, 0], [50, 50], [0, 50]]

        result = detector.detect(
            mixed_color_image, contour_points=contour, use_contour_mask=True, method="histogram"
        )

        assert result["success"] is True
        obj = result["objects"][0]
        # Top-left quadrant is red
        assert obj.properties["dominant_color"] == "red"

    def test_detect_without_contour_mask(self, detector, mixed_color_image):
        """Test color detection without using contour mask"""
        contour = [[0, 0], [50, 0], [50, 50], [0, 50]]

        result = detector.detect(
            mixed_color_image,
            contour_points=contour,
            use_contour_mask=False,  # Don't use mask
            method="histogram",
        )

        assert result["success"] is True
        # Should analyze full image, not just contour region

    def test_detect_mixed_colors(self, detector, mixed_color_image):
        """Test detection on image with multiple colors"""
        result = detector.detect(mixed_color_image, method="histogram")

        assert result["success"] is True
        obj = result["objects"][0]

        # Should detect either red or blue as dominant
        assert obj.properties["dominant_color"] in ["red", "blue"]

        # Both colors should be present in percentages
        color_pcts = obj.properties["color_percentages"]
        assert "red" in color_pcts
        assert "blue" in color_pcts

    def test_result_structure(self, detector, red_image):
        """Test that result has expected structure"""
        result = detector.detect(red_image, method="histogram")

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
        assert hasattr(obj, "properties")

        # Check properties
        assert "dominant_color" in obj.properties
        assert "color_percentages" in obj.properties
        assert "hsv_mean" in obj.properties
        assert "method" in obj.properties

    def test_confidence_calculation(self, detector, red_image):
        """Test that confidence is calculated correctly"""
        result = detector.detect(red_image, method="histogram")

        obj = result["objects"][0]
        # Confidence should be percentage of dominant color / 100
        dominant_pct = obj.properties["color_percentages"]["red"]
        expected_confidence = dominant_pct / 100.0

        assert obj.confidence == pytest.approx(expected_confidence, rel=0.01)

    def test_histogram_method_specified(self, detector, red_image):
        """Test that method parameter is correctly set to histogram"""
        result = detector.detect(red_image, method="histogram")

        assert result["success"] is True
        assert result["method"] == "histogram"
        obj = result["objects"][0]
        assert obj.properties["method"] == "histogram"

    def test_min_percentage_threshold(self, detector, mixed_color_image):
        """Test min_percentage threshold for color matching"""
        # Set very high threshold that won't be met
        result = detector.detect(
            mixed_color_image,
            expected_color="red",
            min_percentage=95.0,  # Very high threshold
            method="histogram",
        )

        obj = result["objects"][0]
        # Should not match because red is only ~50% of image
        assert obj.properties["match"] is False

    def test_hsv_mean_present(self, detector, red_image):
        """Test that HSV mean values are calculated"""
        result = detector.detect(red_image, method="histogram")

        obj = result["objects"][0]
        assert "hsv_mean" in obj.properties
        hsv_mean = obj.properties["hsv_mean"]

        # Should be list of 3 values
        assert len(hsv_mean) == 3
        # All should be integers
        assert all(isinstance(v, int) for v in hsv_mean)

    def test_full_image_no_roi(self, detector, red_image):
        """Test detection on full image without ROI"""
        result = detector.detect(red_image, roi=None, method="histogram")

        assert result["success"] is True
        obj = result["objects"][0]

        # Bounding box should cover full image
        bbox = obj.bounding_box
        assert bbox.x == 0
        assert bbox.y == 0
        assert bbox.width == 100
        assert bbox.height == 100
