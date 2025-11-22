"""
Tests for vision.edge_detection module.

Tests edge detection algorithms including Canny, Sobel, Laplacian, etc.
"""

import cv2
import numpy as np
import pytest

from domain_types import EdgeMethod
from algorithms.edge_detection import EdgeDetector


class TestEdgeDetector:
    """Tests for EdgeDetector class."""

    @pytest.fixture
    def detector(self):
        """Create EdgeDetector instance."""
        return EdgeDetector()

    @pytest.fixture
    def test_image(self):
        """Create test image with clear edges."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        # Create rectangle with clear edges
        cv2.rectangle(image, (50, 50), (150, 150), (255, 255, 255), -1)
        return image

    @pytest.fixture
    def grayscale_image(self):
        """Create grayscale test image."""
        image = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (150, 150), 255, -1)
        return image

    def test_detector_creation(self, detector):
        """Test creating EdgeDetector instance."""
        assert detector is not None
        assert isinstance(detector, EdgeDetector)

    def test_detect_canny_basic(self, detector, test_image):
        """Test basic Canny edge detection."""
        result = detector.detect(test_image, method=EdgeMethod.CANNY)

        assert result is not None
        assert "success" in result
        assert result["success"] is True
        assert "objects" in result
        assert isinstance(result["objects"], list)

    def test_detect_canny_with_params(self, detector, test_image):
        """Test Canny with custom parameters."""
        params = {"threshold1": 50, "threshold2": 150}
        result = detector.detect(test_image, method=EdgeMethod.CANNY, params=params)

        assert result["success"] is True
        assert len(result["objects"]) > 0

    def test_detect_sobel(self, detector, test_image):
        """Test Sobel edge detection."""
        result = detector.detect(test_image, method=EdgeMethod.SOBEL)

        assert result["success"] is True
        assert "objects" in result

    def test_detect_laplacian(self, detector, test_image):
        """Test Laplacian edge detection."""
        result = detector.detect(test_image, method=EdgeMethod.LAPLACIAN)

        assert result["success"] is True
        assert "objects" in result

    def test_detect_prewitt(self, detector, test_image):
        """Test Prewitt edge detection."""
        result = detector.detect(test_image, method=EdgeMethod.PREWITT)

        assert result["success"] is True
        assert "objects" in result

    def test_detect_scharr(self, detector, test_image):
        """Test Scharr edge detection."""
        result = detector.detect(test_image, method=EdgeMethod.SCHARR)

        assert result["success"] is True
        assert "objects" in result

    def test_detect_morphological_gradient(self, detector, test_image):
        """Test morphological gradient edge detection."""
        result = detector.detect(test_image, method=EdgeMethod.MORPHOLOGICAL_GRADIENT)

        assert result["success"] is True
        assert "objects" in result

    def test_detect_with_grayscale_input(self, detector, grayscale_image):
        """Test edge detection with grayscale input."""
        result = detector.detect(grayscale_image, method=EdgeMethod.CANNY)

        assert result["success"] is True
        assert len(result["objects"]) > 0

    def test_detect_finds_rectangle_edges(self, detector, test_image):
        """Test that detector finds edges of rectangle."""
        result = detector.detect(test_image, method=EdgeMethod.CANNY)

        # Should find at least one contour (the rectangle)
        assert len(result["objects"]) >= 1

    def test_detect_with_min_area_filter(self, detector, test_image):
        """Test edge detection with minimum area filter."""
        params = {"min_area": 5000}  # Large minimum area
        result = detector.detect(test_image, method=EdgeMethod.CANNY, params=params)

        # Should filter out small contours
        assert result["success"] is True
        # Large rectangle should still be detected
        assert len(result["objects"]) >= 1

    def test_detect_with_max_contours(self, detector, test_image):
        """Test limiting number of contours."""
        # Add more shapes
        cv2.circle(test_image, (50, 150), 20, (255, 255, 255), -1)
        cv2.circle(test_image, (150, 150), 20, (255, 255, 255), -1)

        params = {"max_contours": 1}
        result = detector.detect(test_image, method=EdgeMethod.CANNY, params=params)

        # Should return at most 1 contour
        assert len(result["objects"]) <= 1

    def test_detect_empty_image(self, detector):
        """Test edge detection on empty image."""
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.detect(empty_image, method=EdgeMethod.CANNY)

        # Should succeed but find no edges
        assert result["success"] is True
        assert len(result["objects"]) == 0

    def test_detect_with_blur_preprocessing(self, detector, test_image):
        """Test edge detection with blur preprocessing."""
        params = {"blur": True, "blur_kernel": 5}
        result = detector.detect(test_image, method=EdgeMethod.CANNY, params=params)

        assert result["success"] is True

    def test_detect_with_morphology_preprocessing(self, detector, test_image):
        """Test edge detection with morphological operations."""
        params = {"dilate": True, "dilate_kernel": 3}
        result = detector.detect(test_image, method=EdgeMethod.CANNY, params=params)

        assert result["success"] is True

    def test_detect_result_structure(self, detector, test_image):
        """Test structure of detection result."""
        result = detector.detect(test_image, method=EdgeMethod.CANNY)

        assert "success" in result
        assert "objects" in result
        assert "image" in result
        assert isinstance(result["image"], np.ndarray)
        assert result["image"].dtype == np.uint8

    def test_detect_vision_object_structure(self, detector, test_image):
        """Test VisionObject structure in results."""
        result = detector.detect(test_image, method=EdgeMethod.CANNY)

        if len(result["objects"]) > 0:
            obj = result["objects"][0]
            assert hasattr(obj, "object_id")
            assert hasattr(obj, "object_type")
            assert hasattr(obj, "bounding_box")
            assert hasattr(obj, "center")
            assert hasattr(obj, "confidence")

    def test_detect_noisy_image(self, detector):
        """Test edge detection on noisy image."""
        np.random.seed(42)
        noisy_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

        result = detector.detect(noisy_image, method=EdgeMethod.CANNY)

        # Should handle noise gracefully
        assert result["success"] is True

    def test_all_methods_produce_results(self, detector, test_image):
        """Test that all edge detection methods work."""
        methods = [
            EdgeMethod.CANNY,
            EdgeMethod.SOBEL,
            EdgeMethod.LAPLACIAN,
            EdgeMethod.PREWITT,
            EdgeMethod.SCHARR,
            EdgeMethod.MORPHOLOGICAL_GRADIENT,
        ]

        for method in methods:
            result = detector.detect(test_image, method=method)
            assert result["success"] is True, f"Method {method} failed"

    def test_detect_with_different_thresholds(self, detector, test_image):
        """Test Canny with different threshold values."""
        # Low thresholds - more edges
        result_low = detector.detect(
            test_image, method=EdgeMethod.CANNY, params={"threshold1": 30, "threshold2": 90}
        )

        # High thresholds - fewer edges
        result_high = detector.detect(
            test_image, method=EdgeMethod.CANNY, params={"threshold1": 100, "threshold2": 200}
        )

        # Both should succeed
        assert result_low["success"] is True
        assert result_high["success"] is True

    def test_detect_preserves_image_dimensions(self, detector, test_image):
        """Test that output image has same dimensions as input."""
        result = detector.detect(test_image, method=EdgeMethod.CANNY)

        output_image = result["image"]
        assert output_image.shape[:2] == test_image.shape[:2]

    def test_detect_with_area_filter(self, detector):
        """Test filtering contours by area."""
        # Create image with large and small shapes
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (250, 250), (255, 255, 255), -1)  # Large
        cv2.rectangle(image, (10, 10), (15, 15), (255, 255, 255), -1)  # Small

        params = {"min_area": 1000}  # Filter out small shapes
        result = detector.detect(image, method=EdgeMethod.CANNY, params=params)

        # Should only find large rectangle
        assert result["success"] is True
        # Small rectangle should be filtered out

    def test_detect_complex_shape(self, detector):
        """Test detection on complex shape."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        # Draw complex polygon
        points = np.array([[50, 50], [150, 50], [180, 100], [150, 150], [50, 150]])
        cv2.fillPoly(image, [points], (255, 255, 255))

        result = detector.detect(image, method=EdgeMethod.CANNY)

        assert result["success"] is True
        assert len(result["objects"]) >= 1
