"""
Tests for image utility functions.

Tests cover image conversion, processing, and geometry functions.
"""

import base64

import cv2
import numpy as np
import pytest

from core.image.converters import encode_image_to_base64, ensure_bgr, ensure_grayscale
from core.image.geometry import calculate_contour_properties, normalize_angle


class TestImageFormatConversion:
    """Test image format conversion utilities"""

    def test_ensure_bgr_from_grayscale(self):
        """Test converting grayscale to BGR"""
        # Create grayscale image (2D array)
        gray = np.ones((100, 100), dtype=np.uint8) * 128

        # Convert to BGR
        bgr = ensure_bgr(gray)

        # Should be 3-channel
        assert len(bgr.shape) == 3
        assert bgr.shape[2] == 3
        assert bgr.shape[0] == 100
        assert bgr.shape[1] == 100

        # All channels should be equal (grayscale converted to BGR)
        assert np.all(bgr[:, :, 0] == bgr[:, :, 1])
        assert np.all(bgr[:, :, 1] == bgr[:, :, 2])

    def test_ensure_bgr_from_bgr(self):
        """Test that BGR image is returned as copy"""
        # Create BGR image
        bgr_original = np.ones((100, 100, 3), dtype=np.uint8) * 128

        # Should return copy
        bgr_result = ensure_bgr(bgr_original)

        assert bgr_result.shape == bgr_original.shape
        assert np.array_equal(bgr_result, bgr_original)
        # Should be a copy, not same object
        assert bgr_result is not bgr_original

    def test_ensure_grayscale_from_bgr(self):
        """Test converting BGR to grayscale"""
        # Create BGR image with different channel values
        bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr[:, :, 0] = 100  # Blue
        bgr[:, :, 1] = 150  # Green
        bgr[:, :, 2] = 200  # Red

        # Convert to grayscale
        gray = ensure_grayscale(bgr)

        # Should be single channel
        assert len(gray.shape) == 2
        assert gray.shape[0] == 100
        assert gray.shape[1] == 100

        # Grayscale value should be weighted average (OpenCV formula)
        # Gray = 0.299*R + 0.587*G + 0.114*B
        expected = int(0.299 * 200 + 0.587 * 150 + 0.114 * 100)
        assert abs(gray[0, 0] - expected) < 2  # Allow small rounding difference

    def test_ensure_grayscale_from_grayscale(self):
        """Test that grayscale image is returned as copy"""
        # Create grayscale image
        gray_original = np.ones((100, 100), dtype=np.uint8) * 128

        # Should return copy
        gray_result = ensure_grayscale(gray_original)

        assert gray_result.shape == gray_original.shape
        assert np.array_equal(gray_result, gray_original)
        # Should be a copy, not same object
        assert gray_result is not gray_original


class TestBase64Encoding:
    """Test base64 image encoding"""

    def test_encode_image_to_base64_png(self):
        """Test encoding image to base64 PNG format"""
        # Create simple test image
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        image[10:40, 10:40] = [255, 0, 0]  # Red square

        # Encode to base64
        encoded = encode_image_to_base64(image, format=".png")

        # Should be valid base64 string
        assert isinstance(encoded, str)
        assert len(encoded) > 0

        # Should be decodable
        decoded_bytes = base64.b64decode(encoded)
        assert len(decoded_bytes) > 0

        # Should start with PNG signature
        assert decoded_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    def test_encode_image_to_base64_jpg(self):
        """Test encoding image to base64 JPG format"""
        # Create test image
        image = np.ones((50, 50, 3), dtype=np.uint8) * 128

        # Encode to base64 JPEG
        encoded = encode_image_to_base64(image, format=".jpg")

        # Should be valid base64 string
        assert isinstance(encoded, str)
        assert len(encoded) > 0

        # Should be decodable
        decoded_bytes = base64.b64decode(encoded)

        # JPEG starts with FFD8
        assert decoded_bytes[:2] == b"\xff\xd8"

    def test_encode_decode_roundtrip(self):
        """Test that encoding and decoding returns similar image"""
        # Create test image with pattern
        original = np.zeros((100, 100, 3), dtype=np.uint8)
        original[25:75, 25:75] = [100, 150, 200]

        # Encode and decode
        encoded = encode_image_to_base64(original, format=".png")
        decoded_bytes = base64.b64decode(encoded)
        decoded = cv2.imdecode(np.frombuffer(decoded_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Should have same shape
        assert decoded.shape == original.shape

        # PNG is lossless, so should be identical
        assert np.array_equal(decoded, original)


class TestAngleNormalization:
    """Test angle normalization to different ranges"""

    def test_normalize_angle_0_360_positive(self):
        """Test normalizing positive angles to 0-360 range"""
        # Test various angles
        assert normalize_angle(0, "0_360") == pytest.approx(0)
        assert normalize_angle(np.pi / 2, "0_360") == pytest.approx(90)
        assert normalize_angle(np.pi, "0_360") == pytest.approx(180)
        assert normalize_angle(3 * np.pi / 2, "0_360") == pytest.approx(270)
        assert normalize_angle(2 * np.pi, "0_360") == pytest.approx(0)  # Wraps to 0

    def test_normalize_angle_0_360_negative(self):
        """Test normalizing negative angles to 0-360 range"""
        # Negative angles should wrap to positive
        assert normalize_angle(-np.pi / 2, "0_360") == pytest.approx(270)
        assert normalize_angle(-np.pi, "0_360") == pytest.approx(180)
        assert normalize_angle(-3 * np.pi / 2, "0_360") == pytest.approx(90)

    def test_normalize_angle_neg180_180(self):
        """Test normalizing to -180 to +180 range"""
        assert normalize_angle(0, "-180_180") == pytest.approx(0)
        assert normalize_angle(np.pi / 2, "-180_180") == pytest.approx(90)
        assert normalize_angle(np.pi, "-180_180") == pytest.approx(180)

        # Angles > 180 should wrap to negative
        assert normalize_angle(3 * np.pi / 2, "-180_180") == pytest.approx(-90)
        assert normalize_angle(7 * np.pi / 4, "-180_180") == pytest.approx(-45)

    def test_normalize_angle_0_180(self):
        """Test normalizing to 0-180 range (symmetric objects)"""
        assert normalize_angle(0, "0_180") == pytest.approx(0)
        assert normalize_angle(np.pi / 2, "0_180") == pytest.approx(90)
        assert normalize_angle(np.pi, "0_180") == pytest.approx(0)  # Wraps

        # Angles > 180 should wrap back by taking modulo 180
        # 270° % 180 = 90°
        assert normalize_angle(3 * np.pi / 2, "0_180") == pytest.approx(90)
        # 315° % 180 = 135°
        assert normalize_angle(7 * np.pi / 4, "0_180") == pytest.approx(135)

    def test_normalize_angle_return_radians(self):
        """Test returning result in radians instead of degrees"""
        # With return_radians=True, should return radians
        result = normalize_angle(np.pi / 2, "0_360", return_radians=True)
        assert 1.5 < result < 1.6  # Close to π/2
        assert result != 90  # Not in degrees

    def test_normalize_angle_large_values(self):
        """Test with angles > 2π"""
        # Should handle multiple rotations
        assert normalize_angle(5 * np.pi, "0_360") == pytest.approx(180)
        assert normalize_angle(10 * np.pi, "0_360") == pytest.approx(0)


class TestContourProperties:
    """Test contour property calculations"""

    def test_calculate_rectangle_contour(self):
        """Test calculating properties of a rectangle contour"""
        # Create rectangular contour (OpenCV format: Nx1x2)
        contour = np.array([[[10, 10]], [[10, 50]], [[90, 50]], [[90, 10]]], dtype=np.float32)

        props = calculate_contour_properties(contour)

        # Check all expected keys
        assert "area" in props
        assert "perimeter" in props
        assert "center" in props
        assert "bounding_box" in props

        # Rectangle is 80x40 = 3200 area
        assert props["area"] == pytest.approx(3200, rel=0.01)

        # Perimeter is 2*(80+40) = 240
        assert props["perimeter"] == pytest.approx(240, rel=0.01)

        # Center should be at (50, 30)
        cx, cy = props["center"]
        assert cx == pytest.approx(50, abs=1)
        assert cy == pytest.approx(30, abs=1)

        # Bounding box (OpenCV uses pixel-based coordinates)
        x, y, w, h = props["bounding_box"]
        assert x == pytest.approx(10, abs=1)
        assert y == pytest.approx(10, abs=1)
        assert w == pytest.approx(80, abs=2)  # Allow small difference for rounding
        assert h == pytest.approx(40, abs=2)

    def test_calculate_circle_contour(self):
        """Test calculating properties of a circular contour"""
        # Generate circle contour
        center = (50, 50)
        radius = 30
        num_points = 100
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        contour = np.array(
            [[[center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)]] for a in angles],
            dtype=np.float32,
        )

        props = calculate_contour_properties(contour)

        # Circle area = π * r^2 ≈ 2827
        expected_area = np.pi * radius**2
        assert props["area"] == pytest.approx(expected_area, rel=0.1)

        # Circle perimeter = 2 * π * r ≈ 188
        expected_perimeter = 2 * np.pi * radius
        assert props["perimeter"] == pytest.approx(expected_perimeter, rel=0.1)

        # Center should be at (50, 50)
        cx, cy = props["center"]
        assert cx == pytest.approx(center[0], abs=2)
        assert cy == pytest.approx(center[1], abs=2)

    def test_calculate_triangle_contour(self):
        """Test calculating properties of a triangle contour"""
        # Equilateral triangle
        contour = np.array([[[50, 10]], [[10, 90]], [[90, 90]]], dtype=np.float32)

        props = calculate_contour_properties(contour)

        # Triangle area (base=80, height≈70)
        assert props["area"] > 0
        assert props["area"] < 5000  # Reasonable bound

        # Should have valid center
        cx, cy = props["center"]
        assert 10 < cx < 90
        assert 10 < cy < 90

    def test_calculate_complex_contour(self):
        """Test with more complex contour shape"""
        # Star-like shape with multiple points
        contour = np.array(
            [
                [[50, 10]],
                [[60, 40]],
                [[90, 40]],
                [[70, 60]],
                [[80, 90]],
                [[50, 70]],
                [[20, 90]],
                [[30, 60]],
                [[10, 40]],
                [[40, 40]],
            ],
            dtype=np.float32,
        )

        props = calculate_contour_properties(contour)

        # Should calculate without errors
        assert props["area"] > 0
        assert props["perimeter"] > 0
        assert len(props["center"]) == 2
        assert len(props["bounding_box"]) == 4
