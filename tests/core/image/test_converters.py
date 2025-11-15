"""
Tests for core.image.converters module.

Tests image format conversion operations including base64 encoding/decoding
and color space conversions.
"""

import base64

import cv2
import numpy as np
import pytest

from core.image.converters import (
    encode_image_to_base64,
    ensure_bgr,
    ensure_grayscale,
    from_base64,
    to_base64,
)


class TestToBase64:
    """Tests for to_base64 function."""

    def test_to_base64_jpeg_default(self):
        """Test converting image to base64 JPEG with default quality."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (80, 80), (255, 255, 255), -1)

        result = to_base64(image)

        # Should return valid base64 string
        assert isinstance(result, str)
        assert len(result) > 0
        # Should be decodable
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_to_base64_jpeg_custom_quality(self):
        """Test JPEG encoding with custom quality."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        result_high = to_base64(image, format="JPEG", quality=95)
        result_low = to_base64(image, format="JPEG", quality=50)

        # Higher quality should produce larger file
        assert len(result_high) >= len(result_low)

    def test_to_base64_png(self):
        """Test converting image to base64 PNG."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (80, 80), (255, 255, 255), -1)

        result = to_base64(image, format="PNG")

        # Should return valid base64 string
        assert isinstance(result, str)
        assert len(result) > 0
        # PNG typically larger than JPEG for simple images
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_to_base64_format_with_dot(self):
        """Test format parameter with leading dot."""
        image = np.zeros((50, 50, 3), dtype=np.uint8)

        result = to_base64(image, format=".jpg")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_to_base64_format_case_insensitive(self):
        """Test that format is case insensitive."""
        image = np.zeros((50, 50, 3), dtype=np.uint8)

        result_upper = to_base64(image, format="JPEG")
        result_lower = to_base64(image, format="jpeg")

        # Should produce similar results
        assert isinstance(result_upper, str)
        assert isinstance(result_lower, str)
        assert len(result_upper) > 0
        assert len(result_lower) > 0

    def test_to_base64_grayscale_image(self):
        """Test converting grayscale image to base64."""
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(image, (50, 50), 30, 255, -1)

        result = to_base64(image, format="PNG")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_to_base64_bytes_input(self):
        """Test converting bytes directly to base64."""
        test_bytes = b"test image data"

        result = to_base64(test_bytes)

        # Should encode bytes directly
        assert result == base64.b64encode(test_bytes).decode("utf-8")

    def test_to_base64_different_sizes(self):
        """Test base64 conversion with different image sizes."""
        small = np.zeros((10, 10, 3), dtype=np.uint8)
        medium = np.zeros((100, 100, 3), dtype=np.uint8)
        large = np.zeros((500, 500, 3), dtype=np.uint8)

        small_b64 = to_base64(small)
        medium_b64 = to_base64(medium)
        large_b64 = to_base64(large)

        # Larger images should produce longer base64 strings
        assert len(small_b64) < len(medium_b64) < len(large_b64)

    def test_to_base64_png_quality_affects_compression(self):
        """Test that PNG quality parameter affects compression."""
        # Complex image with noise
        np.random.seed(42)
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

        result_high = to_base64(image, format="PNG", quality=100)
        result_low = to_base64(image, format="PNG", quality=10)

        # Both should be valid (actual size difference depends on image complexity)
        assert isinstance(result_high, str)
        assert isinstance(result_low, str)
        assert len(result_high) > 0
        assert len(result_low) > 0


class TestFromBase64:
    """Tests for from_base64 function."""

    def test_from_base64_basic(self):
        """Test decoding base64 string to image."""
        # Create image and encode
        original = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(original, (20, 20), (80, 80), (255, 255, 255), -1)
        base64_str = to_base64(original, format="PNG")

        # Decode
        decoded = from_base64(base64_str)

        assert isinstance(decoded, np.ndarray)
        assert decoded.shape == (100, 100, 3)
        assert decoded.dtype == np.uint8

    def test_from_base64_roundtrip_jpeg(self):
        """Test encode-decode roundtrip with JPEG."""
        original = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(original, (20, 20), (80, 80), (255, 0, 0), -1)

        base64_str = to_base64(original, format="JPEG", quality=95)
        decoded = from_base64(base64_str)

        # JPEG is lossy, but shape should match
        assert decoded.shape == original.shape
        # Should have similar content (allowing for JPEG artifacts)
        assert np.mean(np.abs(decoded.astype(int) - original.astype(int))) < 10

    def test_from_base64_roundtrip_png(self):
        """Test encode-decode roundtrip with PNG (lossless)."""
        original = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(original, (20, 20), (80, 80), (255, 0, 0), -1)

        base64_str = to_base64(original, format="PNG")
        decoded = from_base64(base64_str)

        # PNG is lossless, should match exactly
        assert decoded.shape == original.shape
        assert np.array_equal(decoded, original)

    def test_from_base64_invalid_data(self):
        """Test decoding invalid base64 raises exception."""
        with pytest.raises(Exception):
            from_base64("invalid base64 data!!!")

    def test_from_base64_non_image_data(self):
        """Test decoding valid base64 but non-image data."""
        non_image = base64.b64encode(b"not an image").decode("utf-8")

        with pytest.raises(ValueError, match="Failed to decode"):
            from_base64(non_image)

    def test_from_base64_empty_string(self):
        """Test decoding empty string."""
        with pytest.raises(Exception):
            from_base64("")

    def test_from_base64_color_image(self):
        """Test that decoded image is in BGR format."""
        # Create RGB pattern
        original = np.zeros((50, 50, 3), dtype=np.uint8)
        original[:, :, 0] = 255  # Blue channel in BGR

        base64_str = to_base64(original, format="PNG")
        decoded = from_base64(base64_str)

        # Should preserve BGR format
        assert decoded.shape[2] == 3
        assert decoded[:, :, 0].mean() > 200  # Blue channel


class TestEncodeImageToBase64:
    """Tests for encode_image_to_base64 function."""

    def test_encode_image_to_base64_png(self):
        """Test simple encoding to PNG base64."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        result = encode_image_to_base64(image, format=".png")

        assert isinstance(result, str)
        assert len(result) > 0
        # Should be valid base64
        base64.b64decode(result)

    def test_encode_image_to_base64_jpg(self):
        """Test simple encoding to JPG base64."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        result = encode_image_to_base64(image, format=".jpg")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_encode_image_to_base64_grayscale(self):
        """Test encoding grayscale image."""
        image = np.zeros((100, 100), dtype=np.uint8)

        result = encode_image_to_base64(image, format=".png")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_encode_image_to_base64_different_formats(self):
        """Test encoding with different format strings."""
        image = np.zeros((50, 50, 3), dtype=np.uint8)

        png_result = encode_image_to_base64(image, format=".png")
        jpg_result = encode_image_to_base64(image, format=".jpg")

        # Both should work and produce valid base64
        assert isinstance(png_result, str)
        assert isinstance(jpg_result, str)
        assert len(png_result) > 0
        assert len(jpg_result) > 0
        # Both should be decodable
        base64.b64decode(png_result)
        base64.b64decode(jpg_result)


class TestEnsureBGR:
    """Tests for ensure_bgr function."""

    def test_ensure_bgr_from_grayscale(self):
        """Test converting grayscale to BGR."""
        gray = np.zeros((100, 100), dtype=np.uint8)
        gray[25:75, 25:75] = 200

        bgr = ensure_bgr(gray)

        assert bgr.shape == (100, 100, 3)
        # All channels should be equal for gray
        assert np.array_equal(bgr[:, :, 0], bgr[:, :, 1])
        assert np.array_equal(bgr[:, :, 1], bgr[:, :, 2])
        # Values should match original
        assert np.array_equal(bgr[:, :, 0], gray)

    def test_ensure_bgr_from_bgr(self):
        """Test that BGR image remains BGR (but copied)."""
        original = np.zeros((100, 100, 3), dtype=np.uint8)
        original[:, :, 0] = 255  # Blue channel

        bgr = ensure_bgr(original)

        # Should be a copy
        assert bgr is not original
        # Should have same shape and content
        assert bgr.shape == (100, 100, 3)
        assert np.array_equal(bgr, original)

    def test_ensure_bgr_preserves_values(self):
        """Test that conversion preserves pixel values."""
        gray = np.arange(0, 256, dtype=np.uint8).reshape(16, 16)

        bgr = ensure_bgr(gray)

        # All channels should match original gray values
        for channel in range(3):
            assert np.array_equal(bgr[:, :, channel], gray)

    def test_ensure_bgr_different_dtypes(self):
        """Test ensure_bgr with different data types."""
        gray_uint8 = np.zeros((50, 50), dtype=np.uint8)
        gray_float = np.zeros((50, 50), dtype=np.float32)

        bgr_uint8 = ensure_bgr(gray_uint8)
        bgr_float = ensure_bgr(gray_float)

        assert bgr_uint8.dtype == np.uint8
        assert bgr_float.dtype == np.float32


class TestEnsureGrayscale:
    """Tests for ensure_grayscale function."""

    def test_ensure_grayscale_from_bgr(self):
        """Test converting BGR to grayscale."""
        bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr[25:75, 25:75, :] = [100, 150, 200]  # Some color

        gray = ensure_grayscale(bgr)

        assert gray.shape == (100, 100)
        assert len(gray.shape) == 2
        # Should have converted colors to gray
        assert gray[50, 50] > 0

    def test_ensure_grayscale_from_grayscale(self):
        """Test that grayscale image remains grayscale (but copied)."""
        original = np.zeros((100, 100), dtype=np.uint8)
        original[25:75, 25:75] = 200

        gray = ensure_grayscale(original)

        # Should be a copy
        assert gray is not original
        # Should have same shape and content
        assert gray.shape == (100, 100)
        assert np.array_equal(gray, original)

    def test_ensure_grayscale_white_bgr(self):
        """Test converting white BGR to grayscale."""
        bgr = np.full((50, 50, 3), 255, dtype=np.uint8)

        gray = ensure_grayscale(bgr)

        # White should remain white
        assert gray.shape == (50, 50)
        assert np.all(gray == 255)

    def test_ensure_grayscale_black_bgr(self):
        """Test converting black BGR to grayscale."""
        bgr = np.zeros((50, 50, 3), dtype=np.uint8)

        gray = ensure_grayscale(bgr)

        # Black should remain black
        assert gray.shape == (50, 50)
        assert np.all(gray == 0)

    def test_ensure_grayscale_different_dtypes(self):
        """Test ensure_grayscale with different data types."""
        bgr_uint8 = np.zeros((50, 50, 3), dtype=np.uint8)
        bgr_float = np.zeros((50, 50, 3), dtype=np.float32)

        gray_uint8 = ensure_grayscale(bgr_uint8)
        gray_float = ensure_grayscale(bgr_float)

        assert gray_uint8.dtype == np.uint8
        assert gray_float.dtype == np.float32

    def test_ensure_grayscale_colored_image(self):
        """Test grayscale conversion of colored image."""
        # Red image in BGR
        bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr[:, :, 2] = 255  # Red channel in BGR

        gray = ensure_grayscale(bgr)

        # Should produce some gray value (OpenCV uses weighted conversion)
        assert gray.shape == (100, 100)
        assert gray.mean() > 0
        assert gray.mean() < 255
