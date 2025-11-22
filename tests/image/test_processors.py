"""
Tests for core.image.processors module.

Tests image processing operations including thumbnail creation and resizing.
"""

import base64

import cv2
import numpy as np

from image.processors import create_thumbnail, resize_image


class TestCreateThumbnail:
    """Tests for create_thumbnail function."""

    def test_create_thumbnail_basic(self):
        """Test basic thumbnail creation with aspect ratio maintenance."""
        # Create 640x480 test image
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(image, (100, 100), (300, 300), (255, 255, 255), -1)

        thumbnail, base64_str = create_thumbnail(image, width=320)

        # Check thumbnail dimensions (should maintain aspect ratio)
        assert thumbnail.shape[1] == 320  # width
        assert thumbnail.shape[0] == 240  # height (480 * 320/640)
        assert thumbnail.shape[2] == 3  # channels

        # Check base64 output
        assert isinstance(base64_str, str)
        assert len(base64_str) > 0
        # Verify it's valid base64
        base64.b64decode(base64_str)

    def test_create_thumbnail_maintain_aspect_true(self):
        """Test thumbnail with aspect ratio maintained."""
        # 1000x500 image (2:1 ratio)
        image = np.zeros((500, 1000, 3), dtype=np.uint8)

        thumbnail, _ = create_thumbnail(image, width=200, maintain_aspect=True)

        # Should be 200x100 (maintaining 2:1 ratio)
        assert thumbnail.shape[1] == 200
        assert thumbnail.shape[0] == 100

    def test_create_thumbnail_maintain_aspect_false(self):
        """Test thumbnail without aspect ratio (square)."""
        # 1000x500 image (2:1 ratio)
        image = np.zeros((500, 1000, 3), dtype=np.uint8)

        thumbnail, _ = create_thumbnail(image, width=200, maintain_aspect=False)

        # Should be 200x200 (square, aspect not maintained)
        assert thumbnail.shape[1] == 200
        assert thumbnail.shape[0] == 200

    def test_create_thumbnail_custom_width(self):
        """Test thumbnail creation with custom width."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        thumbnail, _ = create_thumbnail(image, width=160)

        assert thumbnail.shape[1] == 160
        assert thumbnail.shape[0] == 120  # 480 * 160/640

    def test_create_thumbnail_upscaling(self):
        """Test thumbnail creation when target is larger than original."""
        # Small 100x100 image
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Request 320 width thumbnail
        thumbnail, _ = create_thumbnail(image, width=320)

        # Should upscale to 320x320
        assert thumbnail.shape[1] == 320
        assert thumbnail.shape[0] == 320

    def test_create_thumbnail_grayscale(self):
        """Test thumbnail creation from grayscale image."""
        # Grayscale image (2D)
        image = np.zeros((480, 640), dtype=np.uint8)

        thumbnail, base64_str = create_thumbnail(image, width=320)

        # Should handle grayscale
        assert thumbnail.shape[1] == 320
        assert thumbnail.shape[0] == 240
        assert len(thumbnail.shape) == 2  # Still grayscale

    def test_create_thumbnail_very_tall_image(self):
        """Test thumbnail of portrait-oriented image."""
        # Tall image 200x1000
        image = np.zeros((1000, 200, 3), dtype=np.uint8)

        thumbnail, _ = create_thumbnail(image, width=100)

        # Should be 100x500 (maintaining 1:5 aspect)
        assert thumbnail.shape[1] == 100
        assert thumbnail.shape[0] == 500

    def test_create_thumbnail_very_wide_image(self):
        """Test thumbnail of landscape-oriented image."""
        # Wide image 1000x200
        image = np.zeros((200, 1000, 3), dtype=np.uint8)

        thumbnail, _ = create_thumbnail(image, width=500)

        # Should be 500x100 (maintaining 5:1 aspect)
        assert thumbnail.shape[1] == 500
        assert thumbnail.shape[0] == 100

    def test_create_thumbnail_single_pixel(self):
        """Test thumbnail of 1x1 image."""
        image = np.zeros((1, 1, 3), dtype=np.uint8)

        thumbnail, base64_str = create_thumbnail(image, width=10)

        # Should upscale to 10x10
        assert thumbnail.shape[1] == 10
        assert thumbnail.shape[0] == 10
        assert isinstance(base64_str, str)

    def test_create_thumbnail_content_preservation(self):
        """Test that thumbnail preserves image content (scaled)."""
        # Create image with white square
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(image, (200, 100), (400, 300), (255, 255, 255), -1)

        thumbnail, _ = create_thumbnail(image, width=320)

        # Check that thumbnail still has white pixels (scaled)
        assert thumbnail.max() > 200  # Should have bright pixels


class TestResizeImage:
    """Tests for resize_image function."""

    def test_resize_by_width_only(self):
        """Test resize with width specified, height auto-calculated."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        resized = resize_image(image, width=320)

        assert resized.shape[1] == 320
        assert resized.shape[0] == 240  # Maintains aspect ratio

    def test_resize_by_height_only(self):
        """Test resize with height specified, width auto-calculated."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        resized = resize_image(image, height=240)

        assert resized.shape[0] == 240
        assert resized.shape[1] == 320  # Maintains aspect ratio

    def test_resize_both_width_and_height(self):
        """Test resize with both dimensions specified."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        resized = resize_image(image, width=400, height=300)

        assert resized.shape[1] == 400
        assert resized.shape[0] == 300

    def test_resize_max_dimension_width_larger(self):
        """Test resize with max_dimension when width is larger."""
        # Wide image 640x480
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        resized = resize_image(image, max_dimension=320)

        # Should scale to fit within 320, width is limiting factor
        assert resized.shape[1] == 320
        assert resized.shape[0] == 240

    def test_resize_max_dimension_height_larger(self):
        """Test resize with max_dimension when height is larger."""
        # Tall image 480x640
        image = np.zeros((640, 480, 3), dtype=np.uint8)

        resized = resize_image(image, max_dimension=320)

        # Should scale to fit within 320, height is limiting factor
        assert resized.shape[0] == 320
        assert resized.shape[1] == 240

    def test_resize_max_dimension_no_scaling_needed(self):
        """Test max_dimension when image is already smaller."""
        # Small image 200x150
        image = np.zeros((150, 200, 3), dtype=np.uint8)

        resized = resize_image(image, max_dimension=500)

        # Should return original size (no upscaling)
        assert resized.shape[0] == 150
        assert resized.shape[1] == 200
        # Should be same object
        assert resized is image

    def test_resize_no_parameters(self):
        """Test resize with no parameters returns original."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        resized = resize_image(image)

        # Should return original unchanged
        assert resized is image
        assert resized.shape == (480, 640, 3)

    def test_resize_grayscale_image(self):
        """Test resizing grayscale image."""
        image = np.zeros((480, 640), dtype=np.uint8)

        resized = resize_image(image, width=320)

        assert resized.shape[1] == 320
        assert resized.shape[0] == 240
        assert len(resized.shape) == 2  # Still grayscale

    def test_resize_very_small_image(self):
        """Test resizing very small image."""
        image = np.zeros((10, 20, 3), dtype=np.uint8)

        resized = resize_image(image, width=100)

        assert resized.shape[1] == 100
        assert resized.shape[0] == 50

    def test_resize_square_image(self):
        """Test resizing square image."""
        image = np.zeros((500, 500, 3), dtype=np.uint8)

        resized = resize_image(image, width=250)

        # Should maintain square aspect
        assert resized.shape[1] == 250
        assert resized.shape[0] == 250

    def test_resize_max_dimension_square(self):
        """Test max_dimension with square image."""
        image = np.zeros((500, 500, 3), dtype=np.uint8)

        resized = resize_image(image, max_dimension=250)

        # Should scale to 250x250
        assert resized.shape[1] == 250
        assert resized.shape[0] == 250

    def test_resize_preserve_dtype(self):
        """Test that resize preserves data type."""
        # uint8 image
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        resized = resize_image(image, width=320)
        assert resized.dtype == np.uint8

        # float32 image
        image_float = image.astype(np.float32)
        resized_float = resize_image(image_float, width=320)
        assert resized_float.dtype == np.float32

    def test_resize_content_preservation(self):
        """Test that resize preserves image content."""
        # Create image with pattern
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(image, (200, 100), (400, 300), (255, 255, 255), -1)

        resized = resize_image(image, width=320)

        # Should still have white pixels (scaled)
        assert resized.max() > 200

    def test_resize_downscale_large_factor(self):
        """Test resizing with large downscale factor."""
        # Large image
        image = np.zeros((2000, 3000, 3), dtype=np.uint8)

        resized = resize_image(image, max_dimension=100)

        # Should fit within 100x100
        assert max(resized.shape[:2]) == 100
        assert min(resized.shape[:2]) <= 100
