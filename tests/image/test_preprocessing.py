"""
Tests for image.preprocessing module.

Tests all preprocessing operations and the pipeline orchestrator.
"""

import cv2
import numpy as np
import pytest

from image.preprocessing import (
    BilateralFilterOperation,
    BrightnessContrastOperation,
    CLAHEOperation,
    GaussianBlurOperation,
    GrayscaleOperation,
    HistogramEqualizationOperation,
    MedianBlurOperation,
    MorphologyOperation,
    PreprocessingPipeline,
    SharpeningOperation,
    ThresholdOperation,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def color_image():
    """Create a test color image (BGR)."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    # Add some features for testing
    cv2.rectangle(image, (20, 20), (80, 80), (255, 128, 64), -1)  # Filled rectangle
    cv2.circle(image, (50, 50), 20, (0, 255, 0), -1)  # Green circle
    return image


@pytest.fixture
def grayscale_image():
    """Create a test grayscale image."""
    image = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (80, 80), 200, -1)
    cv2.circle(image, (50, 50), 20, 100, -1)
    return image


@pytest.fixture
def noisy_image():
    """Create a noisy grayscale image for blur testing."""
    np.random.seed(42)
    image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    return image


# =============================================================================
# Grayscale Operation Tests
# =============================================================================


class TestGrayscaleOperation:
    """Tests for GrayscaleOperation."""

    def test_converts_color_to_grayscale(self, color_image):
        """Test color to grayscale conversion."""
        op = GrayscaleOperation()
        params = {"grayscale_enabled": True}

        result = op.apply(color_image, params)

        assert len(result.shape) == 2  # Grayscale has 2 dimensions
        assert result.shape == (100, 100)

    def test_grayscale_already_grayscale(self, grayscale_image):
        """Test grayscale operation on already grayscale image."""
        op = GrayscaleOperation()
        params = {"grayscale_enabled": True}

        result = op.apply(grayscale_image, params)

        assert len(result.shape) == 2
        assert result.shape == grayscale_image.shape

    def test_is_enabled(self):
        """Test enable flag check."""
        op = GrayscaleOperation()

        assert op.is_enabled({"grayscale_enabled": True}) is True
        assert op.is_enabled({"grayscale_enabled": False}) is False
        assert op.is_enabled({}) is False

    def test_name(self):
        """Test operation name."""
        op = GrayscaleOperation()
        assert op.name == "grayscale"


# =============================================================================
# Gaussian Blur Operation Tests
# =============================================================================


class TestGaussianBlurOperation:
    """Tests for GaussianBlurOperation."""

    def test_applies_blur(self, grayscale_image):
        """Test Gaussian blur is applied."""
        op = GaussianBlurOperation()
        params = {"gaussian_blur_enabled": True, "gaussian_kernel": 5}

        result = op.apply(grayscale_image, params)

        # Blurred image should be different from original
        assert not np.array_equal(result, grayscale_image)
        assert result.shape == grayscale_image.shape

    def test_even_kernel_becomes_odd(self, grayscale_image):
        """Test even kernel size is converted to odd."""
        op = GaussianBlurOperation()
        params = {"gaussian_blur_enabled": True, "gaussian_kernel": 4}

        # Should not raise error (kernel=4 becomes kernel=5)
        result = op.apply(grayscale_image, params)
        assert result.shape == grayscale_image.shape

    def test_default_kernel_size(self, grayscale_image):
        """Test default kernel size is used."""
        op = GaussianBlurOperation()
        params = {"gaussian_blur_enabled": True}

        result = op.apply(grayscale_image, params)
        assert result.shape == grayscale_image.shape

    def test_is_enabled(self):
        """Test enable flag check."""
        op = GaussianBlurOperation()

        assert op.is_enabled({"gaussian_blur_enabled": True}) is True
        assert op.is_enabled({"gaussian_blur_enabled": False}) is False
        assert op.is_enabled({}) is False


# =============================================================================
# Median Blur Operation Tests
# =============================================================================


class TestMedianBlurOperation:
    """Tests for MedianBlurOperation."""

    def test_applies_median_blur(self, noisy_image):
        """Test median blur reduces salt-pepper noise."""
        op = MedianBlurOperation()
        params = {"median_blur_enabled": True, "median_kernel": 5}

        result = op.apply(noisy_image, params)

        # Result should be different from noisy input
        assert not np.array_equal(result, noisy_image)
        assert result.shape == noisy_image.shape

    def test_even_kernel_becomes_odd(self, grayscale_image):
        """Test even kernel size is converted to odd."""
        op = MedianBlurOperation()
        params = {"median_blur_enabled": True, "median_kernel": 4}

        result = op.apply(grayscale_image, params)
        assert result.shape == grayscale_image.shape


# =============================================================================
# Bilateral Filter Operation Tests
# =============================================================================


class TestBilateralFilterOperation:
    """Tests for BilateralFilterOperation."""

    def test_applies_bilateral_filter(self, color_image):
        """Test bilateral filter is applied."""
        op = BilateralFilterOperation()
        params = {
            "bilateral_enabled": True,
            "bilateral_d": 9,
            "bilateral_sigma_color": 75.0,
            "bilateral_sigma_space": 75.0,
        }

        result = op.apply(color_image, params)

        assert result.shape == color_image.shape

    def test_default_params(self, grayscale_image):
        """Test default parameters are used."""
        op = BilateralFilterOperation()
        params = {"bilateral_enabled": True}

        result = op.apply(grayscale_image, params)
        assert result.shape == grayscale_image.shape


# =============================================================================
# Morphology Operation Tests
# =============================================================================


class TestMorphologyOperation:
    """Tests for MorphologyOperation."""

    def test_close_operation(self, grayscale_image):
        """Test morphological close operation."""
        op = MorphologyOperation()
        params = {
            "morphology_enabled": True,
            "morphology_operation": "close",
            "morphology_kernel": 3,
        }

        result = op.apply(grayscale_image, params)
        assert result.shape == grayscale_image.shape

    def test_open_operation(self, grayscale_image):
        """Test morphological open operation."""
        op = MorphologyOperation()
        params = {
            "morphology_enabled": True,
            "morphology_operation": "open",
            "morphology_kernel": 3,
        }

        result = op.apply(grayscale_image, params)
        assert result.shape == grayscale_image.shape

    def test_erode_operation(self, grayscale_image):
        """Test morphological erode operation."""
        op = MorphologyOperation()
        params = {
            "morphology_enabled": True,
            "morphology_operation": "erode",
            "morphology_kernel": 3,
        }

        result = op.apply(grayscale_image, params)
        assert result.shape == grayscale_image.shape

    def test_dilate_operation(self, grayscale_image):
        """Test morphological dilate operation."""
        op = MorphologyOperation()
        params = {
            "morphology_enabled": True,
            "morphology_operation": "dilate",
            "morphology_kernel": 3,
        }

        result = op.apply(grayscale_image, params)
        assert result.shape == grayscale_image.shape


# =============================================================================
# Threshold Operation Tests
# =============================================================================


class TestThresholdOperation:
    """Tests for ThresholdOperation."""

    def test_binary_threshold(self, grayscale_image):
        """Test binary thresholding."""
        op = ThresholdOperation()
        params = {
            "threshold_enabled": True,
            "threshold_method": "binary",
            "threshold_value": 127,
            "threshold_max_value": 255,
        }

        result = op.apply(grayscale_image, params)

        # Result should be binary (only 0 and 255)
        unique_values = np.unique(result)
        assert all(v in [0, 255] for v in unique_values)

    def test_otsu_threshold(self, grayscale_image):
        """Test Otsu's automatic thresholding."""
        op = ThresholdOperation()
        params = {
            "threshold_enabled": True,
            "threshold_method": "otsu",
            "threshold_max_value": 255,
        }

        result = op.apply(grayscale_image, params)

        # Result should be binary
        unique_values = np.unique(result)
        assert all(v in [0, 255] for v in unique_values)

    def test_adaptive_mean_threshold(self, grayscale_image):
        """Test adaptive mean thresholding."""
        op = ThresholdOperation()
        params = {
            "threshold_enabled": True,
            "threshold_method": "adaptive_mean",
            "threshold_max_value": 255,
            "adaptive_block_size": 11,
            "adaptive_c": 2.0,
        }

        result = op.apply(grayscale_image, params)
        assert result.shape == grayscale_image.shape

    def test_adaptive_gaussian_threshold(self, grayscale_image):
        """Test adaptive Gaussian thresholding."""
        op = ThresholdOperation()
        params = {
            "threshold_enabled": True,
            "threshold_method": "adaptive_gaussian",
            "threshold_max_value": 255,
            "adaptive_block_size": 11,
            "adaptive_c": 2.0,
        }

        result = op.apply(grayscale_image, params)
        assert result.shape == grayscale_image.shape

    def test_threshold_converts_color_to_gray(self, color_image):
        """Test thresholding converts color image to grayscale first."""
        op = ThresholdOperation()
        params = {
            "threshold_enabled": True,
            "threshold_method": "binary",
            "threshold_value": 127,
        }

        result = op.apply(color_image, params)

        # Result should be grayscale (2D)
        assert len(result.shape) == 2

    def test_even_block_size_becomes_odd(self, grayscale_image):
        """Test even adaptive block size is converted to odd."""
        op = ThresholdOperation()
        params = {
            "threshold_enabled": True,
            "threshold_method": "adaptive_mean",
            "adaptive_block_size": 10,  # Even, should become 11
        }

        result = op.apply(grayscale_image, params)
        assert result.shape == grayscale_image.shape


# =============================================================================
# Histogram Equalization Operation Tests
# =============================================================================


class TestHistogramEqualizationOperation:
    """Tests for HistogramEqualizationOperation."""

    def test_equalize_grayscale(self, grayscale_image):
        """Test histogram equalization on grayscale image."""
        op = HistogramEqualizationOperation()
        params = {"hist_equalize_enabled": True}

        result = op.apply(grayscale_image, params)

        assert result.shape == grayscale_image.shape
        # Histogram should be more spread out
        assert result.std() != grayscale_image.std()

    def test_equalize_color(self, color_image):
        """Test histogram equalization on color image (LAB space)."""
        op = HistogramEqualizationOperation()
        params = {"hist_equalize_enabled": True}

        result = op.apply(color_image, params)

        # Should maintain color format
        assert result.shape == color_image.shape
        assert len(result.shape) == 3


# =============================================================================
# CLAHE Operation Tests
# =============================================================================


class TestCLAHEOperation:
    """Tests for CLAHEOperation."""

    def test_clahe_grayscale(self, grayscale_image):
        """Test CLAHE on grayscale image."""
        op = CLAHEOperation()
        params = {
            "clahe_enabled": True,
            "clahe_clip_limit": 2.0,
            "clahe_tile_grid_size": 8,
        }

        result = op.apply(grayscale_image, params)

        assert result.shape == grayscale_image.shape

    def test_clahe_color(self, color_image):
        """Test CLAHE on color image (LAB space)."""
        op = CLAHEOperation()
        params = {
            "clahe_enabled": True,
            "clahe_clip_limit": 2.0,
            "clahe_tile_grid_size": 8,
        }

        result = op.apply(color_image, params)

        assert result.shape == color_image.shape
        assert len(result.shape) == 3


# =============================================================================
# Sharpening Operation Tests
# =============================================================================


class TestSharpeningOperation:
    """Tests for SharpeningOperation."""

    def test_applies_sharpening(self, grayscale_image):
        """Test sharpening is applied."""
        op = SharpeningOperation()
        params = {"sharpen_enabled": True, "sharpen_strength": 1.0}

        result = op.apply(grayscale_image, params)

        assert result.shape == grayscale_image.shape
        # Sharpening should increase edge contrast
        assert not np.array_equal(result, grayscale_image)

    def test_sharpening_strength(self, grayscale_image):
        """Test different sharpening strengths produce different results."""
        op = SharpeningOperation()

        result_weak = op.apply(grayscale_image, {"sharpen_strength": 0.5})
        result_strong = op.apply(grayscale_image, {"sharpen_strength": 2.0})

        # Different strengths should produce different results
        assert not np.array_equal(result_weak, result_strong)


# =============================================================================
# Brightness/Contrast Operation Tests
# =============================================================================


class TestBrightnessContrastOperation:
    """Tests for BrightnessContrastOperation."""

    def test_increase_brightness(self, grayscale_image):
        """Test increasing brightness."""
        op = BrightnessContrastOperation()
        params = {
            "brightness_contrast_enabled": True,
            "brightness": 50,
            "contrast": 1.0,
        }

        result = op.apply(grayscale_image, params)

        # Mean brightness should increase
        assert result.mean() > grayscale_image.mean()

    def test_decrease_brightness(self):
        """Test decreasing brightness."""
        # Create bright image (all 200)
        bright_image = np.full((100, 100), 200, dtype=np.uint8)

        op = BrightnessContrastOperation()
        params = {
            "brightness_contrast_enabled": True,
            "brightness": -50,
            "contrast": 1.0,
        }

        result = op.apply(bright_image, params)

        # Mean brightness should decrease
        assert result.mean() < bright_image.mean()

    def test_increase_contrast(self, grayscale_image):
        """Test increasing contrast."""
        op = BrightnessContrastOperation()
        params = {
            "brightness_contrast_enabled": True,
            "brightness": 0,
            "contrast": 2.0,
        }

        result = op.apply(grayscale_image, params)
        assert result.shape == grayscale_image.shape


# =============================================================================
# Pipeline Tests
# =============================================================================


class TestPreprocessingPipeline:
    """Tests for PreprocessingPipeline."""

    def test_empty_params_returns_copy(self):
        """Test that empty params returns a copy of the image."""
        # Create image with non-zero values
        original = np.full((100, 100, 3), 128, dtype=np.uint8)
        pipeline = PreprocessingPipeline()

        result, applied = pipeline.process(original, {})

        # Should return copy of original
        assert np.array_equal(result, original)
        assert applied == []
        # Should be a copy, not the same object
        result[0, 0] = [255, 255, 255]
        assert not np.array_equal(result, original)

    def test_single_operation(self, color_image):
        """Test pipeline with single operation."""
        pipeline = PreprocessingPipeline()
        params = {"grayscale_enabled": True}

        result, applied = pipeline.process(color_image, params)

        assert len(result.shape) == 2  # Grayscale
        assert applied == ["grayscale"]

    def test_multiple_operations(self, color_image):
        """Test pipeline with multiple operations."""
        pipeline = PreprocessingPipeline()
        params = {
            "grayscale_enabled": True,
            "gaussian_blur_enabled": True,
            "gaussian_kernel": 5,
            "threshold_enabled": True,
            "threshold_method": "otsu",
        }

        result, applied = pipeline.process(color_image, params)

        assert applied == ["grayscale", "gaussian_blur", "threshold"]
        # Result should be binary
        unique_values = np.unique(result)
        assert all(v in [0, 255] for v in unique_values)

    def test_operation_order_fixed(self, color_image):
        """Test that operations are applied in fixed order regardless of params order."""
        pipeline = PreprocessingPipeline()

        # Threshold before grayscale in params, but grayscale should be applied first
        params = {
            "threshold_enabled": True,
            "threshold_method": "binary",
            "grayscale_enabled": True,
        }

        result, applied = pipeline.process(color_image, params)

        # Grayscale should be first
        assert applied == ["grayscale", "threshold"]

    def test_none_params(self, color_image):
        """Test pipeline with None params."""
        pipeline = PreprocessingPipeline()

        result, applied = pipeline.process(color_image, None)

        assert np.array_equal(result, color_image)
        assert applied == []

    def test_full_pipeline(self, color_image):
        """Test pipeline with all operations enabled."""
        pipeline = PreprocessingPipeline()
        params = {
            "grayscale_enabled": True,
            "gaussian_blur_enabled": True,
            "gaussian_kernel": 3,
            "median_blur_enabled": True,
            "median_kernel": 3,
            "bilateral_enabled": True,
            "morphology_enabled": True,
            "morphology_operation": "close",
            "threshold_enabled": True,
            "threshold_method": "otsu",
            "hist_equalize_enabled": False,  # Skip this one
            "clahe_enabled": False,  # Skip this one
            "sharpen_enabled": True,
            "sharpen_strength": 0.5,
            "brightness_contrast_enabled": True,
            "brightness": 10,
            "contrast": 1.1,
        }

        result, applied = pipeline.process(color_image, params)

        # New pipeline order (optimized for thresholding/segmentation):
        # 1. brightness/contrast, 2. grayscale, 3. HEQ/CLAHE, 4. blur, 5. sharpen, 6. threshold, 7. morphology
        expected_ops = [
            "brightness_contrast",
            "grayscale",
            "gaussian_blur",
            "median_blur",
            "bilateral_filter",
            "sharpening",
            "threshold",
            "morphology",
        ]
        assert applied == expected_ops

    def test_get_available_operations(self):
        """Test getting list of available operations."""
        pipeline = PreprocessingPipeline()

        ops = pipeline.get_available_operations()

        assert "grayscale" in ops
        assert "gaussian_blur" in ops
        assert "threshold" in ops
        assert len(ops) == 10  # All 10 operations
