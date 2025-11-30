"""
Image preprocessing operations with pipeline architecture.

This module provides a modular preprocessing system using the Strategy pattern.
Each operation is a separate class that can be enabled/disabled via parameters.
Operations are applied in a fixed sequence for deterministic results.

Pipeline order:
1. Grayscale conversion
2. Gaussian blur
3. Median blur
4. Bilateral filter
5. Morphological operations
6. Thresholding
7. Histogram equalization
8. CLAHE
9. Sharpening
10. Brightness/Contrast adjustment

Usage:
    pipeline = PreprocessingPipeline()
    processed_image, applied_ops = pipeline.process(image, params_dict)
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from image.converters import ensure_grayscale

logger = logging.getLogger(__name__)


class PreprocessOperation(ABC):
    """Abstract base class for preprocessing operations."""

    @abstractmethod
    def apply(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Apply operation to image.

        Args:
            image: Input image (BGR or grayscale)
            params: Operation parameters

        Returns:
            Processed image
        """
        pass

    @abstractmethod
    def is_enabled(self, params: Dict[str, Any]) -> bool:
        """
        Check if operation is enabled in params.

        Args:
            params: Operation parameters

        Returns:
            True if operation should be applied
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Operation name for logging and tracking."""
        pass


class GrayscaleOperation(PreprocessOperation):
    """Convert image to grayscale."""

    def apply(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        return ensure_grayscale(image)

    def is_enabled(self, params: Dict[str, Any]) -> bool:
        return params.get("grayscale_enabled", False)

    @property
    def name(self) -> str:
        return "grayscale"


class GaussianBlurOperation(PreprocessOperation):
    """Apply Gaussian blur for noise reduction."""

    def apply(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        kernel_size = int(params.get("gaussian_kernel", 5))
        # Ensure odd kernel size
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def is_enabled(self, params: Dict[str, Any]) -> bool:
        return params.get("gaussian_blur_enabled", False)

    @property
    def name(self) -> str:
        return "gaussian_blur"


class MedianBlurOperation(PreprocessOperation):
    """Apply median blur for salt-pepper noise removal."""

    def apply(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        kernel_size = int(params.get("median_kernel", 5))
        # Ensure odd kernel size
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.medianBlur(image, kernel_size)

    def is_enabled(self, params: Dict[str, Any]) -> bool:
        return params.get("median_blur_enabled", False)

    @property
    def name(self) -> str:
        return "median_blur"


class BilateralFilterOperation(PreprocessOperation):
    """Apply bilateral filter for edge-preserving smoothing."""

    def apply(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        d = int(params.get("bilateral_d", 9))
        sigma_color = float(params.get("bilateral_sigma_color", 75.0))
        sigma_space = float(params.get("bilateral_sigma_space", 75.0))
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    def is_enabled(self, params: Dict[str, Any]) -> bool:
        return params.get("bilateral_enabled", False)

    @property
    def name(self) -> str:
        return "bilateral_filter"


class MorphologyOperation(PreprocessOperation):
    """Apply morphological operations (erode, dilate, open, close)."""

    OPERATIONS = {
        "erode": cv2.MORPH_ERODE,
        "dilate": cv2.MORPH_DILATE,
        "open": cv2.MORPH_OPEN,
        "close": cv2.MORPH_CLOSE,
    }

    def apply(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        operation = params.get("morphology_operation", "close")
        kernel_size = int(params.get("morphology_kernel", 3))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        morph_type = self.OPERATIONS.get(operation, cv2.MORPH_CLOSE)
        return cv2.morphologyEx(image, morph_type, kernel)

    def is_enabled(self, params: Dict[str, Any]) -> bool:
        return params.get("morphology_enabled", False)

    @property
    def name(self) -> str:
        return "morphology"


class ThresholdOperation(PreprocessOperation):
    """Apply thresholding (binary, Otsu, adaptive)."""

    def apply(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        # Ensure grayscale for thresholding
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        method = params.get("threshold_method", "binary")
        max_value = int(params.get("threshold_max_value", 255))

        if method == "binary":
            thresh_val = int(params.get("threshold_value", 127))
            _, result = cv2.threshold(gray, thresh_val, max_value, cv2.THRESH_BINARY)

        elif method == "otsu":
            _, result = cv2.threshold(gray, 0, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        elif method == "adaptive_mean":
            block_size = int(params.get("adaptive_block_size", 11))
            # Ensure odd block size
            if block_size % 2 == 0:
                block_size += 1
            c = float(params.get("adaptive_c", 2.0))
            result = cv2.adaptiveThreshold(
                gray, max_value, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c
            )

        elif method == "adaptive_gaussian":
            block_size = int(params.get("adaptive_block_size", 11))
            # Ensure odd block size
            if block_size % 2 == 0:
                block_size += 1
            c = float(params.get("adaptive_c", 2.0))
            result = cv2.adaptiveThreshold(
                gray, max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
            )
        else:
            # Unknown method, return grayscale
            result = gray

        return result

    def is_enabled(self, params: Dict[str, Any]) -> bool:
        return params.get("threshold_enabled", False)

    @property
    def name(self) -> str:
        return "threshold"


class HistogramEqualizationOperation(PreprocessOperation):
    """Apply histogram equalization for contrast enhancement."""

    def apply(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        if len(image.shape) == 3:
            # Color image: equalize L channel in LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            l_ch = cv2.equalizeHist(l_ch)
            lab = cv2.merge([l_ch, a_ch, b_ch])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale: direct equalization
            return cv2.equalizeHist(image)

    def is_enabled(self, params: Dict[str, Any]) -> bool:
        return params.get("hist_equalize_enabled", False)

    @property
    def name(self) -> str:
        return "histogram_equalization"


class CLAHEOperation(PreprocessOperation):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""

    def apply(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        clip_limit = float(params.get("clahe_clip_limit", 2.0))
        tile_size = int(params.get("clahe_tile_grid_size", 8))

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))

        if len(image.shape) == 3:
            # Color image: apply CLAHE to L channel in LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            l_ch = clahe.apply(l_ch)
            lab = cv2.merge([l_ch, a_ch, b_ch])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale: direct application
            return clahe.apply(image)

    def is_enabled(self, params: Dict[str, Any]) -> bool:
        return params.get("clahe_enabled", False)

    @property
    def name(self) -> str:
        return "clahe"


class SharpeningOperation(PreprocessOperation):
    """Apply unsharp mask sharpening."""

    def apply(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        strength = float(params.get("sharpen_strength", 1.0))
        # Unsharp mask: result = original + strength * (original - blurred)
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        return cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)

    def is_enabled(self, params: Dict[str, Any]) -> bool:
        return params.get("sharpen_enabled", False)

    @property
    def name(self) -> str:
        return "sharpening"


class BrightnessContrastOperation(PreprocessOperation):
    """Adjust brightness and contrast."""

    def apply(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        brightness = int(params.get("brightness", 0))
        contrast = float(params.get("contrast", 1.0))
        # alpha = contrast, beta = brightness
        return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

    def is_enabled(self, params: Dict[str, Any]) -> bool:
        return params.get("brightness_contrast_enabled", False)

    @property
    def name(self) -> str:
        return "brightness_contrast"


class PreprocessingPipeline:
    """
    Pipeline for applying preprocessing operations in sequence.

    Operations are applied in a fixed order that follows best practices
    for image preprocessing in computer vision applications.
    """

    def __init__(self):
        """Initialize pipeline with all operations in fixed order."""
        self.operations: List[PreprocessOperation] = [
            GrayscaleOperation(),  # 1. Grayscale conversion
            GaussianBlurOperation(),  # 2. Gaussian blur
            MedianBlurOperation(),  # 3. Median blur
            BilateralFilterOperation(),  # 4. Bilateral filter
            MorphologyOperation(),  # 5. Morphological ops
            ThresholdOperation(),  # 6. Thresholding
            HistogramEqualizationOperation(),  # 7. Histogram equalization
            CLAHEOperation(),  # 8. CLAHE
            SharpeningOperation(),  # 9. Sharpening
            BrightnessContrastOperation(),  # 10. Brightness/Contrast
        ]

    def process(self, image: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
        """
        Apply enabled operations to image.

        Args:
            image: Input image (BGR or grayscale)
            params: Dictionary of operation parameters

        Returns:
            Tuple of (processed_image, list_of_applied_operation_names)
        """
        if params is None:
            params = {}

        result = image.copy()
        applied: List[str] = []

        for op in self.operations:
            if op.is_enabled(params):
                try:
                    result = op.apply(result, params)
                    applied.append(op.name)
                    logger.debug(f"Applied preprocessing: {op.name}")
                except Exception as e:
                    logger.error(f"Failed to apply {op.name}: {e}")
                    raise

        if not applied:
            logger.debug("No preprocessing operations applied")

        return result, applied

    def get_available_operations(self) -> List[str]:
        """Get list of available operation names."""
        return [op.name for op in self.operations]
