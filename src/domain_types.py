"""
Central types module for machine vision system.

This module consolidates all fundamental types, enums, and constants used throughout
the system. It has no dependencies on other project modules (only stdlib and Pydantic).

Contents:
- Base Models: Point, ROI (geometric primitives)
- Enums: All enumeration types (EdgeMethod, ColorMethod, CameraType, etc.)
- Constants: Configuration values and magic numbers organized by domain

IMPORTANT: This module must NOT import from schemas, core, services, managers,
algorithms, or api to avoid circular dependencies.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

# ==============================================================================
# Base Data Models
# ==============================================================================


class Point(BaseModel):
    """2D Point"""

    x: float
    y: float


class ROI(BaseModel):
    """
    Region of Interest - unified implementation.

    Represents a rectangular region in an image with utility methods for
    geometric operations, validation, and conversions.
    """

    x: int = Field(..., ge=0, description="X coordinate")
    y: int = Field(..., ge=0, description="Y coordinate")
    width: int = Field(..., gt=0, description="Width")
    height: int = Field(..., gt=0, description="Height")

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary for service layer compatibility."""
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ROI":
        """Create ROI from dictionary."""
        return cls(
            x=int(data.get("x", 0)),
            y=int(data.get("y", 0)),
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
        )

    @classmethod
    def from_points(cls, x1: int, y1: int, x2: int, y2: int) -> "ROI":
        """Create ROI from two corner points."""
        return cls(x=min(x1, x2), y=min(y1, y2), width=abs(x2 - x1), height=abs(y2 - y1))

    @property
    def x2(self) -> int:
        """Get right edge coordinate."""
        return self.x + self.width

    @property
    def y2(self) -> int:
        """Get bottom edge coordinate."""
        return self.y + self.height

    @property
    def center_point(self) -> tuple[int, int]:
        """Get center point of ROI as (x, y) tuple."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area_pixels(self) -> int:
        """Get area of ROI in pixels."""
        return self.width * self.height

    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is inside ROI."""
        return self.x <= x < self.x2 and self.y <= y < self.y2

    def intersects(self, other: "ROI") -> bool:
        """Check if this ROI intersects with another."""
        return not (
            self.x2 <= other.x or other.x2 <= self.x or self.y2 <= other.y or other.y2 <= self.y
        )

    def intersection(self, other: "ROI") -> Optional["ROI"]:
        """Get intersection with another ROI, or None if no intersection."""
        if not self.intersects(other):
            return None

        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)

        return ROI.from_points(x1, y1, x2, y2)

    def union(self, other: "ROI") -> "ROI":
        """Get union with another ROI (smallest rectangle containing both)."""
        x1 = min(self.x, other.x)
        y1 = min(self.y, other.y)
        x2 = max(self.x2, other.x2)
        y2 = max(self.y2, other.y2)

        return ROI.from_points(x1, y1, x2, y2)

    def scale(self, factor: float, from_center: bool = False) -> "ROI":
        """
        Scale ROI by factor.

        Args:
            factor: Scale factor (e.g., 1.5 for 150%)
            from_center: If True, scale from center; if False, scale from top-left

        Returns:
            New scaled ROI
        """
        new_width = int(self.width * factor)
        new_height = int(self.height * factor)

        if from_center:
            # Scale from center
            cx, cy = self.center_point
            new_x = cx - new_width // 2
            new_y = cy - new_height // 2
        else:
            # Scale from top-left
            new_x = self.x
            new_y = self.y

        return ROI(x=new_x, y=new_y, width=new_width, height=new_height)

    def expand(self, pixels: int) -> "ROI":
        """Expand ROI by pixels in all directions."""
        return ROI(
            x=self.x - pixels,
            y=self.y - pixels,
            width=self.width + 2 * pixels,
            height=self.height + 2 * pixels,
        )

    def clip(self, image_width: int, image_height: int) -> "ROI":
        """
        Clip ROI to image bounds.

        Args:
            image_width: Maximum width (image width)
            image_height: Maximum height (image height)

        Returns:
            Clipped ROI that fits within image bounds
        """
        x = max(0, min(self.x, image_width))
        y = max(0, min(self.y, image_height))
        x2 = max(0, min(self.x2, image_width))
        y2 = max(0, min(self.y2, image_height))

        return ROI.from_points(x, y, x2, y2)

    def is_valid(
        self, image_width: Optional[int] = None, image_height: Optional[int] = None
    ) -> bool:
        """
        Check if ROI is valid.

        Args:
            image_width: Optional image width for bounds checking
            image_height: Optional image height for bounds checking

        Returns:
            True if ROI is valid
        """
        # Basic validation (Pydantic already ensures width/height > 0 and x/y >= 0)
        # But we check again for runtime safety
        if self.width <= 0 or self.height <= 0:
            return False

        if self.x < 0 or self.y < 0:
            return False

        # Image bounds validation if provided
        if image_width is not None and self.x2 > image_width:
            return False

        if image_height is not None and self.y2 > image_height:
            return False

        return True

    def validate_with_constraints(
        self,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        min_size: int = 1,
        max_size: Optional[int] = None,
    ) -> None:
        """
        Validate ROI with size constraints.

        Args:
            image_width: Optional image width for bounds checking
            image_height: Optional image height for bounds checking
            min_size: Minimum width/height (default: 1)
            max_size: Optional maximum width/height

        Raises:
            ValueError: If ROI is invalid with descriptive error message
        """
        # Check basic validity (coordinates and dimensions)
        if self.width < min_size or self.height < min_size:
            raise ValueError(f"ROI too small: {self.width}x{self.height} (min: {min_size})")

        if max_size and (self.width > max_size or self.height > max_size):
            raise ValueError(f"ROI too large: {self.width}x{self.height} (max: {max_size})")

        if self.x < 0 or self.y < 0:
            raise ValueError(f"ROI has negative coordinates: ({self.x}, {self.y})")

        # Image bounds validation if provided
        if image_width is not None or image_height is not None:
            if not self.is_valid(image_width, image_height):
                raise ValueError(
                    f"ROI {self.to_dict()} exceeds image bounds {image_width}x{image_height}"
                )


# ==============================================================================
# Enumerations
# ==============================================================================


# Template matching enums
class TemplateMethod(str, Enum):
    """Template matching methods."""

    TM_CCOEFF = "TM_CCOEFF"
    TM_CCOEFF_NORMED = "TM_CCOEFF_NORMED"
    TM_CCORR = "TM_CCORR"
    TM_CCORR_NORMED = "TM_CCORR_NORMED"
    TM_SQDIFF = "TM_SQDIFF"
    TM_SQDIFF_NORMED = "TM_SQDIFF_NORMED"


# Inspection result enums
class InspectionResult(str, Enum):
    """Inspection result status."""

    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"


# Edge detection enums
class EdgeMethod(str, Enum):
    """Available edge detection methods."""

    CANNY = "canny"
    SOBEL = "sobel"
    LAPLACIAN = "laplacian"
    PREWITT = "prewitt"
    SCHARR = "scharr"
    MORPHOLOGICAL_GRADIENT = "morphological_gradient"


# Color detection enums
class ColorMethod(str, Enum):
    """Color detection methods."""

    HISTOGRAM = "histogram"


# ArUco marker enums
class ArucoDict(str, Enum):
    """Available ArUco dictionary types."""

    DICT_4X4_50 = "DICT_4X4_50"
    DICT_4X4_100 = "DICT_4X4_100"
    DICT_4X4_250 = "DICT_4X4_250"
    DICT_4X4_1000 = "DICT_4X4_1000"
    DICT_5X5_50 = "DICT_5X5_50"
    DICT_5X5_100 = "DICT_5X5_100"
    DICT_5X5_250 = "DICT_5X5_250"
    DICT_5X5_1000 = "DICT_5X5_1000"
    DICT_6X6_50 = "DICT_6X6_50"
    DICT_6X6_100 = "DICT_6X6_100"
    DICT_6X6_250 = "DICT_6X6_250"
    DICT_6X6_1000 = "DICT_6X6_1000"
    DICT_7X7_50 = "DICT_7X7_50"
    DICT_7X7_100 = "DICT_7X7_100"
    DICT_7X7_250 = "DICT_7X7_250"
    DICT_7X7_1000 = "DICT_7X7_1000"
    DICT_ARUCO_ORIGINAL = "DICT_ARUCO_ORIGINAL"


class ArucoDetectionMode(str, Enum):
    """
    ArUco detection modes (internal use only by ArucoDetector).

    Note: API users should use separate endpoints instead:
    - /api/vision/aruco-detect (MARKERS mode)
    - /api/vision/aruco-reference (SINGLE/PLANE modes)
    """

    MARKERS = "markers"  # Detect all markers (no reference frame)
    SINGLE = "single"  # Single marker reference with scale
    PLANE = "plane"  # 4-marker plane reference with homography


class ArucoReferenceMode(str, Enum):
    """ArUco reference frame modes (for aruco-reference endpoint)."""

    SINGLE = "single"  # Single marker reference with uniform scaling
    PLANE = "plane"  # 4-marker plane reference with perspective homography


# Rotation detection enums
class RotationMethod(str, Enum):
    """Rotation detection methods."""

    MIN_AREA_RECT = "min_area_rect"
    ELLIPSE_FIT = "ellipse_fit"
    PCA = "pca"


class AngleRange(str, Enum):
    """Angle output range options."""

    RANGE_0_360 = "0_360"  # 0 to 360 degrees
    RANGE_NEG180_180 = "-180_180"  # -180 to +180 degrees
    RANGE_0_180 = "0_180"  # 0 to 180 degrees (symmetric objects)


class AsymmetryOrientation(str, Enum):
    """Orientation direction for asymmetric objects (based on thickness)."""

    DISABLED = "disabled"  # No asymmetry-based orientation
    THICK_TO_THIN = "thick_to_thin"  # Orient angle from thick part to thin part
    THIN_TO_THICK = "thin_to_thick"  # Orient angle from thin part to thick part


# Vision object type enums
class VisionObjectType(str, Enum):
    """Types of vision objects."""

    CAMERA_CAPTURE = "camera_capture"
    EDGE_CONTOUR = "edge_contour"
    TEMPLATE_MATCH = "template_match"
    COLOR_REGION = "color_region"
    ARUCO_MARKER = "aruco_marker"
    ROTATION_ANALYSIS = "rotation_analysis"


# Camera type enums
class CameraType(str, Enum):
    """Camera connection types."""

    USB = "usb"
    IP = "ip"
    FILE = "file"
    TEST = "test"


# ==============================================================================
# Constants
# ==============================================================================


# Image Management Constants
class ImageConstants:
    """Constants related to image storage and processing."""

    # Storage limits
    DEFAULT_MAX_IMAGES = 100
    DEFAULT_MAX_MEMORY_MB = 1000
    MIN_IMAGES = 1
    MAX_IMAGES = 1000

    # Image dimensions
    DEFAULT_IMAGE_WIDTH = 1920
    DEFAULT_IMAGE_HEIGHT = 1080
    MAX_IMAGE_DIMENSION = 4096
    MIN_IMAGE_DIMENSION = 10

    # Thumbnail settings
    DEFAULT_THUMBNAIL_WIDTH = 320
    MIN_THUMBNAIL_WIDTH = 50
    MAX_THUMBNAIL_WIDTH = 2000  # Allow full resolution thumbnails in debug mode
    THUMBNAIL_JPEG_QUALITY = 70

    # Memory management
    MEMORY_CLEANUP_THRESHOLD = 0.9  # Start cleanup at 90% memory usage
    LRU_EVICTION_BATCH_SIZE = 5  # Number of images to evict at once


# Camera Constants
class CameraConstants:
    """Constants related to camera operations."""

    # Camera types
    TEST_CAMERA_ID = "test"
    DEFAULT_CAMERA_ID = "test"

    # Capture settings
    DEFAULT_CAPTURE_TIMEOUT_MS = 5000
    MAX_CAPTURE_TIMEOUT_MS = 30000
    MIN_CAPTURE_TIMEOUT_MS = 100

    # USB camera enumeration
    MAX_USB_CAMERAS_TO_CHECK = 5
    USB_CAMERA_CHECK_TIMEOUT_MS = 1000

    # Stream settings
    MJPEG_FPS = 30
    MJPEG_QUALITY = 85
    STREAM_BUFFER_SIZE = 10
    MAX_CONCURRENT_STREAMS = 3

    # Test image generation
    TEST_IMAGE_WIDTH = 1920
    TEST_IMAGE_HEIGHT = 1080
    TEST_PATTERN_TYPES = ["checkerboard", "gradient", "noise", "solid"]


# Template Constants
class TemplateConstants:
    """Constants related to template matching."""

    # Storage
    DEFAULT_STORAGE_PATH = "data/templates"
    MAX_TEMPLATE_SIZE_MB = 10
    ALLOWED_FORMATS = [".png", ".jpg", ".jpeg"]

    # Matching parameters
    DEFAULT_THRESHOLD = 0.8
    MIN_THRESHOLD = 0.0
    MAX_THRESHOLD = 1.0

    DEFAULT_SCALE_MIN = 0.8
    DEFAULT_SCALE_MAX = 1.2
    SCALE_STEP = 0.05

    # Template limits
    MAX_TEMPLATES = 1000
    MIN_TEMPLATE_SIZE = 10  # pixels
    MAX_TEMPLATE_SIZE = 500  # pixels


class TestImageConstants:
    """Constants related to test image storage."""

    # Storage
    # Development: relative path "data/test_images"
    # Production: absolute path "/var/lib/machine-vision/test_images"
    # Override via MV_TEST_IMAGE_STORAGE_PATH environment variable
    DEFAULT_STORAGE_PATH = "data/test_images"
    MAX_FILE_SIZE_MB = 50
    ALLOWED_FORMATS = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"]

    # Test image limits
    MAX_TEST_IMAGES = 100


# Vision Processing Constants
class VisionConstants:
    """Constants for computer vision operations."""

    # Canny edge detection
    CANNY_LOW_THRESHOLD_DEFAULT = 50
    CANNY_HIGH_THRESHOLD_DEFAULT = 150
    CANNY_LOW_THRESHOLD_MIN = 0
    CANNY_LOW_THRESHOLD_MAX = 500
    CANNY_HIGH_THRESHOLD_MIN = 0
    CANNY_HIGH_THRESHOLD_MAX = 500

    # Morphological operations
    MORPH_KERNEL_SIZE_DEFAULT = 3
    MORPH_KERNEL_SIZE_MIN = 1
    MORPH_KERNEL_SIZE_MAX = 21

    # Contour detection
    MAX_CONTOURS_DEFAULT = 100
    MAX_CONTOURS_LIMIT = 1000
    MIN_CONTOUR_AREA_DEFAULT = 100
    MAX_CONTOUR_AREA_DEFAULT = 100000

    # Preprocessing
    GAUSSIAN_BLUR_SIZE_DEFAULT = 5
    GAUSSIAN_BLUR_SIZE_MIN = 1
    GAUSSIAN_BLUR_SIZE_MAX = 31
    BILATERAL_FILTER_D_DEFAULT = 9
    MEDIAN_BLUR_SIZE_DEFAULT = 5

    # Contour approximation
    CONTOUR_APPROX_EPSILON_FACTOR = 0.02  # 2% of perimeter for polygon approximation

    # Rotation detection
    MIN_POINTS_ROTATION = 3  # Minimum points required for rotation detection
    MIN_POINTS_ELLIPSE_FIT = 5  # Minimum points required for ellipse fitting
    CONFIDENCE_FULL = 1.0  # Full confidence score
    CONFIDENCE_HIGH = 0.9  # High confidence score
    CONFIDENCE_SCALING_FACTOR = 10.0  # Factor for ratio-based confidence scaling


# API Constants
class APIConstants:
    """Constants for API endpoints."""

    # Rate limiting
    REQUESTS_PER_MINUTE = 100

    # Timeouts
    REQUEST_TIMEOUT_SECONDS = 30

    # File uploads
    MAX_UPLOAD_SIZE_MB = 50

    # API versions
    API_VERSION = "v1"


# System Constants
class SystemConstants:
    """Constants for system operations."""

    # Logging
    LOG_LEVEL_DEFAULT = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE_MAX_BYTES = 10 * 1024 * 1024  # 10MB
    LOG_FILE_BACKUP_COUNT = 5

    # Threading
    MAX_WORKER_THREADS = 10
    THREAD_POOL_SIZE = 4

    # File system
    TEMP_DIR = "/tmp/machinevision"
    DATA_DIR = "./data"
    CLEANUP_INTERVAL_SECONDS = 3600  # 1 hour

    # Health checks
    HEALTH_CHECK_INTERVAL_SECONDS = 30
    MEMORY_WARNING_THRESHOLD = 0.8  # 80% memory usage
    DISK_WARNING_THRESHOLD = 0.9  # 90% disk usage


# Color Constants (BGR format for OpenCV)
class Colors:
    """Standard colors for drawing operations (BGR format)."""

    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    YELLOW = (0, 255, 255)
    CYAN = (255, 255, 0)
    MAGENTA = (255, 0, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    ORANGE = (0, 165, 255)
    PURPLE = (128, 0, 128)

    # Semantic colors
    SUCCESS = GREEN
    ERROR = RED
    WARNING = YELLOW
    INFO = BLUE
    PRIMARY = BLUE
    SECONDARY = CYAN
