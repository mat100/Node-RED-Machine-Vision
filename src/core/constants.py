"""
Constants and configuration values for Machine Vision Flow system.
Centralizes all magic numbers and configuration constants.
"""


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
