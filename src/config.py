"""
Configuration management using Pydantic for Machine Vision Flow.
Provides type-safe configuration with validation and environment variable support.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseSettings, Field, root_validator, validator

from core.constants import (
    APIConstants,
    CameraConstants,
    ImageConstants,
    SystemConstants,
    TemplateConstants,
    VisionConstants,
)

logger = logging.getLogger(__name__)


class ImageConfig(BaseSettings):
    """Image management configuration."""

    max_images: int = Field(
        default=ImageConstants.DEFAULT_MAX_IMAGES,
        ge=ImageConstants.MIN_IMAGES,
        le=ImageConstants.MAX_IMAGES,
        description="Maximum number of images to store in memory",
    )
    max_memory_mb: int = Field(
        default=ImageConstants.DEFAULT_MAX_MEMORY_MB,
        ge=100,
        le=10000,
        description="Maximum memory for image storage in MB",
    )
    thumbnail_width: int = Field(
        default=ImageConstants.DEFAULT_THUMBNAIL_WIDTH,
        ge=ImageConstants.MIN_THUMBNAIL_WIDTH,
        le=ImageConstants.MAX_THUMBNAIL_WIDTH,
        description="Default thumbnail width in pixels",
    )
    thumbnail_quality: int = Field(
        default=ImageConstants.THUMBNAIL_JPEG_QUALITY,
        ge=1,
        le=100,
        description="JPEG quality for thumbnails",
    )
    cleanup_threshold: float = Field(
        default=ImageConstants.MEMORY_CLEANUP_THRESHOLD,
        ge=0.5,
        le=1.0,
        description="Memory usage threshold to trigger cleanup",
    )

    class Config:
        env_prefix = "MV_IMAGE_"
        extra = "ignore"


class CameraConfig(BaseSettings):
    """Camera configuration."""

    default_camera: str = Field(
        default=CameraConstants.DEFAULT_CAMERA_ID, description="Default camera ID"
    )
    capture_timeout_ms: int = Field(
        default=CameraConstants.DEFAULT_CAPTURE_TIMEOUT_MS,
        ge=CameraConstants.MIN_CAPTURE_TIMEOUT_MS,
        le=CameraConstants.MAX_CAPTURE_TIMEOUT_MS,
        description="Camera capture timeout in milliseconds",
    )
    max_usb_cameras: int = Field(
        default=CameraConstants.MAX_USB_CAMERAS_TO_CHECK,
        ge=1,
        le=10,
        description="Maximum USB cameras to check during enumeration",
    )
    stream_fps: int = Field(
        default=CameraConstants.MJPEG_FPS, ge=1, le=60, description="MJPEG stream FPS"
    )
    stream_quality: int = Field(
        default=CameraConstants.MJPEG_QUALITY, ge=1, le=100, description="MJPEG stream quality"
    )
    max_concurrent_streams: int = Field(
        default=CameraConstants.MAX_CONCURRENT_STREAMS,
        ge=1,
        le=10,
        description="Maximum concurrent MJPEG streams",
    )
    test_image_width: int = Field(
        default=CameraConstants.TEST_IMAGE_WIDTH, ge=100, le=4096, description="Test image width"
    )
    test_image_height: int = Field(
        default=CameraConstants.TEST_IMAGE_HEIGHT, ge=100, le=4096, description="Test image height"
    )

    class Config:
        env_prefix = "MV_CAMERA_"
        extra = "ignore"


class TemplateConfig(BaseSettings):
    """Template management configuration."""

    storage_path: str = Field(
        default=TemplateConstants.DEFAULT_STORAGE_PATH,
        description="Path to template storage directory",
    )
    max_file_size_mb: int = Field(
        default=TemplateConstants.MAX_TEMPLATE_SIZE_MB,
        ge=1,
        le=100,
        description="Maximum template file size in MB",
    )
    allowed_formats: List[str] = Field(
        default=TemplateConstants.ALLOWED_FORMATS, description="Allowed template image formats"
    )
    max_templates: int = Field(
        default=TemplateConstants.MAX_TEMPLATES,
        ge=1,
        le=10000,
        description="Maximum number of templates",
    )
    default_threshold: float = Field(
        default=TemplateConstants.DEFAULT_THRESHOLD,
        ge=TemplateConstants.MIN_THRESHOLD,
        le=TemplateConstants.MAX_THRESHOLD,
        description="Default template matching threshold",
    )

    @validator("storage_path")
    def validate_storage_path(cls, v):
        """Ensure storage path exists or can be created."""
        path = Path(v)
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ValueError(f"Cannot create storage path: {e}")
        return str(path.absolute())

    class Config:
        env_prefix = "MV_TEMPLATE_"
        extra = "ignore"


class VisionConfig(BaseSettings):
    """Vision processing configuration."""

    canny_low_threshold: int = Field(
        default=VisionConstants.CANNY_LOW_THRESHOLD_DEFAULT,
        ge=VisionConstants.CANNY_LOW_THRESHOLD_MIN,
        le=VisionConstants.CANNY_LOW_THRESHOLD_MAX,
        description="Default Canny edge detection low threshold",
    )
    canny_high_threshold: int = Field(
        default=VisionConstants.CANNY_HIGH_THRESHOLD_DEFAULT,
        ge=VisionConstants.CANNY_HIGH_THRESHOLD_MIN,
        le=VisionConstants.CANNY_HIGH_THRESHOLD_MAX,
        description="Default Canny edge detection high threshold",
    )
    max_contours: int = Field(
        default=VisionConstants.MAX_CONTOURS_DEFAULT,
        ge=1,
        le=VisionConstants.MAX_CONTOURS_LIMIT,
        description="Maximum contours to detect",
    )
    min_contour_area: int = Field(
        default=VisionConstants.MIN_CONTOUR_AREA_DEFAULT, ge=1, description="Minimum contour area"
    )
    gaussian_blur_size: int = Field(
        default=VisionConstants.GAUSSIAN_BLUR_SIZE_DEFAULT,
        ge=VisionConstants.GAUSSIAN_BLUR_SIZE_MIN,
        le=VisionConstants.GAUSSIAN_BLUR_SIZE_MAX,
        description="Default Gaussian blur kernel size",
    )

    @validator("gaussian_blur_size")
    def validate_odd_number(cls, v):
        """Ensure kernel size is odd."""
        if v % 2 == 0:
            return v + 1
        return v

    class Config:
        env_prefix = "MV_VISION_"
        extra = "ignore"


class APIConfig(BaseSettings):
    """API configuration."""

    host: str = Field(default="0.0.0.0", description="API host address")
    port: int = Field(default=8000, ge=1, le=65535, description="API port")
    api_version: str = Field(default=APIConstants.API_VERSION, description="API version")
    cors_enabled: bool = Field(default=True, description="Enable CORS")
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")
    max_upload_size_mb: int = Field(
        default=APIConstants.MAX_UPLOAD_SIZE_MB,
        ge=1,
        le=500,
        description="Maximum upload file size in MB",
    )
    request_timeout: int = Field(
        default=APIConstants.REQUEST_TIMEOUT_SECONDS,
        ge=1,
        le=300,
        description="Request timeout in seconds",
    )
    rate_limit_enabled: bool = Field(default=False, description="Enable rate limiting")
    rate_limit_per_minute: int = Field(
        default=APIConstants.REQUESTS_PER_MINUTE,
        ge=1,
        le=10000,
        description="Rate limit per minute",
    )

    class Config:
        env_prefix = "MV_API_"
        extra = "ignore"


class SystemConfig(BaseSettings):
    """System configuration."""

    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default=SystemConstants.LOG_LEVEL_DEFAULT, description="Logging level")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    data_dir: str = Field(default=SystemConstants.DATA_DIR, description="Data directory path")
    temp_dir: str = Field(default=SystemConstants.TEMP_DIR, description="Temporary files directory")
    worker_threads: int = Field(
        default=SystemConstants.THREAD_POOL_SIZE,
        ge=1,
        le=SystemConstants.MAX_WORKER_THREADS,
        description="Number of worker threads",
    )
    cleanup_interval: int = Field(
        default=SystemConstants.CLEANUP_INTERVAL_SECONDS,
        ge=60,
        le=86400,
        description="Cleanup interval in seconds",
    )
    health_check_enabled: bool = Field(default=True, description="Enable health checks")

    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v_upper

    @validator("data_dir", "temp_dir")
    def create_directories(cls, v):
        """Ensure directories exist."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path.absolute())

    class Config:
        env_prefix = "MV_SYSTEM_"
        extra = "ignore"


class Settings(BaseSettings):
    """Main application settings."""

    # Sub-configurations
    image: ImageConfig = Field(default_factory=ImageConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    template: TemplateConfig = Field(default_factory=TemplateConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)

    # Environment
    environment: str = Field(
        default="production", description="Environment (development, staging, production)"
    )

    # Config file support
    config_file: Optional[str] = Field(default=None, description="Path to YAML config file")

    @root_validator(pre=True)
    def load_config_file(cls, values):
        """Load configuration from YAML file if specified."""
        import os

        config_file = values.get("config_file") or os.getenv("MV_CONFIG_FILE")

        if config_file and Path(config_file).exists():
            import yaml

            try:
                with open(config_file, "r") as f:
                    file_config = yaml.safe_load(f)
                    if file_config:
                        # Merge file config with values (env vars take precedence)
                        for key, value in file_config.items():
                            if key not in values or values[key] is None:
                                values[key] = value
            except Exception as e:
                logger.warning(f"Failed to load config file {config_file}: {e}")

        return values

    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment value."""
        valid_envs = ["development", "staging", "production", "test"]
        if v not in valid_envs:
            raise ValueError(f"Invalid environment: {v}. Must be one of {valid_envs}")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return self.dict(exclude_none=True)

    def save_to_file(self, path: str) -> None:
        """Save current configuration to YAML file."""
        import yaml

        config_dict = self.to_dict()
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    class Config:
        env_prefix = "MV_"
        case_sensitive = False
        env_nested_delimiter = "__"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings object with validated configuration
    """
    return Settings()


# Convenience function to reload settings (clears cache)
def reload_settings() -> Settings:
    """
    Reload settings, clearing the cache.

    Returns:
        Fresh Settings object
    """
    get_settings.cache_clear()
    return get_settings()
