"""
Custom exceptions and error handlers for the Machine Vision Flow API.
Provides consistent error handling across all endpoints.
"""

import asyncio
import logging
import traceback
from functools import wraps
from typing import Dict, Optional

from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

logger = logging.getLogger(__name__)


# Custom exception classes
class MVException(Exception):
    """Base exception for Machine Vision Flow."""

    def __init__(self, message: str, status_code: int = 500, details: Optional[Dict] = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


class ImageNotFoundException(MVException):
    """Exception raised when image is not found."""

    def __init__(self, image_id: str):
        super().__init__(
            message=f"Image not found: {image_id}", status_code=404, details={"image_id": image_id}
        )


class CameraNotFoundException(MVException):
    """Exception raised when camera is not found."""

    def __init__(self, camera_id: str):
        super().__init__(
            message=f"Camera not found: {camera_id}",
            status_code=404,
            details={"camera_id": camera_id},
        )


class CameraConnectionException(MVException):
    """Exception raised when camera connection fails."""

    def __init__(self, camera_id: str, reason: str):
        super().__init__(
            message=f"Failed to connect to camera {camera_id}: {reason}",
            status_code=503,
            details={"camera_id": camera_id, "reason": reason},
        )


class TemplateNotFoundException(MVException):
    """Exception raised when template is not found."""

    def __init__(self, template_id: str):
        super().__init__(
            message=f"Template not found: {template_id}",
            status_code=404,
            details={"template_id": template_id},
        )


class InvalidROIException(MVException):
    """Exception raised when ROI is invalid."""

    def __init__(self, roi: Dict, reason: str):
        super().__init__(
            message=f"Invalid ROI: {reason}",
            status_code=400,
            details={"roi": roi, "reason": reason},
        )


class ProcessingException(MVException):
    """Exception raised when image processing fails."""

    def __init__(self, operation: str, reason: str):
        super().__init__(
            message=f"Processing failed for {operation}: {reason}",
            status_code=500,
            details={"operation": operation, "reason": reason},
        )


class StorageException(MVException):
    """Exception raised when storage operations fail."""

    def __init__(self, operation: str, reason: str):
        super().__init__(
            message=f"Storage operation failed: {operation} - {reason}",
            status_code=507,  # Insufficient Storage
            details={"operation": operation, "reason": reason},
        )


class ConfigurationException(MVException):
    """Exception raised when configuration is invalid."""

    def __init__(self, config_key: str, reason: str):
        super().__init__(
            message=f"Invalid configuration for {config_key}: {reason}",
            status_code=500,
            details={"config_key": config_key, "reason": reason},
        )


# Exception handlers for FastAPI
async def mv_exception_handler(request: Request, exc: MVException) -> JSONResponse:
    """
    Handler for custom Machine Vision exceptions.

    Args:
        request: FastAPI request
        exc: MVException instance

    Returns:
        JSON response with error details
    """
    logger.error(f"MVException: {exc.message}", extra={"details": exc.details})

    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message, "details": exc.details, "type": exc.__class__.__name__},
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Handler for request validation errors.

    Args:
        request: FastAPI request
        exc: Validation exception

    Returns:
        JSON response with validation error details
    """
    errors = []
    for error in exc.errors():
        errors.append(
            {
                "field": ".".join(str(loc) for loc in error["loc"][1:]),
                "message": error["msg"],
                "type": error["type"],
            }
        )

    logger.warning(f"Validation error: {errors}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Validation failed", "details": errors, "type": "ValidationError"},
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handler for unexpected exceptions.

    Args:
        request: FastAPI request
        exc: Any exception

    Returns:
        JSON response with generic error message
    """
    # Log full traceback for debugging
    logger.error(f"Unexpected error: {exc}", exc_info=True)

    # Don't expose internal details in production
    # Check if we're in debug mode
    debug_mode = getattr(request.app.state, "debug", False)

    if debug_mode:
        # WARNING: Debug mode exposes full stack traces and internal details
        # This should NEVER be enabled in production environments
        logger.warning(
            "Debug mode is enabled - exposing stack traces. "
            "Ensure this is NOT a production environment!"
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "details": {
                    "exception": str(exc),
                    "type": exc.__class__.__name__,
                    "traceback": traceback.format_exc(),
                },
                "type": "InternalError",
                "debug_warning": "Stack traces exposed - debug mode enabled",
            },
        )
    else:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "details": {}, "type": "InternalError"},
        )


# Exception mapping for safe_endpoint decorator
# Maps exception types to (status_code, error_message_template, log_level)
EXCEPTION_MAPPING = {
    ValidationError: (400, "Validation failed", "warning", lambda e: {"details": e.errors()}),
    KeyError: (400, "Missing required field", "error", lambda e: {"field": str(e)}),
    ValueError: (400, "Invalid value", "error", lambda e: {"details": str(e)}),
    FileNotFoundError: (404, "File not found", "error", lambda e: {"details": str(e)}),
    PermissionError: (403, "Permission denied", "error", lambda e: {"details": str(e)}),
    TimeoutError: (504, "Operation timed out", "error", lambda e: {"details": str(e)}),
}


# Decorator for safe endpoint execution
def safe_endpoint(func):
    """
    Decorator to wrap endpoint functions with error handling.

    Automatically catches and handles common exceptions using EXCEPTION_MAPPING.
    Reduces code duplication by using a configuration-based approach.
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            # Handle both sync and async functions
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        except (MVException, HTTPException):
            # Re-raise custom exceptions (handled by exception handler)
            raise

        except Exception as e:
            # Check if exception type is in mapping
            exception_type = type(e)

            if exception_type in EXCEPTION_MAPPING:
                status_code, error_msg, log_level, detail_builder = EXCEPTION_MAPPING[
                    exception_type
                ]

                # Log with appropriate level
                log_message = f"{exception_type.__name__} in {func.__name__}: {e}"
                if log_level == "warning":
                    logger.warning(log_message)
                else:
                    logger.error(log_message)

                # Build response detail
                detail = {"error": error_msg}
                detail.update(detail_builder(e))

                raise HTTPException(status_code=status_code, detail=detail)

            else:
                # Catch all other exceptions
                logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
                raise HTTPException(
                    status_code=500, detail={"error": "Internal server error", "details": str(e)}
                )

    return wrapper


# Helper function to register all exception handlers
def register_exception_handlers(app):
    """
    Register all exception handlers with the FastAPI app.

    Args:
        app: FastAPI application instance
    """
    app.add_exception_handler(MVException, mv_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    logger.info("Exception handlers registered")


# Standard Messages
class ErrorMessages:
    """Standard error messages."""

    # Image errors
    IMAGE_NOT_FOUND = "Image with ID {image_id} not found"
    IMAGE_STORAGE_FULL = "Image storage is full, cannot store new image"
    INVALID_IMAGE_FORMAT = "Invalid image format: {format}"

    # Camera errors
    CAMERA_NOT_FOUND = "Camera {camera_id} not found"
    CAMERA_CONNECTION_FAILED = "Failed to connect to camera {camera_id}: {error}"
    CAMERA_CAPTURE_FAILED = "Failed to capture from camera {camera_id}: {error}"
    CAMERA_ALREADY_EXISTS = "Camera {camera_id} already exists"

    # Template errors
    TEMPLATE_NOT_FOUND = "Template {template_id} not found"
    TEMPLATE_UPLOAD_FAILED = "Failed to upload template: {error}"
    TEMPLATE_INVALID_SIZE = "Template size {size} is invalid (min: {min}, max: {max})"

    # ROI errors
    ROI_OUT_OF_BOUNDS = "ROI {roi} is out of image bounds {bounds}"
    ROI_INVALID_SIZE = "ROI size is invalid: {width}x{height}"
    ROI_MISSING_PARAMS = "ROI requires all parameters: x, y, width, height"

    # Processing errors
    PROCESSING_FAILED = "Image processing failed: {error}"
    INVALID_PARAMETER = "Invalid parameter {param}: {value}"
    OPERATION_TIMEOUT = "Operation timed out after {timeout} seconds"

    # System errors
    INITIALIZATION_FAILED = "Failed to initialize {component}: {error}"
    CONFIGURATION_ERROR = "Configuration error: {error}"
    RESOURCE_EXHAUSTED = "Resource exhausted: {resource}"


class SuccessMessages:
    """Standard success messages."""

    CAMERA_CONNECTED = "Successfully connected to camera {camera_id}"
    CAMERA_DISCONNECTED = "Successfully disconnected camera {camera_id}"
    IMAGE_CAPTURED = "Successfully captured image {image_id}"
    TEMPLATE_UPLOADED = "Successfully uploaded template {template_id}"
    TEMPLATE_DELETED = "Successfully deleted template {template_id}"
    PROCESSING_COMPLETE = "Processing completed successfully"
