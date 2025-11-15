"""
Types package - fundamental types without external dependencies.

This package contains basic types that are used throughout the system:
- Enums (EdgeMethod, ColorType, etc.)
- Constants (ImageConstants, VisionConstants, etc.)
- Base models (Point, ROI)

IMPORTANT: This package must NOT import from any other project packages
(schemas, core, services, api) to avoid circular dependencies.
"""

# Export base models
from common.base import ROI, Point

# Export all constants
from common.constants import (
    APIConstants,
    CameraConstants,
    Colors,
    ImageConstants,
    SystemConstants,
    TemplateConstants,
    VisionConstants,
)

# Export all enums
from common.enums import (
    AngleRange,
    ArucoDetectionMode,
    ArucoDict,
    ArucoReferenceMode,
    CameraType,
    ColorMethod,
    EdgeMethod,
    InspectionResult,
    RotationMethod,
    TemplateMethod,
    VisionObjectType,
)

__all__ = [
    # Enums
    "AngleRange",
    "ArucoDetectionMode",
    "ArucoDict",
    "ArucoReferenceMode",
    "CameraType",
    "ColorMethod",
    "EdgeMethod",
    "InspectionResult",
    "RotationMethod",
    "TemplateMethod",
    "VisionObjectType",
    # Constants
    "APIConstants",
    "CameraConstants",
    "Colors",
    "ImageConstants",
    "SystemConstants",
    "TemplateConstants",
    "VisionConstants",
    # Base models
    "Point",
    "ROI",
]
