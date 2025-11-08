"""
Centralized enums for machine vision system.

This module contains all enumeration types used throughout the system,
providing a single source of truth for enum definitions.
"""

from enum import Enum


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
