"""
Pydantic models for data validation and serialization.

This module contains all Pydantic models used throughout the application for:
- Request/response validation (API layer)
- Data transfer between layers
- Configuration and parameters

Organized by domain:
- Base: Common base classes
- Common: Core data structures (VisionObject, VisionResponse, etc.)
- Camera: Camera operations and capture
- Image: Image processing operations
- Reference: ArUco reference frames and coordinate transformation
- System: System status and monitoring
- Template: Template matching operations
- Params: Detection algorithm parameters
- Vision: Vision processing requests

Note: This module uses 'models' (not 'schemas') for clarity,
as there are no database models in this project.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, root_validator, validator

# Import base types from src.types module
from src.domain_types import (
    ROI,
    AngleRange,
    ArucoDict,
    ArucoReferenceMode,
    AsymmetryOrientation,
    ColorMethod,
    Point,
    RotationMethod,
    TemplateMethod,
)

# ==============================================================================
# Base Models
# ==============================================================================


class BaseDetectionParams(BaseModel):
    """
    Base class for all detection parameter models.

    Provides common functionality including:
    - to_dict() method with enum conversion
    - Consistent configuration
    - Standardized serialization

    All detection parameter classes should inherit from this base class
    to ensure consistent behavior across the vision system.
    """

    class Config:
        """Pydantic configuration."""

        extra = "forbid"  # Reject unknown fields
        use_enum_values = False  # Keep enums as enum instances

    def to_dict(self) -> Dict[str, Any]:
        """
        Export parameters to dictionary for detector functions.

        Automatically converts enum values to strings, which is required
        by most OpenCV and vision processing functions.

        Returns:
            Dictionary with enum values converted to strings

        Example:
            >>> params = EdgeDetectionParams(method=EdgeMethod.CANNY)
            >>> data = params.to_dict()
            >>> # Returns {"method": "canny", ...}
        """
        data = self.dict(exclude_none=True)

        # Convert enum values to strings
        for key, value in data.items():
            if hasattr(value, "value"):
                data[key] = value.value

        return data

    def get_param(self, name: str, default: Any = None) -> Any:
        """
        Safely get a parameter value.

        Args:
            name: Parameter name
            default: Default value if parameter doesn't exist

        Returns:
            Parameter value or default

        Example:
            >>> threshold = params.get_param("threshold", 100)
        """
        return getattr(self, name, default)


# ==============================================================================
# Common Models
# ==============================================================================


class Size(BaseModel):
    """Image size"""

    width: int
    height: int


class VisionObject(BaseModel):
    """
    Universal interface for any vision processing object.
    (camera capture, contour, template, color region, etc.)
    Provides standardized location, geometry, and quality information.
    """

    # Identification
    object_id: str = Field(..., description="Unique ID of this object")
    object_type: str = Field(
        ...,
        description="Type: camera_capture, edge_contour, template_match, etc.",
    )

    # Position & Geometry
    bounding_box: ROI = Field(
        ..., description="Bounding box of detected object in {x, y, width, height} format"
    )
    center: Point = Field(..., description="Center point of the object")

    # Quality
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0 - 1.0)")

    # Optional geometry
    area: Optional[float] = Field(None, description="Area in pixels")
    perimeter: Optional[float] = Field(None, description="Perimeter in pixels")
    rotation: Optional[float] = Field(None, description="Rotation in degrees (0-360)")

    # Plane coordinates (when reference frame is applied)
    plane_position: Optional[Point] = Field(
        None, description="Position in reference plane coordinates (units from reference_object)"
    )
    plane_rotation: Optional[float] = Field(
        None, description="Rotation in reference plane (degrees, when reference applied)"
    )

    # Type-specific properties
    properties: Dict[str, Any] = Field(default_factory=dict, description="Type-specific properties")

    # Raw data (optional)
    contour: Optional[List] = Field(None, description="Contour points for edge detection")


class ReferenceObject(BaseModel):
    """
    Universal reference object for coordinate transformation.

    Contains homography matrix and metadata for transforming image coordinates
    to reference frame coordinates. Supports multiple reference types (single
    marker, plane, checkerboard, etc.).
    """

    type: str = Field(
        description="Reference type: single_marker | plane | checkerboard | custom",
    )

    units: str = Field(
        description="Units of transformed coordinates: pixels | mm | cm | m",
    )

    homography_matrix: List[List[float]] = Field(
        description="3x3 homography transformation matrix",
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Type-specific metadata (marker IDs, dimensions, etc.)",
    )

    thumbnail: str = Field(
        description=(
            "Base64-encoded JPEG visualization of reference plane showing: "
            "detected markers with IDs, coordinate axes (X/Y), grid in physical units, "
            "and origin point [0,0]. Format: data:image/jpeg;base64,..."
        ),
    )

    @validator("homography_matrix")
    def validate_homography(cls, v):
        if len(v) != 3:
            raise ValueError("Homography matrix must have 3 rows")
        for row in v:
            if len(row) != 3:
                raise ValueError("Each row of homography matrix must have 3 columns")
        return v

    @validator("type")
    def validate_type(cls, v):
        valid_types = ["single_marker", "plane", "checkerboard", "custom"]
        if v not in valid_types:
            raise ValueError(f"type must be one of {valid_types}")
        return v

    @validator("units")
    def validate_units(cls, v):
        valid_units = ["pixels", "mm", "cm", "m"]
        if v not in valid_units:
            raise ValueError(f"units must be one of {valid_units}")
        return v


class VisionResponse(BaseModel):
    """
    Simplified response for all vision processing APIs.

    Contains detected objects, visualization thumbnail, and optional reference
    frame for coordinate transformation (when using single/plane detection modes).
    """

    objects: List[VisionObject] = Field(default_factory=list, description="List of vision objects")
    thumbnail_base64: str = Field(..., description="Base64-encoded thumbnail with visualization")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    reference_object: Optional[ReferenceObject] = Field(
        None,
        description=(
            "Reference frame for coordinate transformation (present when using "
            "ArUco single or plane detection modes)"
        ),
    )


class ArucoReferenceResponse(BaseModel):
    """
    Response for ArUco reference frame creation.

    Contains the created reference object, detected markers used for calibration,
    visualization thumbnail, and processing time.
    """

    reference_object: ReferenceObject = Field(
        ...,
        description=(
            "Reference frame created from ArUco markers. Contains homography matrix "
            "and metadata for transforming coordinates to real-world units."
        ),
    )
    markers: List[VisionObject] = Field(
        ...,
        description="Detected ArUco markers used to create the reference frame",
    )
    thumbnail_base64: str = Field(
        ..., description="Base64-encoded thumbnail with markers visualized"
    )
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")


# ==============================================================================
# Camera Models
# ==============================================================================


class CameraInfo(BaseModel):
    """Camera information"""

    id: str
    name: str
    type: str
    resolution: Size
    connected: bool

    @classmethod
    def from_manager_dict(cls, data: Dict[str, Any]) -> "CameraInfo":
        """
        Create CameraInfo from camera manager dict.

        Args:
            data: Dictionary from camera manager with camera details

        Returns:
            CameraInfo instance
        """
        return cls(
            id=data["id"],
            name=data["name"],
            type=data["type"],
            resolution=Size(
                width=data.get("resolution", {}).get("width", 1920),
                height=data.get("resolution", {}).get("height", 1080),
            ),
            connected=data.get("connected", False),
        )


class CameraConnectRequest(BaseModel):
    """Request to connect to camera"""

    camera_id: str
    resolution: Optional[Size] = None


class CaptureParams(BaseModel):
    """Camera capture parameters."""

    class Config:
        extra = "forbid"

    roi: Optional[ROI] = Field(
        None, description="Region of interest to extract from captured image"
    )


class CaptureRequest(BaseModel):
    """Request to capture image from camera"""

    camera_id: str = Field(
        description="Camera identifier (e.g., 'usb_0', 'ip_192.168.1.100', 'test')"
    )
    params: Optional[CaptureParams] = Field(None, description="Capture parameters (ROI, etc.)")


# ==============================================================================
# Image Models
# ==============================================================================


class ROIExtractRequest(BaseModel):
    """Request to extract ROI from image"""

    image_id: str
    roi: ROI = Field(..., description="Region of interest to extract")


class ImageImportRequest(BaseModel):
    """Request to import image from file system"""

    file_path: str = Field(..., description="Path to image file (JPG, PNG, BMP, etc.)")


# ==============================================================================
# Reference Models
# ==============================================================================


class SingleConfig(BaseModel):
    """
    Configuration for single marker reference mode.

    Uses one ArUco marker with known physical size to establish a reference
    coordinate system. Suitable for perpendicular camera views where uniform
    scaling is acceptable.
    """

    marker_id: int = Field(
        description="ArUco marker ID to use as reference",
        ge=0,
    )

    marker_size_mm: float = Field(
        description="Physical marker size in millimeters (outer edge to outer edge)",
        gt=0,
    )

    origin: str = Field(
        default="marker_center",
        description=(
            "Origin point for coordinate system: "
            "marker_center | marker_top_left | marker_bottom_left"
        ),
    )

    rotation_reference: str = Field(
        default="marker_rotation",
        description=(
            "Rotation reference: "
            "marker_rotation (0° = marker at 0°) | image_axes (0° = horizontal right)"
        ),
    )

    @validator("origin")
    def validate_origin(cls, v):
        valid_origins = ["marker_center", "marker_top_left", "marker_bottom_left"]
        if v not in valid_origins:
            raise ValueError(f"origin must be one of {valid_origins}")
        return v

    @validator("rotation_reference")
    def validate_rotation_reference(cls, v):
        valid_refs = ["marker_rotation", "image_axes"]
        if v not in valid_refs:
            raise ValueError(f"rotation_reference must be one of {valid_refs}")
        return v


class PlaneConfig(BaseModel):
    """
    Configuration for 4-marker plane reference mode.

    Uses four ArUco markers at known positions to establish a reference plane
    with perspective transformation. Robust to camera angle and position changes.
    """

    marker_ids: Dict[str, int] = Field(
        description=(
            "Marker IDs for plane corners. Must contain keys: "
            "top_left, top_right, bottom_right, bottom_left"
        ),
    )

    width_mm: float = Field(
        description="Plane width in millimeters (measured between marker centers)",
        gt=0,
    )

    height_mm: float = Field(
        description="Plane height in millimeters (measured between marker centers)",
        gt=0,
    )

    origin: str = Field(
        default="top_left",
        description=(
            "Which corner is the origin (0,0): " "top_left | top_right | bottom_left | bottom_right"
        ),
    )

    x_direction: str = Field(
        default="right",
        description="Direction of X axis: right | left | up | down",
    )

    y_direction: str = Field(
        default="down",
        description="Direction of Y axis: right | left | up | down",
    )

    @validator("marker_ids")
    def validate_marker_ids(cls, v):
        required_keys = {"top_left", "top_right", "bottom_right", "bottom_left"}
        if set(v.keys()) != required_keys:
            raise ValueError(f"marker_ids must contain exactly these keys: {required_keys}")

        # Check all IDs are unique
        ids = list(v.values())
        if len(ids) != len(set(ids)):
            raise ValueError("All marker IDs must be unique")

        return v

    @validator("origin")
    def validate_origin(cls, v):
        valid_origins = ["top_left", "top_right", "bottom_left", "bottom_right"]
        if v not in valid_origins:
            raise ValueError(f"origin must be one of {valid_origins}")
        return v

    @validator("x_direction", "y_direction")
    def validate_direction(cls, v):
        valid_directions = ["right", "left", "up", "down"]
        if v not in valid_directions:
            raise ValueError(f"direction must be one of {valid_directions}")
        return v


# ==============================================================================
# System Models
# ==============================================================================


class SystemStatus(BaseModel):
    """System status information"""

    status: str
    uptime: float
    memory_usage: Dict[str, float]
    active_cameras: int
    buffer_usage: Dict[str, Any]


class PerformanceMetrics(BaseModel):
    """Performance metrics"""

    avg_processing_time: float
    total_inspections: int
    success_rate: float
    operations_per_minute: float


class DebugSettings(BaseModel):
    """Debug settings"""

    enabled: bool
    save_images: bool
    show_visualizations: bool
    verbose_logging: bool


# ==============================================================================
# Template Models
# ==============================================================================


class TemplateInfo(BaseModel):
    """Template information"""

    id: str
    name: str
    description: Optional[str] = None
    size: Size
    created_at: datetime

    @classmethod
    def from_manager_dict(cls, data: Dict[str, Any]) -> "TemplateInfo":
        """
        Create TemplateInfo from template manager dict.

        Args:
            data: Dictionary from template manager with template details

        Returns:
            TemplateInfo instance
        """
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description"),
            size=Size(width=data["size"]["width"], height=data["size"]["height"]),
            created_at=datetime.fromisoformat(data["created_at"]),
        )


class TemplateUploadResponse(BaseModel):
    """Response from template upload"""

    success: bool
    template_id: str
    name: str
    size: Size


class TemplateLearnRequest(BaseModel):
    """Request to learn template from image"""

    image_id: str
    name: str
    roi: ROI
    description: Optional[str] = None


# ==============================================================================
# Detection Parameters
# ==============================================================================


class EdgeDetectionParams(BaseDetectionParams):
    """
    Unified edge detection parameters (flat structure).

    Contains all preprocessing, filtering, and method-specific parameters
    for all edge detection algorithms with validation and defaults.
    """

    # === Method selection ===
    method: str = Field(
        default="canny",
        description=(
            "Edge detection method (canny, sobel, laplacian, "
            "prewitt, scharr, roberts, morphgrad)"
        ),
    )

    # === Preprocessing parameters (common to all methods) ===
    blur_enabled: bool = Field(default=False, description="Enable Gaussian blur preprocessing")
    blur_kernel: int = Field(
        default=5,
        ge=3,
        description="Gaussian blur kernel size (must be odd)",
    )
    bilateral_enabled: bool = Field(
        default=False, description="Enable bilateral filter (edge-preserving blur)"
    )
    bilateral_d: int = Field(default=9, ge=1, description="Bilateral filter diameter")
    bilateral_sigma_color: float = Field(
        default=75.0,
        ge=0,
        description="Bilateral filter sigma in color space",
    )
    bilateral_sigma_space: float = Field(
        default=75.0,
        ge=0,
        description="Bilateral filter sigma in coordinate space",
    )
    morphology_enabled: bool = Field(
        default=False, description="Enable morphological preprocessing"
    )
    morphology_operation: str = Field(
        default="close", description="Morphological operation (close/open/gradient)"
    )
    morphology_kernel: int = Field(
        default=3,
        ge=1,
        description="Morphological kernel size",
    )
    equalize_enabled: bool = Field(default=False, description="Enable histogram equalization")

    # === Contour filtering parameters (common to all methods) ===
    min_contour_area: float = Field(
        default=10.0,
        ge=0,
        description="Minimum contour area in pixels",
    )
    max_contour_area: float = Field(
        default=100000.0, ge=0, description="Maximum contour area in pixels"
    )
    min_contour_perimeter: float = Field(
        default=0.0,
        ge=0,
        description="Minimum contour perimeter in pixels",
    )
    max_contour_perimeter: float = Field(
        default=float("inf"), description="Maximum contour perimeter in pixels"
    )
    max_contours: int = Field(
        default=100,
        ge=1,
        description="Maximum number of contours to return",
    )
    show_centers: bool = Field(
        default=True,
        description="Show contour centers in visualization",
    )

    # === Canny edge detection parameters ===
    canny_low: int = Field(
        default=50,
        ge=0,
        le=500,
        description="Canny low threshold",
    )
    canny_high: int = Field(
        default=150,
        ge=0,
        le=500,
        description="Canny high threshold",
    )
    canny_aperture: int = Field(
        default=3,
        ge=3,
        le=7,
        description="Canny aperture size (must be odd)",
    )
    canny_l2_gradient: bool = Field(
        default=False,
        description="Use L2 gradient norm (more accurate but slower)",
    )

    # === Sobel edge detection parameters ===
    sobel_threshold: float = Field(default=50.0, ge=0, description="Sobel edge threshold")
    sobel_kernel: int = Field(
        default=3,
        ge=1,
        le=31,
        description="Sobel kernel size (must be odd)",
    )
    sobel_scale: float = Field(default=1.0, ge=0, description="Sobel scale factor")
    sobel_delta: float = Field(default=0.0, ge=0, description="Sobel delta (added to result)")

    # === Laplacian edge detection parameters ===
    laplacian_threshold: float = Field(
        default=30.0,
        ge=0,
        description="Laplacian edge threshold",
    )
    laplacian_kernel: int = Field(
        default=3,
        ge=1,
        le=31,
        description="Laplacian kernel size (must be odd)",
    )
    laplacian_scale: float = Field(default=1.0, ge=0, description="Laplacian scale factor")
    laplacian_delta: float = Field(
        default=0.0,
        ge=0,
        description="Laplacian delta (added to result)",
    )

    # === Prewitt edge detection parameters ===
    prewitt_threshold: float = Field(default=50.0, ge=0, description="Prewitt edge threshold")

    # === Scharr edge detection parameters ===
    scharr_threshold: float = Field(default=50.0, ge=0, description="Scharr edge threshold")
    scharr_scale: float = Field(default=1.0, ge=0, description="Scharr scale factor")
    scharr_delta: float = Field(
        default=0.0,
        ge=0,
        description="Scharr delta (added to result)",
    )

    # === Morphological gradient parameters ===
    morph_threshold: float = Field(
        default=30.0,
        ge=0,
        description="Morphological gradient threshold",
    )
    morph_kernel: int = Field(
        default=3,
        ge=1,
        description="Morphological gradient kernel size",
    )


class ColorDetectionParams(BaseDetectionParams):
    """
    Color detection parameters.

    Supports both histogram (fast) and kmeans (accurate) detection methods.
    """

    # === Common parameters ===
    method: ColorMethod = Field(
        default=ColorMethod.HISTOGRAM,
        description="Detection method: histogram (fast) or kmeans (accurate)",
    )
    min_percentage: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Minimum percentage for color match",
    )
    use_contour_mask: bool = Field(
        default=True,
        description="Use contour mask instead of full bounding box when contour is available",
    )

    # === KMeans-specific parameters (ignored if method != kmeans) ===
    kmeans_clusters: int = Field(
        default=3,
        ge=2,
        le=10,
        description="Number of color clusters for k-means",
    )
    kmeans_random_state: int = Field(
        default=42,
        ge=0,
        description="Random state for reproducible k-means results",
    )
    kmeans_n_init: int = Field(
        default=10,
        ge=1,
        description="Number of k-means initializations",
    )


class ArucoDetectionParams(BaseDetectionParams):
    """
    ArUco marker detection parameters (MARKERS mode only).

    This endpoint detects all visible ArUco markers without creating a reference frame.
    For reference frame creation, use the aruco-reference endpoint with ArucoReferenceParams.
    """

    dictionary: ArucoDict = Field(
        default=ArucoDict.DICT_4X4_50,
        description="ArUco dictionary type (defines marker set and size)",
    )


class RotationDetectionParams(BaseDetectionParams):
    """
    Rotation detection parameters.

    Calculates object orientation using minimum area rectangle,
    ellipse fitting, or PCA analysis.
    """

    method: RotationMethod = Field(
        default=RotationMethod.MIN_AREA_RECT,
        description="Rotation calculation method (min_area_rect, ellipse_fit, pca)",
    )
    angle_range: AngleRange = Field(
        default=AngleRange.RANGE_0_360,
        description="Output angle range format (0_360, -180_180, or 0_180)",
    )
    asymmetry_orientation: AsymmetryOrientation = Field(
        default=AsymmetryOrientation.DISABLED,
        description=(
            "Orient angle based on object asymmetry (thickness). "
            "thick_to_thin: 0° points from thick to thin part. "
            "thin_to_thick: 0° points from thin to thick part. "
            "disabled: no asymmetry-based orientation (default)"
        ),
    )


class TemplateMatchParams(BaseDetectionParams):
    """
    Template matching parameters.

    Supports multiple OpenCV matching methods with configurable thresholds.
    """

    template_id: str = Field(description="Template identifier to match against")
    method: TemplateMethod = Field(
        default=TemplateMethod.TM_CCOEFF_NORMED, description="OpenCV template matching method"
    )
    threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Match confidence threshold (0.0 to 1.0)",
    )
    multi_scale: bool = Field(default=False, description="Enable multi-scale template matching")
    scale_range: list = Field(
        default=[0.8, 1.2], description="Scale range for multi-scale matching [min, max]"
    )
    scale_steps: int = Field(
        default=5, ge=2, le=20, description="Number of scale steps for multi-scale matching"
    )


class AdvancedTemplateMatchParams(BaseDetectionParams):
    """
    Advanced template matching parameters with rotation and multi-instance support.

    Extends basic template matching with:
    - Multiple instance detection using Non-Maximum Suppression (NMS)
    - Rotation-invariant matching across specified angle range
    - Overlap filtering for duplicate removal
    """

    template_id: str = Field(description="Template identifier to match against")
    method: TemplateMethod = Field(
        default=TemplateMethod.TM_CCOEFF_NORMED,
        description="OpenCV template matching method",
    )
    threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Match confidence threshold (0.0 to 1.0)",
    )

    # Multi-instance detection
    find_multiple: bool = Field(
        default=False,
        description="Enable multi-instance detection (find all matches above threshold)",
    )
    max_matches: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of matches to return when find_multiple=True",
    )
    overlap_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description=(
            "IoU threshold for NMS overlap filtering "
            "(0.0 = no overlap allowed, 1.0 = full overlap allowed)"
        ),
    )

    # Rotation detection
    enable_rotation: bool = Field(
        default=False,
        description="Enable rotation-invariant matching (searches across rotation_range)",
    )
    rotation_range: tuple = Field(
        default=(-180.0, 180.0),
        description="Rotation angle range in degrees (min, max) for rotation search",
    )
    rotation_step: float = Field(
        default=10.0,
        ge=1.0,
        le=90.0,
        description="Rotation step size in degrees (smaller = more accurate but slower)",
    )

    @validator("rotation_range")
    def validate_rotation_range(cls, v):
        """Validate rotation range is valid tuple."""
        if not isinstance(v, (tuple, list)) or len(v) != 2:
            raise ValueError("rotation_range must be a tuple/list of 2 values (min, max)")
        min_angle, max_angle = v
        if min_angle >= max_angle:
            raise ValueError(f"rotation_range min ({min_angle}) must be < max ({max_angle})")
        if min_angle < -180 or max_angle > 180:
            raise ValueError("rotation_range must be within [-180, 180] degrees")
        return tuple(v)  # Ensure it's a tuple


class FeatureTemplateMatchParams(BaseDetectionParams):
    """
    Feature-based template matching parameters using ORB.

    Uses keypoint detection and descriptor matching for rotation
    and scale invariant template matching.
    """

    template_id: str = Field(description="Template identifier to match against")
    threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Match confidence threshold (0.0 to 1.0)",
    )
    min_matches: int = Field(
        default=10,
        ge=4,
        le=100,
        description="Minimum number of feature matches required",
    )
    ratio_threshold: float = Field(
        default=0.75,
        ge=0.5,
        le=0.95,
        description="Lowe's ratio test threshold for match filtering",
    )
    find_multiple: bool = Field(
        default=False,
        description="Enable multi-instance detection",
    )
    max_matches: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of instances to detect when find_multiple=True",
    )


class ArucoReferenceParams(BaseDetectionParams):
    """
    ArUco reference frame parameters.

    Creates reference coordinate system from ArUco markers for transforming
    other vision objects from pixel space to real-world coordinates.

    Supports two modes:
    - SINGLE: Single marker with known size → affine transform (uniform scaling)
    - PLANE: Four markers at corners → perspective homography transform
    """

    dictionary: ArucoDict = Field(
        default=ArucoDict.DICT_4X4_50,
        description="ArUco dictionary type (defines marker set and size)",
    )

    mode: ArucoReferenceMode = Field(
        description="Reference mode: single (1-marker) or plane (4-marker)",
    )

    single_config: Optional[SingleConfig] = Field(
        default=None,
        description="Single marker configuration (required when mode=single)",
    )

    plane_config: Optional[PlaneConfig] = Field(
        default=None,
        description="Plane configuration (required when mode=plane)",
    )

    @root_validator
    def validate_mode_config(cls, values):
        """Validate that required config is provided for each mode."""
        mode = values.get("mode")
        single_config = values.get("single_config")
        plane_config = values.get("plane_config")

        if mode == ArucoReferenceMode.SINGLE and single_config is None:
            raise ValueError("single_config is required when mode=single")

        if mode == ArucoReferenceMode.PLANE and plane_config is None:
            raise ValueError("plane_config is required when mode=plane")

        # Validate only correct config is provided
        if mode == ArucoReferenceMode.SINGLE and plane_config is not None:
            raise ValueError("plane_config should not be provided when mode=single")

        if mode == ArucoReferenceMode.PLANE and single_config is not None:
            raise ValueError("single_config should not be provided when mode=plane")

        return values


# ==============================================================================
# Vision Request Models
# ==============================================================================


class TemplateMatchRequest(BaseModel):
    """Request for template matching"""

    image_id: str
    roi: Optional[ROI] = Field(None, description="Region of interest to limit search area")
    params: TemplateMatchParams = Field(
        description="Template matching parameters (template_id is required)"
    )
    reference_object: Optional[ReferenceObject] = Field(
        None,
        description=(
            "Optional reference frame for coordinate transformation. "
            "If provided, adds plane_* properties to detected objects."
        ),
    )


class AdvancedTemplateMatchRequest(BaseModel):
    """Request for advanced template matching with rotation and multi-instance support"""

    image_id: str
    roi: Optional[ROI] = Field(None, description="Region of interest to limit search area")
    params: AdvancedTemplateMatchParams = Field(
        description="Advanced template matching parameters (rotation, multi-instance, etc.)"
    )
    reference_object: Optional[ReferenceObject] = Field(
        None,
        description=(
            "Optional reference frame for coordinate transformation. "
            "If provided, adds plane_* properties to detected objects."
        ),
    )


class FeatureTemplateMatchRequest(BaseModel):
    """Request for feature-based template matching using ORB"""

    image_id: str
    roi: Optional[ROI] = Field(None, description="Region of interest to limit search area")
    params: FeatureTemplateMatchParams = Field(
        description="Feature-based template matching parameters"
    )
    reference_object: Optional[ReferenceObject] = Field(
        None,
        description=(
            "Optional reference frame for coordinate transformation. "
            "If provided, adds plane_* properties to detected objects."
        ),
    )


class EdgeDetectRequest(BaseModel):
    """Request for edge detection"""

    image_id: str
    roi: Optional[ROI] = Field(None, description="Region of interest to limit search area")
    params: Optional[EdgeDetectionParams] = Field(
        None,
        description=("Edge detection parameters (method, filtering, preprocessing)"),
    )


class ColorDetectRequest(BaseModel):
    """Request for color detection"""

    image_id: str = Field(..., description="ID of the image to analyze")
    roi: Optional[ROI] = Field(None, description="Region of interest (if None, analyze full image)")
    expected_color: Optional[str] = Field(
        None,
        description=("Expected color name (red, blue, green, etc.) " "or None to just detect"),
    )
    contour: Optional[List] = Field(
        None, description="Contour points for masking (from edge detection)"
    )
    params: Optional[ColorDetectionParams] = Field(
        None,
        description=(
            "Color detection parameters (method, thresholds, "
            "kmeans settings, defaults applied if None)"
        ),
    )
    reference_object: Optional[ReferenceObject] = Field(
        None,
        description=(
            "Optional reference frame for coordinate transformation. "
            "If provided, adds plane_* properties to detected objects."
        ),
    )


class ArucoDetectRequest(BaseModel):
    """Request for ArUco marker detection"""

    image_id: str = Field(..., description="ID of the image to analyze")
    roi: Optional[ROI] = Field(None, description="Region of interest to search in")
    params: Optional[ArucoDetectionParams] = Field(
        None,
        description=(
            "ArUco detection parameters (dictionary type, "
            "detector settings, defaults applied if None)"
        ),
    )


class ArucoReferenceRequest(BaseModel):
    """Request for ArUco reference frame creation"""

    image_id: str = Field(..., description="ID of the image to analyze")
    roi: Optional[ROI] = Field(None, description="Region of interest to search in")
    params: ArucoReferenceParams = Field(
        description=(
            "ArUco reference parameters (mode, dictionary, marker configuration). "
            "Mode determines if single marker or 4-marker plane is used."
        ),
    )


class RotationDetectRequest(BaseModel):
    """Request for rotation detection"""

    image_id: str = Field(..., description="ID of the image for visualization")
    contour: List = Field(
        ...,
        description="Contour points [[x1,y1], [x2,y2], ...] (minimum 5 points required)",
    )
    roi: Optional[ROI] = Field(None, description="Optional ROI for visualization context")
    params: Optional[RotationDetectionParams] = Field(
        None,
        description=(
            "Rotation detection parameters " "(method, angle range, defaults applied if None)"
        ),
    )
    reference_object: Optional[ReferenceObject] = Field(
        None,
        description=(
            "Optional reference frame for coordinate transformation. "
            "If provided, adds plane_* properties to detected objects."
        ),
    )

    @validator("contour")
    def validate_contour(cls, v):
        """Validate contour has minimum required points."""
        if len(v) < 5:
            raise ValueError(
                f"Contour must have at least 5 points for rotation detection, got {len(v)}"
            )
        return v


# ==============================================================================
# Test Image Models
# ==============================================================================


class TestImageInfo(BaseModel):
    """Test image information"""

    id: str
    filename: str
    original_filename: str
    size: Size
    created_at: datetime

    @classmethod
    def from_manager_dict(cls, data: Dict[str, Any]) -> "TestImageInfo":
        """Create from manager dictionary"""
        return cls(
            id=data["id"],
            filename=data["filename"],
            original_filename=data["original_filename"],
            size=Size(**data["size"]),
            created_at=datetime.fromisoformat(data["created_at"]),
        )


class TestImageUploadResponse(BaseModel):
    """Response from test image upload"""

    success: bool
    test_id: str
    filename: str
    size: Size


class TestImageCaptureResponse(VisionResponse):
    """Response from test image capture endpoint (same as camera capture)"""

    test_id: Optional[str] = Field(None, description="Test image ID that was captured")
