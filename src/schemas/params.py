"""
Detection parameters for all vision algorithms.

Centralized location for all Pydantic parameter classes used across
vision detection modules. This module eliminates circular dependencies
by keeping all parameter schemas in the schemas/ package.
"""

from pydantic import Field

from core.enums import AngleRange, ArucoDict, ColorMethod, RotationMethod, TemplateMethod
from schemas.base import BaseDetectionParams


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
    ArUco marker detection parameters.

    ArUco markers are fiducial markers used for camera calibration,
    object tracking, and pose estimation.
    """

    dictionary: ArucoDict = Field(
        default=ArucoDict.DICT_4X4_50,
        description="ArUco dictionary type (defines marker set and size)",
    )
    # Future: můžeme přidat detector params:
    # adaptive_thresh_constant: float = Field(default=7.0, ge=0)
    # min_marker_perimeter_rate: float = Field(default=0.03, ge=0, le=1)
    # ...


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
