"""
Vision API Router - Vision processing endpoints

All vision detection endpoints follow a unified pattern:
1. Validate request (image exists, ROI valid)
2. Execute vision algorithm (edge detection, color detection, etc.)
3. Return VisionResponse

This consistency reduces code duplication and makes the API predictable.
"""

import logging
from typing import List, Optional, Tuple

from fastapi import APIRouter, Depends

from algorithms.advanced_template_matching import AdvancedTemplateDetector
from algorithms.aruco_detection import ArucoDetector
from algorithms.color_detection import ColorDetector
from algorithms.edge_detection import EdgeDetector
from algorithms.rotation_detection import RotationDetector
from algorithms.template_matching import TemplateDetector
from dependencies import get_image_manager, get_template_manager, validate_vision_request
from domain_types import ROI, EdgeMethod
from exceptions import ImageNotFoundException, TemplateNotFoundException, safe_endpoint
from image.roi import extract_roi
from image.transform import apply_reference_transform_batch
from managers.image_manager import ImageManager
from managers.template_manager import TemplateManager
from models import (
    AdvancedTemplateMatchRequest,
    ArucoDetectRequest,
    ArucoReferenceRequest,
    ArucoReferenceResponse,
    ColorDetectRequest,
    EdgeDetectRequest,
    RotationDetectRequest,
    TemplateMatchRequest,
    VisionObject,
    VisionResponse,
)
from utils import enum_to_string, parse_enum, timer

logger = logging.getLogger(__name__)

router = APIRouter()


def _adjust_for_roi_offset(objects: List[VisionObject], roi_offset: Tuple[int, int]) -> None:
    """
    Adjust object coordinates to account for ROI offset.

    Modifies VisionObject instances in-place by adding the ROI offset
    to all coordinate values (bounding_box, center, contour points).

    Args:
        objects: List of VisionObject instances to adjust
        roi_offset: Tuple of (x_offset, y_offset) from ROI position
    """
    if roi_offset == (0, 0):
        return  # No adjustment needed

    x_offset, y_offset = roi_offset

    for obj in objects:
        # Adjust bounding box position
        obj.bounding_box.x += x_offset
        obj.bounding_box.y += y_offset

        # Adjust center point
        obj.center.x += x_offset
        obj.center.y += y_offset

        # Adjust contour points if present
        if hasattr(obj, "contour") and obj.contour:
            obj.contour = [[x + x_offset, y + y_offset] for x, y in obj.contour]


def _execute_detection(
    image_id: str,
    image_manager: ImageManager,
    detector_func,
    roi: Optional[dict] = None,
    **detector_kwargs,
) -> Tuple[any, str, int]:
    """
    Template method for vision detection operations.

    This method encapsulates common logic:
    - Image retrieval
    - ROI extraction
    - Coordinate adjustment
    - Thumbnail generation

    Args:
        image_id: Image identifier
        image_manager: ImageManager instance
        detector_func: Detection function to execute
            (receives image, returns result dict/objects)
        roi: Optional region of interest
        **detector_kwargs: Additional kwargs passed to detector function

    Returns:
        Tuple of (detection_result, thumbnail_base64, processing_time_ms)
    """
    with timer() as t:
        # Get image
        full_image = image_manager.get(image_id)
        if full_image is None:
            raise ImageNotFoundException(image_id)

        # Extract ROI if specified
        roi_offset = (0, 0)
        if roi:
            roi_obj = ROI.from_dict(roi) if isinstance(roi, dict) else roi
            image = extract_roi(full_image, roi_obj, safe_mode=True)
            roi_offset = (roi_obj.x, roi_obj.y)
        else:
            image = full_image

        # Execute detection
        result = detector_func(image, **detector_kwargs)

        # Adjust coordinates if ROI was used
        _adjust_for_roi_offset(result["objects"], roi_offset)

        # Generate thumbnail (inline - used only here)
        _, thumbnail_base64 = image_manager.create_thumbnail(result["image"])

    # Read processing time AFTER with block (timer updates in finally)
    processing_time_ms = t["ms"]

    return result, thumbnail_base64, processing_time_ms


@router.post("/template-match")
@safe_endpoint
async def template_match(
    request: TemplateMatchRequest,
    image_manager: ImageManager = Depends(get_image_manager),
    template_manager: TemplateManager = Depends(get_template_manager),
) -> VisionResponse:
    """
    Perform template matching on an image.

    INPUT constraints:
    - roi: Optional region to limit template search area

    OUTPUT results:
    - bounding_box: Location where template was found
    """
    # Validate request
    roi_dict = validate_vision_request(request.image_id, request.roi, image_manager)

    # Get template and mask
    template_id = request.params.template_id
    template = template_manager.get_template(template_id)
    if template is None:
        raise TemplateNotFoundException(template_id)

    mask = template_manager.get_template_mask(template_id)

    # Convert params to dict
    params_dict = request.params.to_dict()

    # Create detector function
    def detect_func(image):
        detector = TemplateDetector()
        return detector.detect(
            image=image, template=template, template_id=template_id, params=params_dict, mask=mask
        )

    # Execute detection using helper
    result, thumbnail_base64, processing_time = _execute_detection(
        image_id=request.image_id,
        image_manager=image_manager,
        detector_func=detect_func,
        roi=roi_dict,
    )

    # Apply reference frame transformation if provided
    if request.reference_object is not None:
        result["objects"] = apply_reference_transform_batch(
            result["objects"], request.reference_object
        )

    logger.debug(
        f"Template matching: {len(result['objects'])} matches in {processing_time}ms "
        f"(reference={request.reference_object is not None})"
    )

    return VisionResponse(
        objects=result["objects"],
        thumbnail_base64=thumbnail_base64,
        processing_time_ms=processing_time,
    )


@router.post("/advanced-template-match")
@safe_endpoint
async def advanced_template_match(
    request: AdvancedTemplateMatchRequest,
    image_manager: ImageManager = Depends(get_image_manager),
    template_manager: TemplateManager = Depends(get_template_manager),
) -> VisionResponse:
    """
    Perform advanced template matching with rotation and multi-instance support.

    Features:
    - Multi-instance detection: Find all matches above threshold with NMS filtering
    - Rotation-invariant matching: Search across configurable angle ranges
    - Overlap filtering: Remove duplicate detections using IoU threshold

    INPUT constraints:
    - roi: Optional region to limit template search area
    - params.find_multiple: Enable multi-instance detection
    - params.enable_rotation: Enable rotation-invariant matching
    - params.rotation_range: Angle range for rotation search
    - params.rotation_step: Rotation step size (smaller = more accurate but slower)

    OUTPUT results:
    - bounding_box: Location where template was found
    - rotation: Rotation angle in degrees (if enable_rotation=True)
    - confidence: Match confidence score (0.0 to 1.0)
    """
    # Validate request
    roi_dict = validate_vision_request(request.image_id, request.roi, image_manager)

    # Get template and mask
    template_id = request.params.template_id
    template = template_manager.get_template(template_id)
    if template is None:
        raise TemplateNotFoundException(template_id)

    mask = template_manager.get_template_mask(template_id)

    # Convert params to dict
    params_dict = request.params.to_dict()

    # Create detector function
    def detect_func(image):
        detector = AdvancedTemplateDetector()
        return detector.detect(
            image=image, template=template, template_id=template_id, params=params_dict, mask=mask
        )

    # Execute detection using helper
    result, thumbnail_base64, processing_time = _execute_detection(
        image_id=request.image_id,
        image_manager=image_manager,
        detector_func=detect_func,
        roi=roi_dict,
    )

    # Apply reference frame transformation if provided
    if request.reference_object is not None:
        result["objects"] = apply_reference_transform_batch(
            result["objects"], request.reference_object
        )

    logger.debug(
        f"Advanced template matching: {len(result['objects'])} matches in {processing_time}ms "
        f"(rotation={request.params.enable_rotation}, "
        f"multi={request.params.find_multiple}, "
        f"reference={request.reference_object is not None})"
    )

    return VisionResponse(
        objects=result["objects"],
        thumbnail_base64=thumbnail_base64,
        processing_time_ms=processing_time,
    )


@router.post("/edge-detect")
@safe_endpoint
async def edge_detect(
    request: EdgeDetectRequest,
    image_manager: ImageManager = Depends(get_image_manager),
) -> VisionResponse:
    """
    Perform edge detection with multiple methods.

    INPUT constraints:
    - roi: Optional region to limit edge detection area

    OUTPUT results:
    - bounding_box: Bounding box of detected contour
    - contour: Actual contour points for precise shape representation
    """
    # Validate request
    roi_dict = validate_vision_request(request.image_id, request.roi, image_manager)

    # Parse method string to enum
    edge_method = parse_enum(request.params.method, EdgeMethod, EdgeMethod.CANNY, normalize=True)

    # Convert params to dict
    params_dict = request.params.to_dict()

    # Create detector function
    def detect_func(image):
        detector = EdgeDetector()
        return detector.detect(image=image, method=edge_method, params=params_dict)

    # Execute detection using helper
    result, thumbnail_base64, processing_time = _execute_detection(
        image_id=request.image_id,
        image_manager=image_manager,
        detector_func=detect_func,
        roi=roi_dict,
    )

    logger.debug(
        f"Edge detection completed: {len(result['objects'])} contours "
        f"found in {processing_time}ms"
    )

    return VisionResponse(
        objects=result["objects"],
        thumbnail_base64=thumbnail_base64,
        processing_time_ms=processing_time,
    )


@router.post("/color-detect")
@safe_endpoint
async def color_detect(
    request: ColorDetectRequest,
    image_manager: ImageManager = Depends(get_image_manager),
) -> VisionResponse:
    """
    Perform color detection with automatic dominant color recognition.

    INPUT constraints:
    - roi: Optional region for color analysis
    - contour: Optional contour points for precise masking (auto-used from msg.payload)

    OUTPUT results:
    - bounding_box: Region where color was analyzed
    """
    # Validate request
    roi_dict = validate_vision_request(request.image_id, request.roi, image_manager)

    # Create default params if not provided
    from models import ColorDetectionParams

    params = request.params if request.params else ColorDetectionParams()

    # Extract individual params for ColorDetector.detect()
    method_str = enum_to_string(params.method)
    use_contour_mask = params.use_contour_mask
    min_percentage = params.min_percentage

    # Create detector instance
    color_detector = ColorDetector()

    # Create detector function
    def detect_func(image):
        return color_detector.detect(
            image=image,
            roi=roi_dict,
            contour_points=request.contour,
            use_contour_mask=use_contour_mask,
            expected_color=request.expected_color,
            min_percentage=min_percentage,
            method=method_str,
        )

    # Execute detection using helper (ROI handled by ColorDetector internally)
    result, thumbnail_base64, processing_time = _execute_detection(
        image_id=request.image_id,
        image_manager=image_manager,
        detector_func=detect_func,
        roi=None,  # ColorDetector handles ROI internally
    )

    # Apply reference frame transformation if provided
    if request.reference_object is not None:
        result["objects"] = apply_reference_transform_batch(
            result["objects"], request.reference_object
        )

    detected_object = result["objects"][0]

    logger.debug(
        f"Color detection completed: {detected_object.properties['dominant_color']} "
        f"({detected_object.confidence*100:.1f}%) in {processing_time}ms "
        f"(reference={request.reference_object is not None})"
    )

    # Filter result: if expected_color specified and doesn't match, return empty list
    if request.expected_color is not None:
        is_match = detected_object.properties.get("match", False)
        if not is_match:
            return VisionResponse(
                objects=[],
                thumbnail_base64=thumbnail_base64,
                processing_time_ms=processing_time,
            )

    return VisionResponse(
        objects=result["objects"],
        thumbnail_base64=thumbnail_base64,
        processing_time_ms=processing_time,
    )


@router.post("/aruco-detect")
@safe_endpoint
async def aruco_detect(
    request: ArucoDetectRequest,
    image_manager: ImageManager = Depends(get_image_manager),
) -> VisionResponse:
    """
    Detect ArUco fiducial markers in image (MARKERS mode only).

    Detects all visible ArUco markers without creating a reference frame.
    For reference frame creation, use /aruco-reference endpoint.

    INPUT constraints:
    - roi: Optional region to limit marker search area

    OUTPUT results:
    - bounding_box: Bounding box of detected marker
    - properties.marker_id: Unique marker ID
    - properties.corners: 4 corner points
    - rotation: Marker rotation in degrees (0-360)
    """
    # Validate request
    roi_dict = validate_vision_request(request.image_id, request.roi, image_manager)

    # Create default params if not provided
    from models import ArucoDetectionParams

    params = request.params if request.params else ArucoDetectionParams()

    # Convert params to dict
    params_dict = params.to_dict()

    # Extract parameters for detector
    dictionary_str = enum_to_string(params.dictionary)

    # Force MARKERS mode
    mode = "markers"

    # Create detector instance
    aruco_detector = ArucoDetector(image_manager=image_manager)

    # Create detector function
    def detect_func(image):
        return aruco_detector.detect(
            image,
            dictionary=dictionary_str,
            mode=mode,
            single_config=None,
            plane_config=None,
            params=params_dict,
        )

    # Execute detection using helper
    result, thumbnail_base64, processing_time = _execute_detection(
        image_id=request.image_id,
        image_manager=image_manager,
        detector_func=detect_func,
        roi=roi_dict,
    )

    logger.debug(
        f"ArUco detection completed: {len(result['objects'])} markers "
        f"in {processing_time}ms (mode=markers)"
    )

    return VisionResponse(
        objects=result["objects"],
        thumbnail_base64=thumbnail_base64,
        processing_time_ms=processing_time,
    )


@router.post("/aruco-reference")
@safe_endpoint
async def aruco_reference(
    request: ArucoReferenceRequest,
    image_manager: ImageManager = Depends(get_image_manager),
) -> ArucoReferenceResponse:
    """
    Create reference frame from ArUco markers (SINGLE or PLANE mode).

    Creates a coordinate transformation reference frame from ArUco markers
    for converting pixel coordinates to real-world units (millimeters).

    MODES:
    - SINGLE: One marker with known size → affine transform (uniform scaling)
    - PLANE: Four markers at corners → perspective homography transform

    INPUT constraints:
    - roi: Optional region to limit marker search area
    - params.mode: Required (single or plane)
    - params.single_config: Required if mode=single (marker_id, marker_size_mm)
    - params.plane_config: Required if mode=plane (marker_ids, width_mm, height_mm)

    OUTPUT results:
    - reference_object: Reference frame with homography matrix and metadata
    - markers: Detected ArUco markers used for calibration
    - thumbnail: Visualization with markers highlighted
    """
    # Validate request
    roi_dict = validate_vision_request(request.image_id, request.roi, image_manager)

    # Convert params to dict
    params_dict = request.params.to_dict()

    # Extract parameters for detector
    dictionary_str = enum_to_string(request.params.dictionary)
    mode = enum_to_string(request.params.mode)

    # Convert Pydantic models to dicts for detector
    single_config_dict = (
        request.params.single_config.dict() if request.params.single_config else None
    )
    plane_config_dict = request.params.plane_config.dict() if request.params.plane_config else None

    # Create detector instance
    aruco_detector = ArucoDetector(image_manager=image_manager)

    # Create detector function
    def detect_func(image):
        return aruco_detector.detect(
            image,
            dictionary=dictionary_str,
            mode=mode,
            single_config=single_config_dict,
            plane_config=plane_config_dict,
            params=params_dict,
        )

    # Execute detection using helper
    result, thumbnail_base64, processing_time = _execute_detection(
        image_id=request.image_id,
        image_manager=image_manager,
        detector_func=detect_func,
        roi=roi_dict,
    )

    # Extract reference_object (must be present for SINGLE/PLANE modes)
    reference_object = result.get("reference_object")
    if reference_object is None:
        raise ValueError(
            f"Failed to create reference frame in {mode} mode. "
            "Check that required markers are visible and correctly configured."
        )

    markers = result["objects"]

    logger.debug(
        f"ArUco reference created: {len(markers)} markers detected, "
        f"reference type={reference_object.type} in {processing_time}ms"
    )

    # Return specialized response
    return ArucoReferenceResponse(
        reference_object=reference_object,
        markers=markers,
        thumbnail_base64=thumbnail_base64,
        processing_time_ms=processing_time,
    )


@router.post("/rotation-detect")
@safe_endpoint
async def rotation_detect(
    request: RotationDetectRequest,
    image_manager: ImageManager = Depends(get_image_manager),
) -> VisionResponse:
    """
    Detect rotation angle from contour points.

    INPUT constraints:
    - roi: Optional ROI for visualization context only (not a constraint)
    - contour: Required contour points from edge detection

    OUTPUT results:
    - rotation: Calculated rotation in degrees
    """
    # Validate request
    roi_dict = validate_vision_request(request.image_id, request.roi, image_manager)

    # Create default params if not provided
    from models import RotationDetectionParams

    params = request.params if request.params else RotationDetectionParams()

    # Extract params
    method_enum = params.method
    range_enum = params.angle_range
    asymmetry_enum = params.asymmetry_orientation

    # Create detector instance
    rotation_detector = RotationDetector()

    # Create detector function
    def detect_func(image):
        return rotation_detector.detect(
            image,
            contour=request.contour,
            method=method_enum,
            angle_range=range_enum,
            asymmetry_orientation=asymmetry_enum,
            roi=roi_dict,
        )

    # Execute detection using helper (ROI handled by RotationDetector for visualization)
    result, thumbnail_base64, processing_time = _execute_detection(
        image_id=request.image_id,
        image_manager=image_manager,
        detector_func=detect_func,
        roi=None,  # RotationDetector handles ROI for visualization
    )

    # Apply reference frame transformation if provided
    if request.reference_object is not None:
        result["objects"] = apply_reference_transform_batch(
            result["objects"], request.reference_object
        )

    detected_object = result["objects"][0]

    logger.debug(
        f"Rotation detection completed: {detected_object.rotation:.1f}° "
        f"({params.method.value}) in {processing_time}ms "
        f"(reference={request.reference_object is not None})"
    )

    return VisionResponse(
        objects=result["objects"],
        thumbnail_base64=thumbnail_base64,
        processing_time_ms=processing_time,
    )
