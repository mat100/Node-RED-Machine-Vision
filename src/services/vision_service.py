"""
Vision Service - Business logic for vision processing operations.

This service orchestrates vision processing operations including
template matching, edge detection, and other computer vision tasks.
"""

import logging
from typing import Dict, List, Optional, Tuple

from api.exceptions import ImageNotFoundException, TemplateNotFoundException
from core.image import extract_roi
from core.image_manager import ImageManager
from core.template_manager import TemplateManager
from core.utils import enum_to_string, parse_enum, timer
from schemas import (
    ROI,
    ArucoDetectionParams,
    ColorDetectionParams,
    EdgeDetectionParams,
    RotationDetectionParams,
    TemplateMatchParams,
    VisionObject,
)
from vision.aruco_detection import ArucoDetector
from vision.color_detection import ColorDetector
from vision.rotation_detection import RotationDetector

logger = logging.getLogger(__name__)


class VisionService:
    """
    Service for vision processing operations.

    This service combines template matching, edge detection, and other
    vision algorithms with image management and history tracking.
    """

    def __init__(
        self,
        image_manager: ImageManager,
        template_manager: TemplateManager,
    ):
        """
        Initialize vision service.

        Args:
            image_manager: Image manager instance
            template_manager: Template manager instance
        """
        self.image_manager = image_manager
        self.template_manager = template_manager
        self.color_detector = ColorDetector()
        self.aruco_detector = ArucoDetector()
        self.rotation_detector = RotationDetector()

    @staticmethod
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
        self,
        image_id: str,
        detector_func,
        roi: Optional[Dict] = None,
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
            detector_func: Detection function to execute
                (receives image, returns result dict/objects)
            roi: Optional region of interest
            **detector_kwargs: Additional kwargs passed to detector function

        Returns:
            Tuple of (detection_result, thumbnail_base64, processing_time_ms)
        """
        with timer() as t:
            # Get image
            full_image = self.image_manager.get(image_id)
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
            self._adjust_for_roi_offset(result["objects"], roi_offset)

            # Generate thumbnail (inline - used only here)
            _, thumbnail_base64 = self.image_manager.create_thumbnail(result["image"])

        # Read processing time AFTER with block (timer updates in finally)
        processing_time_ms = t["ms"]

        return result, thumbnail_base64, processing_time_ms

    def template_match(
        self,
        image_id: str,
        template_id: str,
        roi: Optional[Dict],
        params: TemplateMatchParams,
    ) -> Tuple[List[VisionObject], str, int]:
        """
        Perform template matching on an image.

        Args:
            image_id: Image identifier
            template_id: Template identifier (extracted from params for
                backward compat in service layer)
            roi: Optional region of interest to limit search area
                (dict with x, y, width, height)
            params: Template matching parameters (includes template_id)

        Returns:
            Tuple of (detected_objects, thumbnail_base64, processing_time_ms)

        Raises:
            ImageNotFoundException: If image not found
            TemplateNotFoundException: If template not found
        """
        from vision.template_matching import TemplateDetector

        # Get template first (before detector function)
        template = self.template_manager.get_template(template_id)
        if template is None:
            raise TemplateNotFoundException(template_id)

        # Convert params to dict (all defaults already applied by Pydantic!)
        params_dict = params.to_dict()

        # Create detector function
        def detect_func(image):
            detector = TemplateDetector()
            return detector.detect(
                image=image, template=template, template_id=template_id, params=params_dict
            )

        # Execute using template method - it handles ROI, coordinate adjustment, thumbnail
        result, thumbnail_base64, processing_time = self._execute_detection(
            image_id=image_id,
            detector_func=detect_func,
            roi=roi,
        )

        logger.debug(f"Template matching: {len(result['objects'])} matches in {processing_time}ms")

        return result["objects"], thumbnail_base64, processing_time

    def learn_template_from_roi(
        self, image_id: str, roi: ROI, name: str, description: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Learn a template from an image region.

        Args:
            image_id: Source image identifier
            roi: ROI to extract as template
            name: Template name
            description: Optional description

        Returns:
            Tuple of (template_id, thumbnail_base64)

        Raises:
            ImageNotFoundException: If image not found
            ValueError: If ROI is invalid
        """
        # Get source image
        source_image = self.image_manager.get(image_id)
        if source_image is None:
            raise ImageNotFoundException(image_id)

        # Validate ROI
        img_height, img_width = source_image.shape[:2]
        roi.validate_with_constraints(image_width=img_width, image_height=img_height)

        # Learn template
        template_id = self.template_manager.learn_template(
            name=name, source_image=source_image, roi=roi.to_dict(), description=description
        )

        # Get thumbnail
        thumbnail_base64 = self.template_manager.create_template_thumbnail(template_id)

        logger.info(f"Template learned from ROI: {template_id}")
        return template_id, thumbnail_base64

    def edge_detect(
        self,
        image_id: str,
        method: str,
        params: "EdgeDetectionParams",
        roi: Optional[Dict] = None,
    ) -> Tuple[List[VisionObject], str, int]:
        """
        Perform edge detection on an image.

        Args:
            image_id: Image identifier
            method: Edge detection method (extracted from params for
                backward compat in service layer)
            params: Edge detection parameters (includes method)
            roi: Optional region of interest to limit detection area

        Returns:
            Tuple of (detected_objects, thumbnail_base64, processing_time_ms)

        Raises:
            ImageNotFoundException: If image not found
        """
        from core.enums import EdgeMethod
        from vision.edge_detection import EdgeDetector

        # Parse method string to enum
        edge_method = parse_enum(method, EdgeMethod, EdgeMethod.CANNY, normalize=True)

        # Convert params to dict (all defaults already applied by Pydantic!)
        params_dict = params.to_dict()

        # Create detector function
        def detect_func(image):
            detector = EdgeDetector()
            return detector.detect(image=image, method=edge_method, params=params_dict)

        # Execute using template method
        result, thumbnail_base64, processing_time = self._execute_detection(
            image_id=image_id,
            detector_func=detect_func,
            roi=roi,
        )

        logger.debug(
            f"Edge detection completed: {len(result['objects'])} contours "
            f"found in {processing_time}ms"
        )

        return result["objects"], thumbnail_base64, processing_time

    def color_detect(
        self,
        image_id: str,
        roi: Optional[Dict[str, int]],
        contour: Optional[list],
        expected_color: Optional[str],
        params: Optional[ColorDetectionParams],
    ) -> Tuple[List[VisionObject], str, int]:
        """
        Perform color detection on an image.

        Args:
            image_id: Image identifier
            roi: Optional region of interest {x, y, width, height}
            contour: Optional contour points for masking
            expected_color: Expected color name (or None to just detect)
            params: Color detection parameters (validated Pydantic model, or None for defaults)

        Returns:
            Tuple of (detected_objects, thumbnail_base64, processing_time_ms)

        Raises:
            ImageNotFoundException: If image not found
        """
        # Create default params if not provided
        if params is None:
            params = ColorDetectionParams()

        # Extract individual params for ColorDetector.detect()
        method_str = enum_to_string(params.method)
        use_contour_mask = params.use_contour_mask
        min_percentage = params.min_percentage

        # Create detector function
        def detect_func(image):
            return self.color_detector.detect(
                image=image,
                roi=roi,
                contour_points=contour,
                use_contour_mask=use_contour_mask,
                expected_color=expected_color,
                min_percentage=min_percentage,
                method=method_str,
            )

        # Execute using template method (handles image retrieval, thumbnail, timing)
        result, thumbnail_base64, processing_time = self._execute_detection(
            image_id=image_id,
            detector_func=detect_func,
            roi=None,  # ColorDetector handles ROI internally
        )

        detected_object = result["objects"][0]

        logger.debug(
            f"Color detection completed: {detected_object.properties['dominant_color']} "
            f"({detected_object.confidence*100:.1f}%) in {processing_time}ms"
        )

        # Filter result: if expected_color specified and doesn't match, return empty list
        # This business logic belongs in service layer, not router
        if expected_color is not None:
            is_match = detected_object.properties.get("match", False)
            if not is_match:
                return [], thumbnail_base64, processing_time

        return result["objects"], thumbnail_base64, processing_time

    def aruco_detect(
        self,
        image_id: str,
        roi: Optional[Dict],
        params: Optional[ArucoDetectionParams],
    ) -> Tuple[List[VisionObject], str, int]:
        """
        Detect ArUco markers in image.

        Args:
            image_id: ID of the image to process
            roi: Optional region of interest to search in (dict with x, y, width, height)
            params: ArUco detection parameters (validated Pydantic model, or None for defaults)

        Returns:
            (detected_objects, thumbnail_base64, processing_time_ms)

        Raises:
            ImageNotFoundException: If image not found
        """
        # Create default params if not provided
        if params is None:
            params = ArucoDetectionParams()

        # Convert params to dict (all defaults already applied by Pydantic!)
        params_dict = params.to_dict()

        # Extract dictionary parameter
        dictionary_str = enum_to_string(params.dictionary)

        # Create detector function
        def detect_func(image):
            return self.aruco_detector.detect(image, dictionary=dictionary_str, params=params_dict)

        # Execute using template method
        result, thumbnail_base64, processing_time = self._execute_detection(
            image_id=image_id,
            detector_func=detect_func,
            roi=roi,
        )

        logger.debug(
            f"ArUco detection completed: {len(result['objects'])} markers "
            f"in {processing_time}ms"
        )

        return result["objects"], thumbnail_base64, processing_time

    def rotation_detect(
        self,
        image_id: str,
        contour: List,
        roi: Optional[Dict[str, int]],
        params: Optional[RotationDetectionParams],
    ) -> Tuple[List[VisionObject], str, int]:
        """
        Detect rotation angle from contour.

        Args:
            image_id: ID of the image (for visualization)
            contour: Contour points [[x1,y1], [x2,y2], ...]
            roi: Optional ROI for visualization context
            params: Rotation detection parameters (validated Pydantic model, or None for defaults)

        Returns:
            (detected_objects, thumbnail_base64, processing_time_ms)

        Raises:
            ImageNotFoundException: If image not found
        """
        # Create default params if not provided
        if params is None:
            params = RotationDetectionParams()

        # Extract params (all defaults already applied by Pydantic!)
        method_enum = params.method
        range_enum = params.angle_range

        # Create detector function
        def detect_func(image):
            return self.rotation_detector.detect(
                image, contour=contour, method=method_enum, angle_range=range_enum, roi=roi
            )

        # Execute using template method (handles image retrieval, thumbnail, timing)
        result, thumbnail_base64, processing_time = self._execute_detection(
            image_id=image_id,
            detector_func=detect_func,
            roi=None,  # RotationDetector handles ROI for visualization
        )

        detected_object = result["objects"][0]

        logger.debug(
            f"Rotation detection completed: {detected_object.rotation:.1f}° "
            f"({params.method.value}) in {processing_time}ms"
        )

        return result["objects"], thumbnail_base64, processing_time
