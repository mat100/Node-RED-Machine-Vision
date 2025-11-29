"""
Advanced template matching with rotation and multi-instance support.

Provides enhanced template matching capabilities:
- Multi-instance detection with Non-Maximum Suppression (NMS)
- Rotation-invariant matching across configurable angle ranges
- Overlap filtering for duplicate removal
"""

import logging
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from algorithms.base_detector import BaseDetector
from domain_types import ROI, Point, VisionObjectType
from image.overlay import render_template_matches
from models import VisionObject

logger = logging.getLogger(__name__)


class AdvancedTemplateDetector(BaseDetector):
    """Advanced template matching with rotation and multi-instance support."""

    def __init__(self):
        """Initialize advanced template detector."""
        super().__init__()

    def detect(
        self,
        image: np.ndarray,
        template: np.ndarray,
        template_id: str,
        params: Dict[str, Any],
        mask: np.ndarray = None,
    ) -> Dict:
        """
        Perform advanced template matching with rotation and multi-instance support.

        Args:
            image: Input image (BGR format)
            template: Template image to search for
            template_id: Template identifier for metadata
            params: Detection parameters dict with keys:
                - method: OpenCV matching method
                - threshold: Match confidence threshold
                - find_multiple: Enable multi-instance detection
                - max_matches: Maximum number of matches
                - overlap_threshold: IoU threshold for NMS
                - enable_rotation: Enable rotation search
                - rotation_range: (min, max) angle range in degrees
                - rotation_step: Rotation step size in degrees
            mask: Optional mask (alpha channel) for template matching

        Returns:
            Dictionary with detection results
        """
        # Extract params
        method = params.get("method", "TM_CCOEFF_NORMED")
        threshold = params.get("threshold", 0.8)
        find_multiple = params.get("find_multiple", False)
        max_matches = params.get("max_matches", 10)
        overlap_threshold = params.get("overlap_threshold", 0.3)
        enable_rotation = params.get("enable_rotation", False)
        rotation_range = params.get("rotation_range", (-180.0, 180.0))
        rotation_step = params.get("rotation_step", 10.0)

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            search_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            search_gray = image

        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template

        # Perform matching (with or without rotation)
        if enable_rotation:
            detected_objects = self._match_with_rotation(
                search_gray,
                template_gray,
                template_id,
                method,
                threshold,
                rotation_range,
                rotation_step,
                find_multiple,
                max_matches,
                overlap_threshold,
                mask,
            )
        else:
            # Standard matching (potentially multi-instance)
            detected_objects = self._match_template(
                search_gray,
                template_gray,
                template_id,
                method,
                threshold,
                find_multiple,
                max_matches,
                overlap_threshold,
                rotation_angle=0.0,
                mask=mask,
            )

        # Create visualization
        if detected_objects:
            result_image = render_template_matches(image, detected_objects)
        else:
            result_image = image.copy()

        return {
            "success": True,
            "objects": detected_objects,
            "image": result_image,
        }

    def _match_template(
        self,
        image: np.ndarray,
        template: np.ndarray,
        template_id: str,
        method: str,
        threshold: float,
        find_multiple: bool,
        max_matches: int,
        overlap_threshold: float,
        rotation_angle: float = 0.0,
        mask: np.ndarray = None,
    ) -> List[VisionObject]:
        """
        Perform template matching (single or multi-instance).

        Args:
            image: Grayscale search image
            template: Grayscale template image
            template_id: Template identifier
            method: OpenCV matching method
            threshold: Confidence threshold
            find_multiple: Whether to find multiple instances
            max_matches: Maximum number of matches
            overlap_threshold: IoU threshold for NMS
            rotation_angle: Rotation angle of template (for metadata)
            mask: Optional mask for template matching

        Returns:
            List of VisionObject instances
        """
        # Prepare mask for template matching if provided
        mask_for_matching = None
        if mask is not None:
            # Ensure mask is uint8
            if mask.dtype != np.uint8:
                mask_for_matching = mask.astype(np.uint8)
            else:
                mask_for_matching = mask

            # Ensure mask is same size as template
            if mask_for_matching.shape[:2] != template.shape[:2]:
                mask_for_matching = cv2.resize(
                    mask_for_matching,
                    (template.shape[1], template.shape[0]),
                )

        # Perform template matching
        cv_method = getattr(cv2, method)
        if mask_for_matching is not None:
            result = cv2.matchTemplate(image, template, cv_method, mask=mask_for_matching)
        else:
            result = cv2.matchTemplate(image, template, cv_method)

        # Get template dimensions
        h, w = template.shape[:2]

        detected_objects = []

        if find_multiple:
            # Find multiple instances using NMS
            matches = self._find_multiple_matches(
                result, method, threshold, w, h, max_matches, overlap_threshold
            )

            for idx, (x, y, score) in enumerate(matches):
                detected_objects.append(
                    self._create_vision_object(
                        x, y, w, h, score, rotation_angle, template_id, method, idx
                    )
                )
        else:
            # Find single best match
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # For SQDIFF methods, lower is better
            if method in ["TM_SQDIFF", "TM_SQDIFF_NORMED"]:
                if min_val <= (1 - threshold):
                    score = 1 - min_val
                    loc = min_loc
                else:
                    score = 0
                    loc = None
            else:
                if max_val >= threshold:
                    score = max_val
                    loc = max_loc
                else:
                    score = 0
                    loc = None

            if loc is not None:
                detected_objects.append(
                    self._create_vision_object(
                        loc[0], loc[1], w, h, score, rotation_angle, template_id, method, 0
                    )
                )

        return detected_objects

    def _match_with_rotation(
        self,
        image: np.ndarray,
        template: np.ndarray,
        template_id: str,
        method: str,
        threshold: float,
        rotation_range: Tuple[float, float],
        rotation_step: float,
        find_multiple: bool,
        max_matches: int,
        overlap_threshold: float,
        mask: np.ndarray = None,
    ) -> List[VisionObject]:
        """
        Perform template matching with rotation search.

        Args:
            image: Grayscale search image
            template: Grayscale template image
            template_id: Template identifier
            method: OpenCV matching method
            threshold: Confidence threshold
            rotation_range: (min, max) angle range in degrees
            rotation_step: Rotation step size in degrees
            find_multiple: Whether to find multiple instances
            max_matches: Maximum number of matches
            overlap_threshold: IoU threshold for NMS
            mask: Optional mask for template matching

        Returns:
            List of VisionObject instances with best rotation matches
        """
        min_angle, max_angle = rotation_range

        # Generate rotation angles
        angles = np.arange(min_angle, max_angle + rotation_step, rotation_step)

        # Store all matches across all rotations
        all_matches = []

        for angle in angles:
            # Rotate template
            rotated_template = self._rotate_image(template, angle)

            # Rotate mask if provided
            rotated_mask = None
            if mask is not None:
                rotated_mask = self._rotate_image(mask, angle)

            # Match with rotated template
            matches = self._match_template(
                image,
                rotated_template,
                template_id,
                method,
                threshold,
                find_multiple=True,  # Always find multiple to collect all candidates
                max_matches=max_matches * 2,  # Collect more candidates for NMS
                overlap_threshold=1.0,  # No NMS yet (we'll apply it globally)
                rotation_angle=angle,
                mask=rotated_mask,
            )

            all_matches.extend(matches)

        # Apply global NMS across all rotation candidates
        if all_matches:
            # Sort by confidence (descending)
            all_matches.sort(key=lambda obj: obj.confidence, reverse=True)

            # Apply NMS
            filtered_matches = self._apply_nms_to_objects(
                all_matches, overlap_threshold, max_matches
            )

            return filtered_matches
        else:
            return []

    def _find_multiple_matches(
        self,
        result: np.ndarray,
        method: str,
        threshold: float,
        template_width: int,
        template_height: int,
        max_matches: int,
        overlap_threshold: float,
    ) -> List[Tuple[int, int, float]]:
        """
        Find multiple template matches using Non-Maximum Suppression.

        Args:
            result: Template matching result matrix
            method: OpenCV matching method
            threshold: Confidence threshold
            template_width: Template width
            template_height: Template height
            max_matches: Maximum number of matches
            overlap_threshold: IoU threshold for NMS

        Returns:
            List of (x, y, score) tuples
        """
        # For SQDIFF methods, lower is better - invert the result
        if method in ["TM_SQDIFF", "TM_SQDIFF_NORMED"]:
            result = 1 - result
            comparison_op = np.greater_equal
        else:
            comparison_op = np.greater_equal

        # Find all matches above threshold
        locations = np.where(comparison_op(result, threshold))
        matches = []

        for pt in zip(*locations[::-1]):  # Switch x and y
            x, y = pt
            score = float(result[y, x])

            # For SQDIFF methods, score is already inverted
            if method in ["TM_SQDIFF", "TM_SQDIFF_NORMED"]:
                score = min(score, 1.0)

            matches.append((x, y, score))

        # Apply Non-Maximum Suppression
        if len(matches) > 0:
            matches = self._apply_nms(
                matches, template_width, template_height, overlap_threshold, max_matches
            )

        return matches

    def _apply_nms(
        self,
        matches: List[Tuple[int, int, float]],
        width: int,
        height: int,
        overlap_threshold: float,
        max_matches: int,
    ) -> List[Tuple[int, int, float]]:
        """
        Apply Non-Maximum Suppression to remove overlapping detections.

        Args:
            matches: List of (x, y, score) tuples
            width: Template width
            height: Template height
            overlap_threshold: IoU threshold (0.0 = no overlap, 1.0 = full overlap)
            max_matches: Maximum number of matches to return

        Returns:
            Filtered list of (x, y, score) tuples
        """
        if len(matches) == 0:
            return []

        # Sort by score (descending)
        matches = sorted(matches, key=lambda m: m[2], reverse=True)

        # Convert to bounding boxes for NMS
        boxes = []
        for x, y, score in matches:
            boxes.append([x, y, x + width, y + height, score])

        boxes = np.array(boxes)

        # Apply NMS
        keep_indices = []
        while len(boxes) > 0 and len(keep_indices) < max_matches:
            # Pick box with highest score
            keep_indices.append(len(matches) - len(boxes))
            if len(boxes) == 1:
                break

            # Calculate IoU with remaining boxes
            current_box = boxes[0]
            remaining_boxes = boxes[1:]

            ious = self._calculate_iou_vectorized(current_box[:4], remaining_boxes[:, :4])

            # Keep only boxes with IoU below threshold
            keep_mask = ious < overlap_threshold
            boxes = remaining_boxes[keep_mask]

        return [matches[i] for i in keep_indices]

    def _apply_nms_to_objects(
        self,
        objects: List[VisionObject],
        overlap_threshold: float,
        max_matches: int,
    ) -> List[VisionObject]:
        """
        Apply NMS to VisionObject list.

        Args:
            objects: List of VisionObject instances
            overlap_threshold: IoU threshold
            max_matches: Maximum number of matches

        Returns:
            Filtered list of VisionObject instances
        """
        if len(objects) == 0:
            return []

        # Convert to boxes
        boxes = []
        for obj in objects:
            bb = obj.bounding_box
            boxes.append([bb.x, bb.y, bb.x + bb.width, bb.y + bb.height, obj.confidence])

        boxes = np.array(boxes)

        # Apply NMS
        keep_indices = []
        while len(boxes) > 0 and len(keep_indices) < max_matches:
            # Pick box with highest score
            keep_indices.append(len(objects) - len(boxes))
            if len(boxes) == 1:
                break

            # Calculate IoU
            current_box = boxes[0]
            remaining_boxes = boxes[1:]

            ious = self._calculate_iou_vectorized(current_box[:4], remaining_boxes[:, :4])

            # Keep only boxes with IoU below threshold
            keep_mask = ious < overlap_threshold
            boxes = remaining_boxes[keep_mask]

        # Re-index objects
        result = []
        for idx, obj_idx in enumerate(keep_indices):
            obj = objects[obj_idx]
            # Update object_id to reflect new index
            obj.object_id = f"match_{idx}"
            result.append(obj)

        return result

    def _calculate_iou_vectorized(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Calculate IoU between one box and multiple boxes (vectorized).

        Args:
            box: Single box [x1, y1, x2, y2]
            boxes: Multiple boxes [[x1, y1, x2, y2], ...]

        Returns:
            Array of IoU values
        """
        # Calculate intersection
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        # Calculate union
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = box_area + boxes_area - intersection

        # Calculate IoU (handle division by zero)
        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=float),
            where=union != 0,
        )

        return iou

    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by given angle.

        Args:
            image: Input image
            angle: Rotation angle in degrees (positive = counter-clockwise)

        Returns:
            Rotated image (cropped to avoid black borders)
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new bounding box size
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust rotation matrix for translation
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        # Perform rotation
        rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR)

        return rotated

    def _create_vision_object(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        score: float,
        rotation_angle: float,
        template_id: str,
        method: str,
        index: int,
    ) -> VisionObject:
        """
        Create VisionObject from match parameters.

        Args:
            x: Top-left x coordinate
            y: Top-left y coordinate
            width: Template width
            height: Template height
            score: Match score
            rotation_angle: Rotation angle in degrees
            template_id: Template identifier
            method: Matching method
            index: Match index

        Returns:
            VisionObject instance
        """
        return VisionObject(
            object_id=f"match_{index}",
            object_type=VisionObjectType.TEMPLATE_MATCH.value,
            bounding_box=ROI(x=int(x), y=int(y), width=int(width), height=int(height)),
            center=Point(x=float(x + width // 2), y=float(y + height // 2)),
            confidence=min(float(score), 1.0),
            rotation=float(rotation_angle) if rotation_angle != 0.0 else None,
            properties={
                "template_id": template_id,
                "method": method,
                "rotation_angle": float(rotation_angle),
                "scale": 1.0,
                "raw_score": float(score),
            },
        )
