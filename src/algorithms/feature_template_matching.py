"""
Feature-based template matching using ORB keypoint detection.

Provides rotation and scale invariant template matching using:
- ORB feature detection and description
- BFMatcher with Lowe's ratio test
- RANSAC-based homography estimation
- Multi-instance detection via keypoint clustering
"""

import logging
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from algorithms.base_detector import BaseDetector
from domain_types import ROI, Point, VisionObjectType
from image.overlay import render_feature_matches
from models import VisionObject

logger = logging.getLogger(__name__)

# Minimum keypoints for reliable detection
MIN_TEMPLATE_KEYPOINTS = 20
MIN_MATCH_COUNT = 4  # Minimum for homography


class FeatureTemplateDetector(BaseDetector):
    """Feature-based template matching using ORB."""

    def __init__(self, n_features: int = 500):
        """
        Initialize feature template detector.

        Args:
            n_features: Maximum number of ORB features to detect
        """
        super().__init__()
        self._orb = cv2.ORB_create(nfeatures=n_features)
        self._bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect(
        self,
        image: np.ndarray,
        template: np.ndarray,
        template_id: str,
        params: Dict[str, Any],
        mask: np.ndarray = None,
    ) -> Dict:
        """
        Perform feature-based template matching.

        Args:
            image: Input image (BGR format)
            template: Template image to search for
            template_id: Template identifier for metadata
            params: Detection parameters dict with keys:
                - threshold: Match confidence threshold (0.0-1.0)
                - min_matches: Minimum feature matches required
                - ratio_threshold: Lowe's ratio test threshold
                - find_multiple: Enable multi-instance detection
                - max_matches: Maximum number of matches to return
            mask: Optional mask for template (alpha channel)

        Returns:
            Dictionary with detection results
        """
        # Extract params
        threshold = params.get("threshold", 0.6)
        min_matches = params.get("min_matches", 10)
        ratio_threshold = params.get("ratio_threshold", 0.75)
        find_multiple = params.get("find_multiple", False)
        max_matches = params.get("max_matches", 10)

        # Convert to grayscale
        if len(image.shape) == 3:
            scene_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            scene_gray = image

        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template

        # Prepare mask for ORB
        orb_mask = None
        if mask is not None:
            if mask.dtype != np.uint8:
                orb_mask = mask.astype(np.uint8)
            else:
                orb_mask = mask
            # Threshold mask to binary
            _, orb_mask = cv2.threshold(orb_mask, 127, 255, cv2.THRESH_BINARY)

        # Detect keypoints and descriptors
        kp_template, desc_template = self._orb.detectAndCompute(template_gray, orb_mask)
        kp_scene, desc_scene = self._orb.detectAndCompute(scene_gray, None)

        # Check for sufficient keypoints
        if kp_template is None or len(kp_template) < MIN_TEMPLATE_KEYPOINTS:
            kp_count = len(kp_template) if kp_template else 0
            logger.warning(
                f"Template {template_id} has only {kp_count} keypoints. "
                "Detection may be unreliable. Consider using a more textured template."
            )

        # Early exit if insufficient features
        if desc_template is None or desc_scene is None:
            logger.debug("No descriptors found in template or scene")
            return self._empty_result(image)

        if len(kp_template) < MIN_MATCH_COUNT or len(kp_scene) < MIN_MATCH_COUNT:
            logger.debug(
                f"Insufficient keypoints: template={len(kp_template)}, scene={len(kp_scene)}"
            )
            return self._empty_result(image)

        # Match features
        good_matches = self._match_features(desc_template, desc_scene, ratio_threshold)

        if len(good_matches) < min_matches:
            logger.debug(f"Insufficient good matches: {len(good_matches)} < {min_matches}")
            return self._empty_result(image)

        # Find instances
        if find_multiple:
            detected_objects = self._find_multiple_instances(
                template_gray,
                kp_template,
                kp_scene,
                good_matches,
                template_id,
                threshold,
                min_matches,
                max_matches,
            )
        else:
            detected_objects = self._find_single_instance(
                template_gray,
                kp_template,
                kp_scene,
                good_matches,
                template_id,
                threshold,
            )

        # Create visualization
        if detected_objects:
            result_image = render_feature_matches(image, detected_objects)
        else:
            result_image = image.copy()

        return {
            "success": True,
            "objects": detected_objects,
            "image": result_image,
        }

    def _match_features(
        self,
        desc_template: np.ndarray,
        desc_scene: np.ndarray,
        ratio_threshold: float,
    ) -> List[cv2.DMatch]:
        """
        Match features using BFMatcher with Lowe's ratio test.

        Args:
            desc_template: Template descriptors
            desc_scene: Scene descriptors
            ratio_threshold: Ratio test threshold

        Returns:
            List of good matches
        """
        # kNN matching with k=2
        matches = self._bf_matcher.knnMatch(desc_template, desc_scene, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)

        return good_matches

    def _find_single_instance(
        self,
        template: np.ndarray,
        kp_template: List,
        kp_scene: List,
        matches: List[cv2.DMatch],
        template_id: str,
        threshold: float,
    ) -> List[VisionObject]:
        """
        Find single best template instance using homography.

        Args:
            template: Template image (grayscale)
            kp_template: Template keypoints
            kp_scene: Scene keypoints
            matches: Good feature matches
            template_id: Template identifier
            threshold: Confidence threshold

        Returns:
            List with single VisionObject or empty
        """
        # Get matched point coordinates
        src_pts = np.float32([kp_template[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find homography using RANSAC
        H, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is None:
            return []

        # Calculate confidence from inlier ratio
        inlier_count = int(np.sum(inliers)) if inliers is not None else 0
        total_matches = len(matches)
        confidence = inlier_count / total_matches if total_matches > 0 else 0

        if confidence < threshold:
            logger.debug(f"Match confidence {confidence:.2f} below threshold {threshold}")
            return []

        # Create VisionObject from homography
        obj = self._create_object_from_homography(
            template, H, template_id, confidence, inlier_count, total_matches, 0
        )

        return [obj] if obj else []

    def _find_multiple_instances(
        self,
        template: np.ndarray,
        kp_template: List,
        kp_scene: List,
        matches: List[cv2.DMatch],
        template_id: str,
        threshold: float,
        min_matches: int,
        max_instances: int,
    ) -> List[VisionObject]:
        """
        Find multiple template instances by iterative RANSAC.

        Args:
            template: Template image (grayscale)
            kp_template: Template keypoints
            kp_scene: Scene keypoints
            matches: Good feature matches
            template_id: Template identifier
            threshold: Confidence threshold
            min_matches: Minimum matches per instance
            max_instances: Maximum instances to find

        Returns:
            List of VisionObject instances
        """
        detected_objects = []
        remaining_matches = list(matches)
        used_scene_indices = set()

        for instance_idx in range(max_instances):
            if len(remaining_matches) < min_matches:
                break

            # Get point coordinates for remaining matches
            src_pts = np.float32([kp_template[m.queryIdx].pt for m in remaining_matches]).reshape(
                -1, 1, 2
            )
            dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in remaining_matches]).reshape(
                -1, 1, 2
            )

            # Find homography
            H, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is None:
                break

            # Get inlier matches
            inlier_mask = inliers.ravel().astype(bool) if inliers is not None else []
            inlier_matches = [
                m for m, is_inlier in zip(remaining_matches, inlier_mask) if is_inlier
            ]

            if len(inlier_matches) < min_matches:
                break

            # Calculate confidence
            confidence = len(inlier_matches) / len(remaining_matches)

            if confidence < threshold:
                # Try to continue with remaining matches
                remaining_matches = [
                    m for m, is_inlier in zip(remaining_matches, inlier_mask) if not is_inlier
                ]
                continue

            # Create object
            obj = self._create_object_from_homography(
                template,
                H,
                template_id,
                confidence,
                len(inlier_matches),
                len(remaining_matches),
                instance_idx,
            )

            if obj:
                detected_objects.append(obj)

            # Remove inlier matches from remaining
            inlier_scene_indices = {m.trainIdx for m in inlier_matches}
            used_scene_indices.update(inlier_scene_indices)
            remaining_matches = [
                m for m in remaining_matches if m.trainIdx not in used_scene_indices
            ]

        return detected_objects

    def _create_object_from_homography(
        self,
        template: np.ndarray,
        H: np.ndarray,
        template_id: str,
        confidence: float,
        inlier_count: int,
        match_count: int,
        index: int,
    ) -> Optional[VisionObject]:
        """
        Create VisionObject from homography matrix.

        Args:
            template: Template image
            H: Homography matrix (3x3)
            template_id: Template identifier
            confidence: Match confidence
            inlier_count: Number of inlier matches
            match_count: Total number of matches
            index: Instance index

        Returns:
            VisionObject or None if invalid
        """
        h, w = template.shape[:2]

        # Transform template corners
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

        try:
            transformed = cv2.perspectiveTransform(corners, H)
        except cv2.error:
            logger.warning("Failed to transform template corners")
            return None

        pts = transformed.reshape(-1, 2)

        # Validate transformed corners (check for degenerate shapes)
        if not self._is_valid_quadrilateral(pts):
            logger.debug("Invalid quadrilateral detected, skipping")
            return None

        # Get axis-aligned bounding box
        x_min, y_min = pts.min(axis=0).astype(int)
        x_max, y_max = pts.max(axis=0).astype(int)

        # Ensure positive dimensions
        if x_max <= x_min or y_max <= y_min:
            return None

        # Extract rotation from homography
        rotation = np.degrees(np.arctan2(H[1, 0], H[0, 0]))

        # Extract scale
        scale_x = np.sqrt(H[0, 0] ** 2 + H[1, 0] ** 2)
        scale_y = np.sqrt(H[0, 1] ** 2 + H[1, 1] ** 2)
        scale = (scale_x + scale_y) / 2

        # Calculate center
        center_x = float(pts[:, 0].mean())
        center_y = float(pts[:, 1].mean())

        return VisionObject(
            object_id=f"match_{index}",
            object_type=VisionObjectType.FEATURE_TEMPLATE_MATCH.value,
            bounding_box=ROI(
                x=max(0, int(x_min)),
                y=max(0, int(y_min)),
                width=max(1, int(x_max - x_min)),
                height=max(1, int(y_max - y_min)),
            ),
            center=Point(x=center_x, y=center_y),
            confidence=min(float(confidence), 1.0),
            rotation=float(rotation),
            properties={
                "template_id": template_id,
                "method": "feature_orb",
                "rotation_angle": float(rotation),
                "scale": float(scale),
                "inlier_count": inlier_count,
                "match_count": match_count,
                "corners": pts.tolist(),
            },
        )

    def _is_valid_quadrilateral(self, pts: np.ndarray) -> bool:
        """
        Check if transformed corners form a valid quadrilateral.

        Args:
            pts: 4x2 array of corner points

        Returns:
            True if valid quadrilateral
        """
        if pts.shape != (4, 2):
            return False

        # Check area is positive and not too small
        # Using shoelace formula
        area = 0.5 * abs(
            (pts[0, 0] - pts[2, 0]) * (pts[1, 1] - pts[3, 1])
            - (pts[1, 0] - pts[3, 0]) * (pts[0, 1] - pts[2, 1])
        )

        if area < 100:  # Minimum 100 pixels area
            return False

        # Check convexity (all cross products should have same sign)
        def cross_product_sign(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        signs = []
        for i in range(4):
            o = pts[i]
            a = pts[(i + 1) % 4]
            b = pts[(i + 2) % 4]
            signs.append(cross_product_sign(o, a, b) > 0)

        # All signs should be the same for convex shape
        return all(signs) or not any(signs)

    def _empty_result(self, image: np.ndarray) -> Dict:
        """Create empty result dictionary."""
        return {
            "success": True,
            "objects": [],
            "image": image.copy(),
        }
