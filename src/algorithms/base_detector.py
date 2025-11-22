"""
Base detector class for vision algorithms.

Provides common interface and utilities for all vision detectors.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from domain_types import ROI, Point, VisionObjectType
from models import VisionObject


class BaseDetector(ABC):
    """
    Abstract base class for vision detectors.

    Provides common interface and utilities for all vision algorithms.
    All detectors should inherit from this class and implement the detect method.
    """

    def __init__(self):
        """Initialize base detector with logger."""
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Perform detection on image.

        Args:
            image: Input image (BGR or grayscale)
            params: Detection parameters
            **kwargs: Additional detector-specific arguments

        Returns:
            Dictionary with detection results containing:
                - success: bool
                - objects: List[VisionObject]
                - image: np.ndarray (processed/annotated image)
                - metadata: Dict (optional additional info)
        """
        pass

    def _create_vision_object(
        self,
        object_id: str,
        object_type: VisionObjectType,
        bounding_box: ROI,
        confidence: float,
        center: Optional[Point] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VisionObject:
        """
        Create VisionObject from detection results.

        Args:
            object_id: Unique object identifier
            object_type: Type of detected object
            bounding_box: Bounding box ROI
            confidence: Detection confidence (0-1)
            center: Object center point
            metadata: Additional metadata

        Returns:
            VisionObject instance
        """
        if center is None:
            # Calculate center from bounding box
            center = Point(
                x=bounding_box.x + bounding_box.width / 2,
                y=bounding_box.y + bounding_box.height / 2,
            )

        return VisionObject(
            object_id=object_id,
            object_type=object_type,
            bounding_box=bounding_box,
            center=center,
            confidence=confidence,
            metadata=metadata or {},
        )

    def _create_result(
        self,
        success: bool,
        objects: List[VisionObject],
        image: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create standardized detection result dictionary.

        Args:
            success: Whether detection was successful
            objects: List of detected objects
            image: Processed/annotated image
            metadata: Additional metadata

        Returns:
            Standardized result dictionary
        """
        return {
            "success": success,
            "objects": objects,
            "image": image,
            "metadata": metadata or {},
        }

    def _ensure_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale if needed.

        Args:
            image: Input image

        Returns:
            Grayscale image
        """
        from managers.image.converters import ensure_grayscale

        return ensure_grayscale(image)

    def _ensure_bgr(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to BGR if needed.

        Args:
            image: Input image

        Returns:
            BGR image
        """
        from managers.image.converters import ensure_bgr

        return ensure_bgr(image)
