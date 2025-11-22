"""
Coordinate transformation utilities for vision objects.

Handles transformation of VisionObject coordinates and properties
from image space to reference frame space using homography matrices.
"""

from typing import List, Optional

import numpy as np

from domain_types import Point
from models import ReferenceObject, VisionObject

from .geometry import transform_point_homography, transform_rotation_homography


def apply_reference_transform(
    obj: "VisionObject",
    reference_object: "ReferenceObject",
) -> "VisionObject":
    """
    Apply reference frame transformation to VisionObject.

    Transforms object's position and rotation from image coordinates
    to reference frame coordinates and sets plane_position and plane_rotation fields.

    Args:
        obj: VisionObject in image coordinates
        reference_object: Reference frame with homography matrix

    Returns:
        Modified VisionObject with plane_position and plane_rotation set

    Note:
        Modifies the object in place and returns the same object.
    """
    # Extract homography matrix
    H = np.array(reference_object.homography_matrix, dtype=np.float64)

    # Transform center point
    center_tuple = (obj.center.x, obj.center.y)
    transformed_center = transform_point_homography(center_tuple, H)

    # Set plane position
    obj.plane_position = Point(x=transformed_center[0], y=transformed_center[1])

    # Transform rotation if present
    if obj.rotation is not None:
        obj.plane_rotation = transform_rotation_homography(obj.rotation, H, center_tuple)

    return obj


def apply_reference_transform_batch(
    objects: List["VisionObject"],
    reference_object: Optional["ReferenceObject"],
) -> List["VisionObject"]:
    """
    Apply reference frame transformation to multiple VisionObjects.

    Convenience function for batch transformation. If reference_object is None,
    returns objects unchanged.

    Args:
        objects: List of VisionObjects in image coordinates
        reference_object: Reference frame with homography matrix (or None to skip)

    Returns:
        List of VisionObjects with plane_* properties added (if reference provided)
    """
    if reference_object is None:
        return objects

    # Transform each object
    for obj in objects:
        apply_reference_transform(obj, reference_object)

    return objects
