"""
Unit tests for coordinate transformation utilities
"""

import pytest

from domain_types import ROI, Point
from image.transform import apply_reference_transform
from models import ReferenceObject, VisionObject


class TestApplyReferenceTransform:
    """Tests for apply_reference_transform function"""

    def test_apply_transform_sets_plane_position(self):
        """Test that transformation sets plane_position field"""
        # Create test object
        obj = VisionObject(
            object_id="test_1",
            object_type="test",
            center=Point(x=100.0, y=100.0),
            bounding_box=ROI(x=90, y=90, width=20, height=20),
            confidence=0.95,
        )

        # Create identity homography (should not transform coordinates)
        ref_obj = ReferenceObject(
            type="single_marker",
            units="mm",
            homography_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            metadata={},
            thumbnail="data:image/jpeg;base64,test",
        )

        # Apply transformation
        result = apply_reference_transform(obj, ref_obj)

        # Check that plane_position is set
        assert result.plane_position is not None
        assert isinstance(result.plane_position, Point)
        assert result.plane_position.x == pytest.approx(100.0, abs=0.1)
        assert result.plane_position.y == pytest.approx(100.0, abs=0.1)

    def test_apply_transform_sets_plane_rotation(self):
        """Test that transformation sets plane_rotation field when rotation present"""
        # Create test object with rotation
        obj = VisionObject(
            object_id="test_2",
            object_type="test",
            center=Point(x=100.0, y=100.0),
            bounding_box=ROI(x=90, y=90, width=20, height=20),
            confidence=0.95,
            rotation=45.0,
        )

        # Create identity homography
        ref_obj = ReferenceObject(
            type="single_marker",
            units="mm",
            homography_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            metadata={},
            thumbnail="data:image/jpeg;base64,test",
        )

        # Apply transformation
        result = apply_reference_transform(obj, ref_obj)

        # Check that plane_rotation is set
        assert result.plane_rotation is not None
        assert result.plane_rotation == pytest.approx(45.0, abs=0.1)

    def test_apply_transform_without_rotation(self):
        """Test that plane_rotation is None when object has no rotation"""
        # Create test object without rotation
        obj = VisionObject(
            object_id="test_3",
            object_type="test",
            center=Point(x=100.0, y=100.0),
            bounding_box=ROI(x=90, y=90, width=20, height=20),
            confidence=0.95,
        )

        # Create identity homography
        ref_obj = ReferenceObject(
            type="single_marker",
            units="mm",
            homography_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            metadata={},
            thumbnail="data:image/jpeg;base64,test",
        )

        # Apply transformation
        result = apply_reference_transform(obj, ref_obj)

        # Check that plane_rotation is None
        assert result.plane_rotation is None

    def test_transform_with_scale(self):
        """Test transformation with scaling homography"""
        # Create test object
        obj = VisionObject(
            object_id="test_4",
            object_type="test",
            center=Point(x=100.0, y=100.0),
            bounding_box=ROI(x=90, y=90, width=20, height=20),
            confidence=0.95,
        )

        # Create scaling homography (2x scale)
        ref_obj = ReferenceObject(
            type="single_marker",
            units="mm",
            homography_matrix=[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]],
            metadata={},
            thumbnail="data:image/jpeg;base64,test",
        )

        # Apply transformation
        result = apply_reference_transform(obj, ref_obj)

        # Check that plane_position is scaled
        assert result.plane_position is not None
        assert result.plane_position.x == pytest.approx(200.0, abs=0.1)
        assert result.plane_position.y == pytest.approx(200.0, abs=0.1)

    def test_no_properties_pollution(self):
        """Test that transformation doesn't add to properties dict"""
        # Create test object
        obj = VisionObject(
            object_id="test_5",
            object_type="test",
            center=Point(x=100.0, y=100.0),
            bounding_box=ROI(x=90, y=90, width=20, height=20),
            confidence=0.95,
            properties={"custom_prop": "value"},
        )

        # Create identity homography
        ref_obj = ReferenceObject(
            type="single_marker",
            units="mm",
            homography_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            metadata={},
            thumbnail="data:image/jpeg;base64,test",
        )

        # Apply transformation
        result = apply_reference_transform(obj, ref_obj)

        # Check that properties doesn't contain plane_* keys
        assert "plane_position_mm" not in result.properties
        assert "plane_rotation_deg" not in result.properties
        assert "plane_applied" not in result.properties
        assert "plane_type" not in result.properties
        assert "plane_units" not in result.properties

        # Check that custom property is preserved
        assert result.properties["custom_prop"] == "value"
