"""
Unit tests for ArUco Reference schemas and validation
"""

import pytest
from pydantic import ValidationError

from schemas import (
    ArucoReferenceMode,
    ArucoReferenceParams,
    ArucoReferenceRequest,
    ArucoReferenceResponse,
    PlaneConfig,
    ReferenceObject,
    SingleConfig,
)


class TestSingleConfig:
    """Tests for SingleConfig schema"""

    def test_single_config_basic(self):
        """Test basic SingleConfig creation"""
        config = SingleConfig(
            marker_id=0,
            marker_size_mm=50.0,
            origin="marker_center",
            rotation_reference="marker_rotation",
        )

        assert config.marker_id == 0
        assert config.marker_size_mm == 50.0
        assert config.origin == "marker_center"
        assert config.rotation_reference == "marker_rotation"

    def test_single_config_defaults(self):
        """Test SingleConfig default values"""
        config = SingleConfig(marker_id=0, marker_size_mm=50.0)

        assert config.origin == "marker_center"  # Default
        assert config.rotation_reference == "marker_rotation"  # Default

    def test_single_config_all_origins(self):
        """Test all valid origin values"""
        origins = ["marker_center", "marker_top_left", "marker_bottom_left"]

        for origin in origins:
            config = SingleConfig(marker_id=0, marker_size_mm=50.0, origin=origin)
            assert config.origin == origin

    def test_single_config_invalid_origin(self):
        """Test validation error for invalid origin"""
        with pytest.raises(ValidationError) as exc_info:
            SingleConfig(marker_id=0, marker_size_mm=50.0, origin="invalid_origin")

        error = exc_info.value
        assert "origin" in str(error).lower()

    def test_single_config_all_rotation_references(self):
        """Test all valid rotation_reference values"""
        rotation_refs = ["marker_rotation", "image_axes"]

        for rotation_ref in rotation_refs:
            config = SingleConfig(
                marker_id=0, marker_size_mm=50.0, rotation_reference=rotation_ref
            )
            assert config.rotation_reference == rotation_ref

    def test_single_config_invalid_rotation_reference(self):
        """Test validation error for invalid rotation_reference"""
        with pytest.raises(ValidationError) as exc_info:
            SingleConfig(
                marker_id=0, marker_size_mm=50.0, rotation_reference="invalid_ref"
            )

        error = exc_info.value
        assert "rotation_reference" in str(error).lower()

    def test_single_config_negative_marker_size(self):
        """Test validation error for negative marker size"""
        with pytest.raises(ValidationError) as exc_info:
            SingleConfig(marker_id=0, marker_size_mm=-10.0)

        error = exc_info.value
        assert "marker_size_mm" in str(error).lower()

    def test_single_config_zero_marker_size(self):
        """Test validation error for zero marker size"""
        with pytest.raises(ValidationError) as exc_info:
            SingleConfig(marker_id=0, marker_size_mm=0.0)

        error = exc_info.value
        assert "marker_size_mm" in str(error).lower()


class TestPlaneConfig:
    """Tests for PlaneConfig schema"""

    def test_plane_config_basic(self):
        """Test basic PlaneConfig creation"""
        config = PlaneConfig(
            marker_ids={"top_left": 0, "top_right": 1, "bottom_right": 2, "bottom_left": 3},
            width_mm=200.0,
            height_mm=150.0,
            origin="top_left",
            x_direction="right",
            y_direction="down",
        )

        assert config.marker_ids["top_left"] == 0
        assert config.marker_ids["top_right"] == 1
        assert config.width_mm == 200.0
        assert config.height_mm == 150.0

    def test_plane_config_defaults(self):
        """Test PlaneConfig default values"""
        config = PlaneConfig(
            marker_ids={"top_left": 0, "top_right": 1, "bottom_right": 2, "bottom_left": 3},
            width_mm=200.0,
            height_mm=150.0,
        )

        assert config.origin == "top_left"  # Default
        assert config.x_direction == "right"  # Default
        assert config.y_direction == "down"  # Default

    def test_plane_config_all_origins(self):
        """Test all valid origin values"""
        origins = ["top_left", "top_right", "bottom_left", "bottom_right"]

        for origin in origins:
            config = PlaneConfig(
                marker_ids={"top_left": 0, "top_right": 1, "bottom_right": 2, "bottom_left": 3},
                width_mm=200.0,
                height_mm=150.0,
                origin=origin,
            )
            assert config.origin == origin

    def test_plane_config_all_directions(self):
        """Test all valid direction combinations"""
        x_directions = ["right", "left"]
        y_directions = ["down", "up"]

        for x_dir in x_directions:
            for y_dir in y_directions:
                config = PlaneConfig(
                    marker_ids={
                        "top_left": 0,
                        "top_right": 1,
                        "bottom_right": 2,
                        "bottom_left": 3,
                    },
                    width_mm=200.0,
                    height_mm=150.0,
                    x_direction=x_dir,
                    y_direction=y_dir,
                )
                assert config.x_direction == x_dir
                assert config.y_direction == y_dir

    def test_plane_config_missing_marker_ids(self):
        """Test validation error when marker_ids are missing"""
        with pytest.raises(ValidationError) as exc_info:
            PlaneConfig(
                marker_ids={"top_left": 0, "top_right": 1},  # Missing bottom corners
                width_mm=200.0,
                height_mm=150.0,
            )

        error = exc_info.value
        assert "marker_ids" in str(error).lower() or "bottom" in str(error).lower()

    def test_plane_config_negative_dimensions(self):
        """Test validation error for negative dimensions"""
        with pytest.raises(ValidationError) as exc_info:
            PlaneConfig(
                marker_ids={
                    "top_left": 0,
                    "top_right": 1,
                    "bottom_right": 2,
                    "bottom_left": 3,
                },
                width_mm=-200.0,
                height_mm=150.0,
            )

        error = exc_info.value
        assert "width_mm" in str(error).lower()


class TestArucoReferenceParams:
    """Tests for ArucoReferenceParams schema"""

    def test_params_single_mode_valid(self):
        """Test valid SINGLE mode params"""
        params = ArucoReferenceParams(
            mode=ArucoReferenceMode.SINGLE,
            single_config=SingleConfig(marker_id=0, marker_size_mm=50.0),
        )

        assert params.mode == ArucoReferenceMode.SINGLE
        assert params.single_config is not None
        assert params.plane_config is None

    def test_params_plane_mode_valid(self):
        """Test valid PLANE mode params"""
        params = ArucoReferenceParams(
            mode=ArucoReferenceMode.PLANE,
            plane_config=PlaneConfig(
                marker_ids={
                    "top_left": 0,
                    "top_right": 1,
                    "bottom_right": 2,
                    "bottom_left": 3,
                },
                width_mm=200.0,
                height_mm=150.0,
            ),
        )

        assert params.mode == ArucoReferenceMode.PLANE
        assert params.plane_config is not None
        assert params.single_config is None

    def test_params_single_mode_missing_config(self):
        """Test validation error when single_config is missing in SINGLE mode"""
        with pytest.raises(ValidationError) as exc_info:
            ArucoReferenceParams(
                mode=ArucoReferenceMode.SINGLE,
                # Missing single_config
            )

        error = exc_info.value
        assert "single_config" in str(error).lower()

    def test_params_plane_mode_missing_config(self):
        """Test validation error when plane_config is missing in PLANE mode"""
        with pytest.raises(ValidationError) as exc_info:
            ArucoReferenceParams(
                mode=ArucoReferenceMode.PLANE,
                # Missing plane_config
            )

        error = exc_info.value
        assert "plane_config" in str(error).lower()

    def test_params_single_mode_wrong_config(self):
        """Test validation error when plane_config provided in SINGLE mode"""
        with pytest.raises(ValidationError) as exc_info:
            ArucoReferenceParams(
                mode=ArucoReferenceMode.SINGLE,
                single_config=SingleConfig(marker_id=0, marker_size_mm=50.0),
                plane_config=PlaneConfig(
                    marker_ids={
                        "top_left": 0,
                        "top_right": 1,
                        "bottom_right": 2,
                        "bottom_left": 3,
                    },
                    width_mm=200.0,
                    height_mm=150.0,
                ),
            )

        error = exc_info.value
        assert "plane_config" in str(error).lower()

    def test_params_plane_mode_wrong_config(self):
        """Test validation error when single_config provided in PLANE mode"""
        with pytest.raises(ValidationError) as exc_info:
            ArucoReferenceParams(
                mode=ArucoReferenceMode.PLANE,
                single_config=SingleConfig(marker_id=0, marker_size_mm=50.0),
                plane_config=PlaneConfig(
                    marker_ids={
                        "top_left": 0,
                        "top_right": 1,
                        "bottom_right": 2,
                        "bottom_left": 3,
                    },
                    width_mm=200.0,
                    height_mm=150.0,
                ),
            )

        error = exc_info.value
        assert "single_config" in str(error).lower()

    def test_params_dictionary_default(self):
        """Test default dictionary value"""
        params = ArucoReferenceParams(
            mode=ArucoReferenceMode.SINGLE,
            single_config=SingleConfig(marker_id=0, marker_size_mm=50.0),
        )

        assert params.dictionary is not None  # Should have default


class TestArucoReferenceRequest:
    """Tests for ArucoReferenceRequest schema"""

    def test_request_basic(self):
        """Test basic request creation"""
        request = ArucoReferenceRequest(
            image_id="test-image-123",
            params=ArucoReferenceParams(
                mode=ArucoReferenceMode.SINGLE,
                single_config=SingleConfig(marker_id=0, marker_size_mm=50.0),
            ),
        )

        assert request.image_id == "test-image-123"
        assert request.params.mode == ArucoReferenceMode.SINGLE
        assert request.roi is None

    def test_request_with_roi(self):
        """Test request with ROI"""
        from schemas import ROI

        request = ArucoReferenceRequest(
            image_id="test-image-123",
            roi=ROI(x=0, y=0, width=400, height=400),
            params=ArucoReferenceParams(
                mode=ArucoReferenceMode.SINGLE,
                single_config=SingleConfig(marker_id=0, marker_size_mm=50.0),
            ),
        )

        assert request.roi is not None
        assert request.roi.width == 400


class TestArucoReferenceResponse:
    """Tests for ArucoReferenceResponse schema"""

    def test_response_basic(self):
        """Test basic response creation"""
        from schemas import VisionObject, VisionObjectType, ROI, Point

        ref_obj = ReferenceObject(
            type="single_marker",
            units="mm",
            homography_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            metadata={"marker_id": 0, "scale_mm_per_pixel": 0.5},
        )

        marker = VisionObject(
            object_id="aruco_0",
            object_type=VisionObjectType.ARUCO_MARKER.value,
            bounding_box=ROI(x=100, y=100, width=50, height=50),
            center=Point(x=125, y=125),
            confidence=1.0,
            properties={"marker_id": 0},
        )

        response = ArucoReferenceResponse(
            reference_object=ref_obj,
            markers=[marker],
            thumbnail_base64="base64data",
            processing_time_ms=42,
        )

        assert response.reference_object.type == "single_marker"
        assert len(response.markers) == 1
        assert response.processing_time_ms == 42

    def test_response_reference_object_required(self):
        """Test that reference_object is required"""
        from schemas import VisionObject, VisionObjectType, ROI, Point

        marker = VisionObject(
            object_id="aruco_0",
            object_type=VisionObjectType.ARUCO_MARKER.value,
            bounding_box=ROI(x=100, y=100, width=50, height=50),
            center=Point(x=125, y=125),
            confidence=1.0,
            properties={"marker_id": 0},
        )

        with pytest.raises(ValidationError) as exc_info:
            ArucoReferenceResponse(
                # Missing reference_object
                markers=[marker],
                thumbnail_base64="base64data",
                processing_time_ms=42,
            )

        error = exc_info.value
        assert "reference_object" in str(error).lower()


class TestReferenceObject:
    """Tests for ReferenceObject schema"""

    def test_reference_object_single_marker(self):
        """Test ReferenceObject for single marker"""
        ref_obj = ReferenceObject(
            type="single_marker",
            units="mm",
            homography_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            metadata={
                "marker_id": 0,
                "marker_size_mm": 50.0,
                "scale_mm_per_pixel": 0.5,
                "origin": "marker_center",
            },
        )

        assert ref_obj.type == "single_marker"
        assert ref_obj.units == "mm"
        assert ref_obj.metadata["marker_id"] == 0

    def test_reference_object_plane(self):
        """Test ReferenceObject for plane"""
        ref_obj = ReferenceObject(
            type="plane",
            units="mm",
            homography_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            metadata={
                "marker_ids": {
                    "top_left": 0,
                    "top_right": 1,
                    "bottom_right": 2,
                    "bottom_left": 3,
                },
                "width_mm": 200.0,
                "height_mm": 150.0,
            },
        )

        assert ref_obj.type == "plane"
        assert ref_obj.metadata["width_mm"] == 200.0

    def test_reference_object_homography_structure(self):
        """Test that homography matrix has correct structure"""
        ref_obj = ReferenceObject(
            type="single_marker",
            units="mm",
            homography_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            metadata={},
        )

        H = ref_obj.homography_matrix
        assert len(H) == 3
        assert all(len(row) == 3 for row in H)
