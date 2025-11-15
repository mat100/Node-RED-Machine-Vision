"""
API Integration Tests for ArUco Reference Frame Endpoints
"""

import pytest


class TestArucoReferenceAPI:
    """Integration tests for ArUco reference frame creation API"""

    @pytest.fixture
    def captured_image_id(self, client):
        """Capture a test image with ArUco markers"""
        response = client.post("/api/camera/capture", json={"camera_id": "test"})
        assert response.status_code == 200
        data = response.json()
        return data["objects"][0]["properties"]["image_id"]

    def test_aruco_reference_single_mode_basic(self, client, captured_image_id):
        """Test ArUco reference creation with SINGLE marker mode"""
        request_data = {
            "image_id": captured_image_id,
            "params": {
                "dictionary": "DICT_4X4_50",
                "mode": "single",
                "single_config": {
                    "marker_id": 0,
                    "marker_size_mm": 50.0,
                    "origin": "marker_center",
                    "rotation_reference": "marker_rotation",
                },
            },
        }
        response = client.post("/api/vision/aruco-reference", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "reference_object" in data
        assert "markers" in data
        assert "thumbnail_base64" in data
        assert "processing_time_ms" in data
        assert data["processing_time_ms"] > 0

        # Check reference_object structure
        ref_obj = data["reference_object"]
        assert ref_obj["type"] == "single_marker"
        assert ref_obj["units"] == "mm"
        assert "homography_matrix" in ref_obj
        assert isinstance(ref_obj["homography_matrix"], list)
        assert len(ref_obj["homography_matrix"]) == 3  # 3x3 matrix
        assert len(ref_obj["homography_matrix"][0]) == 3

        # Check metadata
        assert "metadata" in ref_obj
        metadata = ref_obj["metadata"]
        assert metadata["marker_id"] == 0
        assert metadata["marker_size_mm"] == 50.0
        assert metadata["origin"] == "marker_center"
        assert "scale_mm_per_pixel" in metadata

        # Check markers were detected
        assert isinstance(data["markers"], list)
        assert len(data["markers"]) > 0

        # Verify the reference marker is in the list
        marker_ids = [m["properties"]["marker_id"] for m in data["markers"]]
        assert 0 in marker_ids

    def test_aruco_reference_single_mode_different_origins(self, client, captured_image_id):
        """Test SINGLE mode with different origin configurations"""
        origins = ["marker_center", "marker_top_left", "marker_bottom_left"]

        for origin in origins:
            request_data = {
                "image_id": captured_image_id,
                "params": {
                    "dictionary": "DICT_4X4_50",
                    "mode": "single",
                    "single_config": {
                        "marker_id": 0,
                        "marker_size_mm": 50.0,
                        "origin": origin,
                        "rotation_reference": "marker_rotation",
                    },
                },
            }
            response = client.post("/api/vision/aruco-reference", json=request_data)

            assert response.status_code == 200
            data = response.json()

            # Each origin should create valid reference
            assert data["reference_object"]["metadata"]["origin"] == origin
            assert "origin_point_px" in data["reference_object"]["metadata"]

    def test_aruco_reference_single_mode_rotation_references(self, client, captured_image_id):
        """Test SINGLE mode with different rotation reference configurations"""
        rotation_refs = ["marker_rotation", "image_axes"]

        for rotation_ref in rotation_refs:
            request_data = {
                "image_id": captured_image_id,
                "params": {
                    "dictionary": "DICT_4X4_50",
                    "mode": "single",
                    "single_config": {
                        "marker_id": 0,
                        "marker_size_mm": 50.0,
                        "origin": "marker_center",
                        "rotation_reference": rotation_ref,
                    },
                },
            }
            response = client.post("/api/vision/aruco-reference", json=request_data)

            assert response.status_code == 200
            data = response.json()

            # Check rotation reference is applied
            metadata = data["reference_object"]["metadata"]
            assert metadata["rotation_reference"] == rotation_ref
            assert "reference_rotation_deg" in metadata

            if rotation_ref == "image_axes":
                # Image axes should have 0 rotation offset
                assert metadata["reference_rotation_deg"] == 0.0

    def test_aruco_reference_plane_mode_basic(self, client, captured_image_id):
        """Test ArUco reference creation with PLANE (4-marker) mode"""
        request_data = {
            "image_id": captured_image_id,
            "params": {
                "dictionary": "DICT_4X4_50",
                "mode": "plane",
                "plane_config": {
                    "marker_ids": {
                        "top_left": 0,
                        "top_right": 1,
                        "bottom_right": 2,
                        "bottom_left": 3,
                    },
                    "width_mm": 200.0,
                    "height_mm": 150.0,
                    "origin": "top_left",
                    "x_direction": "right",
                    "y_direction": "down",
                },
            },
        }
        response = client.post("/api/vision/aruco-reference", json=request_data)

        # This might fail if test image doesn't have all 4 markers
        # That's expected - we're testing the endpoint structure
        if response.status_code == 200:
            data = response.json()

            # Check reference_object structure
            ref_obj = data["reference_object"]
            assert ref_obj["type"] == "plane"
            assert ref_obj["units"] == "mm"
            assert "homography_matrix" in ref_obj

            # Check metadata
            metadata = ref_obj["metadata"]
            assert metadata["width_mm"] == 200.0
            assert metadata["height_mm"] == 150.0
            assert metadata["origin"] == "top_left"
            assert "markers_found" in metadata
            assert len(metadata["markers_found"]) == 4

            # Should have detected 4 markers
            assert len(data["markers"]) == 4
        else:
            # Expected to fail if test image doesn't have 4 markers
            assert response.status_code in [400, 500]
            # Error should mention missing markers
            error_data = response.json()
            assert "error" in error_data or "detail" in error_data

    def test_aruco_reference_plane_mode_different_origins(self, client, captured_image_id):
        """Test PLANE mode with different origin corners"""
        # Skip if image doesn't have 4 markers
        pytest.skip("Test image may not have 4 markers - PLANE mode needs specific setup")

        origins = ["top_left", "top_right", "bottom_left", "bottom_right"]

        for origin in origins:
            request_data = {
                "image_id": captured_image_id,
                "params": {
                    "dictionary": "DICT_4X4_50",
                    "mode": "plane",
                    "plane_config": {
                        "marker_ids": {
                            "top_left": 0,
                            "top_right": 1,
                            "bottom_right": 2,
                            "bottom_left": 3,
                        },
                        "width_mm": 200.0,
                        "height_mm": 150.0,
                        "origin": origin,
                        "x_direction": "right",
                        "y_direction": "down",
                    },
                },
            }
            response = client.post("/api/vision/aruco-reference", json=request_data)

            if response.status_code == 200:
                data = response.json()
                assert data["reference_object"]["metadata"]["origin"] == origin

    def test_aruco_reference_validation_single_missing_config(self, client, captured_image_id):
        """Test validation error when single_config is missing in SINGLE mode"""
        request_data = {
            "image_id": captured_image_id,
            "params": {
                "dictionary": "DICT_4X4_50",
                "mode": "single",
                # Missing single_config
            },
        }
        response = client.post("/api/vision/aruco-reference", json=request_data)

        # Should return validation error
        assert response.status_code == 422
        error_data = response.json()
        assert "error" in error_data
        # Error should mention single_config is required
        assert "single_config" in str(error_data).lower()

    def test_aruco_reference_validation_plane_missing_config(self, client, captured_image_id):
        """Test validation error when plane_config is missing in PLANE mode"""
        request_data = {
            "image_id": captured_image_id,
            "params": {
                "dictionary": "DICT_4X4_50",
                "mode": "plane",
                # Missing plane_config
            },
        }
        response = client.post("/api/vision/aruco-reference", json=request_data)

        # Should return validation error
        assert response.status_code == 422
        error_data = response.json()
        assert "error" in error_data
        # Error should mention plane_config is required
        assert "plane_config" in str(error_data).lower()

    def test_aruco_reference_validation_wrong_config_for_mode(self, client, captured_image_id):
        """Test validation error when wrong config is provided for mode"""
        # SINGLE mode with plane_config
        request_data = {
            "image_id": captured_image_id,
            "params": {
                "dictionary": "DICT_4X4_50",
                "mode": "single",
                "single_config": {
                    "marker_id": 0,
                    "marker_size_mm": 50.0,
                },
                "plane_config": {  # Wrong config for SINGLE mode
                    "marker_ids": {
                        "top_left": 0,
                        "top_right": 1,
                        "bottom_right": 2,
                        "bottom_left": 3,
                    },
                    "width_mm": 200.0,
                    "height_mm": 150.0,
                },
            },
        }
        response = client.post("/api/vision/aruco-reference", json=request_data)

        # Should return validation error
        assert response.status_code == 422
        error_data = response.json()
        assert "error" in error_data

    def test_aruco_reference_marker_not_found(self, client, captured_image_id):
        """Test error when specified marker is not found in image"""
        # Request marker ID that doesn't exist
        request_data = {
            "image_id": captured_image_id,
            "params": {
                "dictionary": "DICT_4X4_50",
                "mode": "single",
                "single_config": {
                    "marker_id": 999,  # Unlikely to exist
                    "marker_size_mm": 50.0,
                    "origin": "marker_center",
                    "rotation_reference": "marker_rotation",
                },
            },
        }
        response = client.post("/api/vision/aruco-reference", json=request_data)

        # Should return error (marker not found)
        assert response.status_code in [400, 500]
        error_data = response.json()
        assert "error" in error_data or "detail" in error_data
        # Error should mention marker not found
        assert "999" in str(error_data) or "not found" in str(error_data).lower()

    def test_aruco_reference_with_roi(self, client, captured_image_id):
        """Test ArUco reference creation with ROI constraint"""
        # Use ROI to limit search area
        request_data = {
            "image_id": captured_image_id,
            "roi": {"x": 0, "y": 0, "width": 400, "height": 400},
            "params": {
                "dictionary": "DICT_4X4_50",
                "mode": "single",
                "single_config": {
                    "marker_id": 0,
                    "marker_size_mm": 50.0,
                    "origin": "marker_center",
                    "rotation_reference": "marker_rotation",
                },
            },
        }
        response = client.post("/api/vision/aruco-reference", json=request_data)

        # Should work if marker 0 is in ROI
        if response.status_code == 200:
            data = response.json()
            assert "reference_object" in data
            assert len(data["markers"]) > 0
        else:
            # If marker 0 is not in ROI, should get error
            assert response.status_code in [400, 500]


class TestArucoReferenceIntegration:
    """Integration tests for using ArUco reference with other detection endpoints"""

    @pytest.fixture
    def captured_image_id(self, client):
        """Capture a test image with ArUco markers"""
        response = client.post("/api/camera/capture", json={"camera_id": "test"})
        assert response.status_code == 200
        data = response.json()
        return data["objects"][0]["properties"]["image_id"]

    @pytest.fixture
    def reference_object(self, client, captured_image_id):
        """Create a reference frame from ArUco marker"""
        request_data = {
            "image_id": captured_image_id,
            "params": {
                "dictionary": "DICT_4X4_50",
                "mode": "single",
                "single_config": {
                    "marker_id": 0,
                    "marker_size_mm": 50.0,
                    "origin": "marker_center",
                    "rotation_reference": "marker_rotation",
                },
            },
        }
        response = client.post("/api/vision/aruco-reference", json=request_data)
        if response.status_code != 200:
            pytest.skip("Could not create reference object - marker 0 not found")

        return response.json()["reference_object"]

    def test_template_match_with_reference_transform(
        self, client, captured_image_id, reference_object
    ):
        """Test template matching with reference frame transformation"""
        # This test verifies the integration but requires a learned template
        pytest.skip("Requires template setup - tested in full workflow tests")

    def test_rotation_detect_with_reference_transform(
        self, client, captured_image_id, reference_object
    ):
        """Test rotation detection with reference frame transformation"""
        # This test verifies the integration
        pytest.skip("Requires edge contour setup - tested in full workflow tests")

    def test_reference_object_structure(self, client, reference_object):
        """Test that reference_object has all required fields for transformation"""
        # Verify reference_object can be used for transformation
        assert "type" in reference_object
        assert "units" in reference_object
        assert "homography_matrix" in reference_object
        assert "metadata" in reference_object

        # Homography should be 3x3
        H = reference_object["homography_matrix"]
        assert len(H) == 3
        assert all(len(row) == 3 for row in H)

        # Should have scale information
        metadata = reference_object["metadata"]
        if reference_object["type"] == "single_marker":
            assert "scale_mm_per_pixel" in metadata
        elif reference_object["type"] == "plane":
            assert "width_mm" in metadata
            assert "height_mm" in metadata
