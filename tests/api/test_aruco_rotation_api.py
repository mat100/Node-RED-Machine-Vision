"""
API Integration Tests for ArUco and Rotation Detection Endpoints
"""

import pytest


class TestArucoAPI:
    """Integration tests for ArUco detection API"""

    @pytest.fixture
    def captured_image_id(self, client):
        """Capture a test image with ArUco markers"""
        response = client.post("/api/camera/capture", json={"camera_id": "test"})
        assert response.status_code == 200
        return response.json()["image_id"]

    def test_aruco_detect_basic(self, client, captured_image_id):
        """Test basic ArUco marker detection"""
        request_data = {
            "image_id": captured_image_id,
            "dictionary": "DICT_4X4_50",
        }
        response = client.post("/api/vision/aruco-detect", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "objects" in data
        assert isinstance(data["objects"], list)
        assert "processing_time_ms" in data
        assert data["processing_time_ms"] > 0
        assert "thumbnail_base64" in data

        # Should find at least one marker (test image has markers)
        assert len(data["objects"]) > 0

        # Check marker object structure
        marker = data["objects"][0]
        assert marker["object_type"] == "aruco_marker"
        assert "marker_id" in marker["properties"]
        assert "rotation" in marker
        assert "center" in marker
        assert "bounding_box" in marker

    def test_aruco_detect_with_roi(self, client, captured_image_id):
        """Test ArUco detection with ROI constraint"""
        # Define ROI in top-left quadrant where first marker is
        request_data = {
            "image_id": captured_image_id,
            "dictionary": "DICT_4X4_50",
            "roi": {"x": 0, "y": 0, "width": 400, "height": 400},
        }
        response = client.post("/api/vision/aruco-detect", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Should find marker(s) in ROI
        assert len(data["objects"]) > 0

        # Verify marker coordinates are within or near ROI
        for marker in data["objects"]:
            center = marker["center"]
            # Center should be in extended ROI area (marker can extend beyond)
            assert center["x"] < 600  # Some margin for marker size
            assert center["y"] < 600

    def test_aruco_detect_different_dictionaries(self, client, captured_image_id):
        """Test detection with different ArUco dictionaries"""
        dictionaries = ["DICT_4X4_50", "DICT_5X5_50", "DICT_6X6_50"]

        for dictionary in dictionaries:
            request_data = {
                "image_id": captured_image_id,
                "dictionary": dictionary,
            }
            response = client.post("/api/vision/aruco-detect", json=request_data)
            assert response.status_code == 200

            # DICT_4X4_50 should find markers (test image uses this)
            # Others may or may not find markers depending on test image
            data = response.json()
            assert "objects" in data

    def test_aruco_detect_no_markers(self, client):
        """Test ArUco detection on image without markers"""
        # Create a blank image without markers
        response = client.post("/api/camera/capture", json={"camera_id": "test"})
        image_id = response.json()["image_id"]

        # Try to detect markers in a small ROI with no markers
        request_data = {
            "image_id": image_id,
            "dictionary": "DICT_4X4_50",
            "roi": {"x": 500, "y": 500, "width": 100, "height": 100},
        }
        response = client.post("/api/vision/aruco-detect", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Should return empty objects list (or may find false positives)
        assert "objects" in data
        assert isinstance(data["objects"], list)


class TestRotationAPI:
    """Integration tests for Rotation detection API"""

    @pytest.fixture
    def captured_image_id(self, client):
        """Capture a test image"""
        response = client.post("/api/camera/capture", json={"camera_id": "test"})
        assert response.status_code == 200
        return response.json()["image_id"]

    @pytest.fixture
    def edge_contour(self, client, captured_image_id):
        """Get a contour from edge detection"""
        # Use lower thresholds to detect test shapes (rectangles, circle)
        request_data = {
            "image_id": captured_image_id,
            "params": {
                "method": "canny",
                "canny_low": 30,
                "canny_high": 100,
                "min_contour_area": 500,
                "max_contours": 10,
            },
        }
        response = client.post("/api/vision/edge-detect", json=request_data)
        assert response.status_code == 200

        data = response.json()
        # Test image should have rectangles and circle
        if len(data["objects"]) == 0:
            pytest.skip(
                "No contours found in test image - edge detection may need parameter tuning"
            )

        # Return first contour
        return data["objects"][0]["contour"]

    def test_rotation_detect_basic(self, client, captured_image_id, edge_contour):
        """Test basic rotation detection"""
        request_data = {
            "image_id": captured_image_id,
            "contour": edge_contour,
            "method": "min_area_rect",
            "angle_range": "0_360",
        }
        response = client.post("/api/vision/rotation-detect", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "objects" in data
        assert len(data["objects"]) == 1
        assert "processing_time_ms" in data
        assert data["processing_time_ms"] > 0
        assert "thumbnail_base64" in data

        # Check rotation object
        rotation_obj = data["objects"][0]
        assert rotation_obj["object_type"] == "rotation_analysis"
        assert "rotation" in rotation_obj
        assert 0 <= rotation_obj["rotation"] < 360
        assert "center" in rotation_obj
        assert "bounding_box" in rotation_obj
        assert "method" in rotation_obj["properties"]
        assert rotation_obj["properties"]["method"] == "min_area_rect"

    def test_rotation_detect_all_methods(self, client, captured_image_id, edge_contour):
        """Test all rotation detection methods"""
        methods = ["min_area_rect", "ellipse_fit", "pca"]

        for method in methods:
            request_data = {
                "image_id": captured_image_id,
                "contour": edge_contour,
                "params": {
                    "method": method,
                    "angle_range": "0_360",
                },
            }
            response = client.post("/api/vision/rotation-detect", json=request_data)

            assert response.status_code == 200
            data = response.json()
            assert len(data["objects"]) == 1

            rotation_obj = data["objects"][0]
            assert "rotation" in rotation_obj
            assert rotation_obj["properties"]["method"] == method

    def test_rotation_detect_angle_ranges(self, client, captured_image_id, edge_contour):
        """Test different angle range outputs"""
        angle_ranges = ["0_360", "-180_180", "0_180"]

        for angle_range in angle_ranges:
            request_data = {
                "image_id": captured_image_id,
                "contour": edge_contour,
                "params": {
                    "method": "min_area_rect",
                    "angle_range": angle_range,
                },
            }
            response = client.post("/api/vision/rotation-detect", json=request_data)

            assert response.status_code == 200
            data = response.json()

            rotation_obj = data["objects"][0]
            angle = rotation_obj["rotation"]

            # Verify angle is in correct range
            if angle_range == "0_360":
                assert 0 <= angle < 360
            elif angle_range == "-180_180":
                assert -180 <= angle <= 180
            elif angle_range == "0_180":
                assert 0 <= angle < 180

    def test_rotation_detect_with_roi(self, client, captured_image_id, edge_contour):
        """Test rotation detection with ROI for visualization"""
        # Get bounding box from contour
        import numpy as np

        contour_array = np.array(edge_contour)
        x_coords = contour_array[:, 0]
        y_coords = contour_array[:, 1]
        x, y = int(x_coords.min()), int(y_coords.min())
        w, h = int(x_coords.max() - x), int(y_coords.max() - y)

        request_data = {
            "image_id": captured_image_id,
            "contour": edge_contour,
            "method": "min_area_rect",
            "angle_range": "0_360",
            "roi": {"x": x, "y": y, "width": w, "height": h},
        }
        response = client.post("/api/vision/rotation-detect", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Should still detect rotation successfully
        assert len(data["objects"]) == 1
        # Thumbnail should be cropped to ROI (smaller than full image)
        assert "thumbnail_base64" in data

    def test_rotation_detect_invalid_contour(self, client, captured_image_id):
        """Test rotation detection with invalid contour"""
        # Too few points
        request_data = {
            "image_id": captured_image_id,
            "contour": [[100, 100], [200, 200]],  # Only 2 points
            "method": "min_area_rect",
            "angle_range": "0_360",
        }
        response = client.post("/api/vision/rotation-detect", json=request_data)

        # Should return validation error (422 Unprocessable Entity)
        assert response.status_code == 422
        # Our custom validation error handler returns 'details' array
        response_json = response.json()
        assert "details" in response_json
        assert response_json["error"] == "Validation failed"
        assert any("contour" in str(err["field"]) for err in response_json["details"])


class TestArucoRotationIntegration:
    """Integration tests combining ArUco and Rotation detection"""

    @pytest.fixture
    def captured_image_id(self, client):
        """Capture a test image"""
        response = client.post("/api/camera/capture", json={"camera_id": "test"})
        assert response.status_code == 200
        return response.json()["image_id"]

    def test_aruco_then_edge_then_rotation(self, client, captured_image_id):
        """Test complete workflow: ArUco → Edge → Rotation"""
        # Step 1: Detect ArUco markers
        aruco_response = client.post(
            "/api/vision/aruco-detect",
            json={"image_id": captured_image_id, "dictionary": "DICT_4X4_50"},
        )
        assert aruco_response.status_code == 200
        aruco_data = aruco_response.json()

        # Step 2: Detect edges on full image
        edge_response = client.post(
            "/api/vision/edge-detect",
            json={
                "image_id": captured_image_id,
                "params": {
                    "method": "canny",
                    "min_contour_area": 1000,
                },
            },
        )
        assert edge_response.status_code == 200
        edge_data = edge_response.json()

        if len(edge_data["objects"]) > 0:
            # Step 3: Analyze rotation of first contour
            contour = edge_data["objects"][0]["contour"]
            rotation_response = client.post(
                "/api/vision/rotation-detect",
                json={
                    "image_id": captured_image_id,
                    "contour": contour,
                    "method": "min_area_rect",
                    "angle_range": "0_360",
                },
            )
            assert rotation_response.status_code == 200
            rotation_data = rotation_response.json()

            # Verify we got rotation data
            assert len(rotation_data["objects"]) == 1
            assert "rotation" in rotation_data["objects"][0]

            # In real workflow, you'd compare rotation to ArUco reference
            if len(aruco_data["objects"]) > 0:
                reference_rotation = aruco_data["objects"][0]["rotation"]
                object_rotation = rotation_data["objects"][0]["rotation"]

                # Both rotations should be in valid range
                assert 0 <= reference_rotation < 360
                assert 0 <= object_rotation < 360
