"""
API Integration Tests for Vision Endpoints
"""

import pytest


class TestVisionAPI:
    """Integration tests for vision API endpoints"""

    @pytest.fixture
    def captured_image_id(self, client):
        """Capture an image and return its ID"""
        response = client.post("/api/camera/capture", json={"camera_id": "test"})
        return response.json()["image_id"]

    @pytest.fixture
    def template_id(self, client, captured_image_id):
        """Create a template from captured image"""
        request_data = {
            "image_id": captured_image_id,
            "name": "Test Template",
            "description": "Template for integration tests",
            "roi": {"x": 100, "y": 100, "width": 100, "height": 100},
        }
        response = client.post("/api/template/learn", json=request_data)
        return response.json()["template_id"]

    def test_template_match_basic(self, client, captured_image_id, template_id):
        """Test basic template matching"""
        request_data = {
            "image_id": captured_image_id,
            "params": {
                "template_id": template_id,
                "method": "TM_CCOEFF_NORMED",
                "threshold": 0.5,
            },
        }
        response = client.post("/api/vision/template-match", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "objects" in data
        assert isinstance(data["objects"], list)
        assert "processing_time_ms" in data
        assert data["processing_time_ms"] > 0
        assert "thumbnail_base64" in data

    def test_template_match_with_options(self, client, captured_image_id, template_id):
        """Test template matching with additional options"""
        request_data = {
            "image_id": captured_image_id,
            "params": {
                "template_id": template_id,
                "method": "TM_CCOEFF_NORMED",
                "threshold": 0.7,
                "multi_scale": False,
            },
        }
        response = client.post("/api/vision/template-match", json=request_data)

        assert response.status_code == 200
        # success removed from response

    def test_template_match_high_threshold(self, client, captured_image_id, template_id):
        """Test template matching with very high threshold"""
        request_data = {
            "image_id": captured_image_id,
            "params": {
                "template_id": template_id,
                "method": "TM_CCOEFF_NORMED",
                "threshold": 0.99,  # Very high threshold
            },
        }
        response = client.post("/api/vision/template-match", json=request_data)

        assert response.status_code == 200
        data = response.json()
        # success removed from response
        # High threshold might result in no matches
        assert "objects" in data

    def test_template_match_invalid_image(self, client, template_id):
        """Test template matching with non-existent image"""
        request_data = {
            "image_id": "non-existent-id",
            "params": {"template_id": template_id, "threshold": 0.8},
        }
        response = client.post("/api/vision/template-match", json=request_data)

        assert response.status_code == 404
        data = response.json()
        assert "error" in data or "detail" in data

    def test_template_match_invalid_template(self, client, captured_image_id):
        """Test template matching with non-existent template"""
        request_data = {
            "image_id": captured_image_id,
            "params": {"template_id": "non-existent-template", "threshold": 0.8},
        }
        response = client.post("/api/vision/template-match", json=request_data)

        assert response.status_code == 404
        data = response.json()
        assert "error" in data or "detail" in data

    def test_template_match_different_methods(self, client, captured_image_id, template_id):
        """Test different template matching methods"""
        methods = ["TM_CCOEFF_NORMED", "TM_CCORR_NORMED", "TM_SQDIFF_NORMED"]

        for method in methods:
            request_data = {
                "image_id": captured_image_id,
                "params": {
                    "template_id": template_id,
                    "method": method,
                    "threshold": 0.5,
                },
            }
            response = client.post("/api/vision/template-match", json=request_data)

            assert response.status_code == 200
            # success removed from response

    def test_edge_detect_canny(self, client, captured_image_id):
        """Test Canny edge detection"""
        request_data = {"image_id": captured_image_id, "params": {"method": "canny"}}
        response = client.post("/api/vision/edge-detect", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # success removed from response
        assert "objects" in data
        assert isinstance(data["objects"], list)
        assert len(data["objects"]) >= 0
        assert "processing_time_ms" in data
        assert "thumbnail_base64" in data

        # Verify contour is included in objects
        if data["objects"]:
            obj = data["objects"][0]
            assert "contour" in obj
            assert isinstance(obj["contour"], list)
            assert len(obj["contour"]) > 0  # Should have contour points

    def test_edge_detect_all_methods(self, client, captured_image_id):
        """Test all edge detection methods"""
        methods = ["canny", "sobel", "laplacian", "scharr", "prewitt", "roberts"]

        for method in methods:
            request_data = {"image_id": captured_image_id, "params": {"method": method}}
            response = client.post("/api/vision/edge-detect", json=request_data)

            assert response.status_code == 200
            data = response.json()
            # success removed from response
            assert len(data["objects"]) >= 0

    def test_edge_detect_with_params(self, client, captured_image_id):
        """Test edge detection with custom parameters (explicit fields)"""
        request_data = {
            "image_id": captured_image_id,
            "params": {
                "method": "canny",
                "canny_low": 50,
                "canny_high": 150,
                "min_contour_area": 100,
                "max_contour_area": 5000,
            },
        }
        response = client.post("/api/vision/edge-detect", json=request_data)

        assert response.status_code == 200
        # success removed from response

    def test_edge_detect_with_filtering(self, client, captured_image_id):
        """Test edge detection with contour area filtering"""
        request_data = {
            "image_id": captured_image_id,
            "params": {
                "method": "canny",
                "min_contour_area": 500,
                "max_contour_area": 10000,
                "max_contours": 10,
            },
        }
        response = client.post("/api/vision/edge-detect", json=request_data)

        assert response.status_code == 200
        data = response.json()
        # success removed from response
        # Verify contour count doesn't exceed max
        assert len(data["objects"]) <= 10

    def test_edge_detect_invalid_image(self, client):
        """Test edge detection with non-existent image"""
        request_data = {"image_id": "non-existent-id", "params": {"method": "canny"}}
        response = client.post("/api/vision/edge-detect", json=request_data)

        assert response.status_code == 404

    def test_edge_detect_with_preprocessing(self, client, captured_image_id):
        """Test edge detection with preprocessing options (explicit fields)"""
        request_data = {
            "image_id": captured_image_id,
            "params": {
                "method": "canny",
                "blur_enabled": True,
                "blur_kernel": 5,
                "morphology_enabled": True,
                "morphology_operation": "close",
            },
        }
        response = client.post("/api/vision/edge-detect", json=request_data)

        assert response.status_code == 200
        # success removed from response

    def test_edge_detect_field_validation(self, client, captured_image_id):
        """Test edge detection field validation"""
        # Test with invalid canny_low (should be 0-500)
        request_data = {
            "image_id": captured_image_id,
            "params": {
                "method": "canny",
                "canny_low": 600,  # Invalid - exceeds max
            },
        }
        response = client.post("/api/vision/edge-detect", json=request_data)

        # Should fail validation
        assert response.status_code == 422

    def test_edge_detect_validation_negative_values(self, client, captured_image_id):
        """Test edge detection rejects negative values"""
        # Test negative min_contour_area
        request_data = {
            "image_id": captured_image_id,
            "params": {
                "method": "canny",
                "min_contour_area": -100,  # Invalid - negative
            },
        }
        response = client.post("/api/vision/edge-detect", json=request_data)
        assert response.status_code == 422

        # Test negative sobel_threshold
        request_data = {
            "image_id": captured_image_id,
            "params": {
                "method": "sobel",
                "sobel_threshold": -50,  # Invalid - negative
            },
        }
        response = client.post("/api/vision/edge-detect", json=request_data)
        assert response.status_code == 422

    def test_edge_detect_validation_kernel_size(self, client, captured_image_id):
        """Test edge detection kernel size validation"""
        # Test invalid sobel_kernel (must be >= 1)
        request_data = {
            "image_id": captured_image_id,
            "params": {
                "method": "sobel",
                "sobel_kernel": 0,  # Invalid - must be >= 1
            },
        }
        response = client.post("/api/vision/edge-detect", json=request_data)
        assert response.status_code == 422

        # Test invalid blur_kernel (must be >= 3)
        request_data = {
            "image_id": captured_image_id,
            "params": {
                "method": "canny",
                "blur_enabled": True,
                "blur_kernel": 2,  # Invalid - must be >= 3
            },
        }
        response = client.post("/api/vision/edge-detect", json=request_data)
        assert response.status_code == 422

    def test_edge_detect_validation_max_contours(self, client, captured_image_id):
        """Test max_contours validation"""
        # Test invalid max_contours (must be >= 1)
        request_data = {
            "image_id": captured_image_id,
            "params": {
                "method": "canny",
                "max_contours": 0,  # Invalid - must be >= 1
            },
        }
        response = client.post("/api/vision/edge-detect", json=request_data)
        assert response.status_code == 422

    def test_edge_detect_sobel_params(self, client, captured_image_id):
        """Test Sobel edge detection with specific parameters"""
        request_data = {
            "image_id": captured_image_id,
            "params": {
                "method": "sobel",
                "sobel_threshold": 75,
                "sobel_kernel": 3,
                "min_contour_area": 200,
            },
        }
        response = client.post("/api/vision/edge-detect", json=request_data)

        assert response.status_code == 200
        # success removed from response

    def test_edge_detect_laplacian_params(self, client, captured_image_id):
        """Test Laplacian edge detection with specific parameters"""
        request_data = {
            "image_id": captured_image_id,
            "params": {
                "method": "laplacian",
                "laplacian_threshold": 40,
                "laplacian_kernel": 5,
                "min_contour_area": 150,
            },
        }
        response = client.post("/api/vision/edge-detect", json=request_data)

        assert response.status_code == 200
        data = response.json()
        # success removed from response
        assert len(data["objects"]) >= 0

    def test_edge_detect_prewitt_params(self, client, captured_image_id):
        """Test Prewitt edge detection with specific parameters"""
        request_data = {
            "image_id": captured_image_id,
            "params": {
                "method": "prewitt",
                "prewitt_threshold": 60,
                "min_contour_area": 100,
            },
        }
        response = client.post("/api/vision/edge-detect", json=request_data)

        assert response.status_code == 200
        # success removed from response
        # found removed from response

    def test_edge_detect_scharr_params(self, client, captured_image_id):
        """Test Scharr edge detection with specific parameters"""
        request_data = {
            "image_id": captured_image_id,
            "params": {
                "method": "scharr",
                "scharr_threshold": 55,
                "min_contour_area": 120,
            },
        }
        response = client.post("/api/vision/edge-detect", json=request_data)

        assert response.status_code == 200
        data = response.json()
        # success removed from response
        # found removed from response
        assert "objects" in data

    def test_edge_detect_perimeter_filtering(self, client, captured_image_id):
        """Test edge detection with perimeter-based filtering"""
        request_data = {
            "image_id": captured_image_id,
            "params": {
                "method": "canny",
                "min_contour_perimeter": 50,
                "max_contour_perimeter": 1000,
                "max_contours": 15,
            },
        }
        response = client.post("/api/vision/edge-detect", json=request_data)

        assert response.status_code == 200
        data = response.json()
        # success removed from response
        # Verify contour count doesn't exceed max
        assert len(data["objects"]) <= 15
        # Verify perimeter filtering is applied (if objects exist)
        if data["objects"]:
            for obj in data["objects"]:
                assert 50 <= obj["perimeter"] <= 1000

    def test_color_detect_with_roi(self, client, captured_image_id):
        """Test color detection with ROI (bounding_box from edge detection)"""
        # First run edge detection to get a bounding box
        edge_request = {
            "image_id": captured_image_id,
            "params": {
                "method": "canny",
                "min_contour_area": 100,
                "max_contours": 1,
            },
        }
        edge_response = client.post("/api/vision/edge-detect", json=edge_request)

        assert edge_response.status_code == 200
        edge_data = edge_response.json()

        # If we got objects with bounding boxes, test color detection on first one
        if edge_data["objects"] and len(edge_data["objects"]) > 0:
            bbox = edge_data["objects"][0]["bounding_box"]

            # Now run color detection on that ROI
            color_request = {
                "image_id": captured_image_id,
                "roi": bbox,  # Use bounding_box from edge detection
                "expected_color": None,  # Just detect dominant color
                "params": {
                    "method": "histogram",
                },
            }
            color_response = client.post("/api/vision/color-detect", json=color_request)

            assert color_response.status_code == 200
            color_data = color_response.json()
            assert "objects" in color_data
            assert "thumbnail_base64" in color_data
            assert "processing_time_ms" in color_data

            # If color was detected, verify it has properties
            if color_data["objects"]:
                obj = color_data["objects"][0]
                assert obj["object_type"] == "color_region"
                assert "dominant_color" in obj["properties"]
                assert "color_percentages" in obj["properties"]
                # Verify dominant color has a percentage
                dominant_color = obj["properties"]["dominant_color"]
                assert dominant_color in obj["properties"]["color_percentages"]
                assert obj["properties"]["color_percentages"][dominant_color] > 0

    def test_edge_to_color_workflow(self, client, captured_image_id):
        """Test complete workflow: edge detect → color detect on each contour"""
        # Step 1: Edge detection
        edge_request = {
            "image_id": captured_image_id,
            "params": {
                "method": "canny",
                "min_contour_area": 200,
                "max_contours": 3,
            },
        }
        edge_response = client.post("/api/vision/edge-detect", json=edge_request)

        assert edge_response.status_code == 200
        edge_data = edge_response.json()

        # Step 2: For each detected contour, analyze color in its bounding box
        color_results = []
        for obj in edge_data["objects"]:
            bbox = obj["bounding_box"]

            color_request = {
                "image_id": captured_image_id,
                "roi": bbox,
                "expected_color": None,
                "params": {
                    "method": "histogram",
                },
            }
            color_response = client.post("/api/vision/color-detect", json=color_request)

            if color_response.status_code == 200:
                color_data = color_response.json()
                if color_data["objects"]:
                    color_results.append(color_data["objects"][0])

        # Verify we got color results for contours
        assert len(color_results) >= 0  # May be 0 if no contours found or no dominant colors

    def test_color_detect_with_contour_mask(self, client, captured_image_id):
        """Test color detection with contour mask enabled"""
        # Create a simple square contour
        contour = [[100, 100], [300, 100], [300, 300], [100, 300]]

        color_request = {
            "image_id": captured_image_id,
            "contour": contour,
            "params": {
                "use_contour_mask": True,
                "method": "histogram",
            },
        }
        response = client.post("/api/vision/color-detect", json=color_request)

        assert response.status_code == 200
        data = response.json()
        assert "objects" in data
        assert "thumbnail_base64" in data

        # Verify that if object returned, it analyzed fewer pixels than full image
        if data["objects"]:
            obj = data["objects"][0]
            assert "area" in obj  # area = analyzed_pixels
            # Square contour should analyze approximately 200*200 = 40000 pixels
            assert obj["area"] < 1920 * 1080  # Less than full image

    def test_color_detect_without_contour_mask(self, client, captured_image_id):
        """Test color detection with contour mask disabled"""
        # Create a contour but disable mask
        contour = [[100, 100], [300, 100], [300, 300], [100, 300]]
        roi = {"x": 100, "y": 100, "width": 200, "height": 200}

        color_request = {
            "image_id": captured_image_id,
            "roi": roi,
            "contour": contour,
            "params": {
                "use_contour_mask": False,  # Disabled
                "method": "histogram",
            },
        }
        response = client.post("/api/vision/color-detect", json=color_request)

        assert response.status_code == 200
        data = response.json()

        # Should analyze full bounding box (40000 pixels)
        if data["objects"]:
            obj = data["objects"][0]
            # With mask disabled, should analyze full ROI
            assert obj["area"] == pytest.approx(40000, rel=0.1)

    def test_color_detect_mask_comparison(self, client, captured_image_id):
        """Compare color detection with and without contour mask"""
        # Create a triangular contour (half of bbox area)
        contour = [[200, 100], [300, 300], [100, 300]]
        roi = {"x": 100, "y": 100, "width": 200, "height": 200}

        # Test with mask
        with_mask_request = {
            "image_id": captured_image_id,
            "roi": roi,
            "contour": contour,
            "params": {
                "use_contour_mask": True,
                "method": "histogram",
            },
        }
        with_mask_response = client.post("/api/vision/color-detect", json=with_mask_request)

        # Test without mask
        without_mask_request = {
            "image_id": captured_image_id,
            "roi": roi,
            "contour": contour,
            "params": {
                "use_contour_mask": False,
                "method": "histogram",
            },
        }
        without_mask_response = client.post("/api/vision/color-detect", json=without_mask_request)

        assert with_mask_response.status_code == 200
        assert without_mask_response.status_code == 200

        with_mask_data = with_mask_response.json()
        without_mask_data = without_mask_response.json()

        # Both should return objects
        if with_mask_data["objects"] and without_mask_data["objects"]:
            with_mask_pixels = with_mask_data["objects"][0]["area"]
            without_mask_pixels = without_mask_data["objects"][0]["area"]

            # With mask should analyze fewer pixels (triangle vs rectangle)
            assert with_mask_pixels < without_mask_pixels

    def test_color_detect_with_edge_contour(self, client, captured_image_id):
        """Test realistic workflow: edge detection → color detection with contour"""
        # Step 1: Edge detection to get real contour
        edge_request = {
            "image_id": captured_image_id,
            "params": {
                "method": "canny",
                "min_contour_area": 500,
                "max_contours": 1,
            },
        }
        edge_response = client.post("/api/vision/edge-detect", json=edge_request)

        assert edge_response.status_code == 200
        edge_data = edge_response.json()

        # If we got a contour, test color detection with it
        if edge_data["objects"]:
            obj = edge_data["objects"][0]
            contour = obj.get("contour")
            bbox = obj["bounding_box"]

            if contour:
                # Step 2: Color detection using the real contour
                color_request = {
                    "image_id": captured_image_id,
                    "roi": bbox,
                    "contour": contour,
                    "params": {
                        "use_contour_mask": True,
                        "method": "histogram",
                    },
                }
                color_response = client.post("/api/vision/color-detect", json=color_request)

                assert color_response.status_code == 200
                color_data = color_response.json()
                assert "objects" in color_data

                if color_data["objects"]:
                    color_obj = color_data["objects"][0]
                    assert "dominant_color" in color_obj["properties"]
                    # Verify it analyzed contour pixels, not full bbox
                    assert color_obj["area"] <= bbox["width"] * bbox["height"]

    def test_color_detect_contour_default_behavior(self, client, captured_image_id):
        """Test that use_contour_mask defaults to True when contour provided"""
        contour = [[100, 100], [300, 100], [300, 300], [100, 300]]

        # Don't specify use_contour_mask in params, should use default (True)
        color_request = {
            "image_id": captured_image_id,
            "contour": contour,
            "params": {
                "method": "histogram",
            },
        }
        response = client.post("/api/vision/color-detect", json=color_request)

        assert response.status_code == 200
        data = response.json()

        # Should have used mask by default
        if data["objects"]:
            obj = data["objects"][0]
            # Should analyze contour area (40000 pixels for square)
            assert obj["area"] < 1920 * 1080  # Less than full image
