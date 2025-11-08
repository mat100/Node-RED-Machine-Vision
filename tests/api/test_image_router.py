"""
API Integration Tests for Image Router Endpoints
"""

import base64

import pytest


class TestImageRouterAPI:
    """Integration tests for image router endpoints"""

    @pytest.fixture
    def captured_image_id(self, client):
        """Capture a test image and return its ID"""
        response = client.post("/api/camera/capture", json={"camera_id": "test"})
        assert response.status_code == 200
        return response.json()["image_id"]

    @pytest.fixture
    def large_image_id(self, client):
        """Create a large test image (>320px wide) for thumbnail scaling tests"""
        # Test camera generates 640x480 images by default
        response = client.post("/api/camera/capture", json={"camera_id": "test"})
        assert response.status_code == 200
        return response.json()["image_id"]

    def test_extract_roi_basic(self, client, captured_image_id):
        """Test basic ROI extraction"""
        request_data = {
            "image_id": captured_image_id,
            "roi": {"x": 50, "y": 50, "width": 100, "height": 100},
        }
        response = client.post("/api/image/extract-roi", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "success" in data
        assert data["success"] is True
        assert "thumbnail" in data
        assert "bounding_box" in data

        # Check thumbnail is valid base64
        assert isinstance(data["thumbnail"], str)
        assert len(data["thumbnail"]) > 0
        # Should be decodable
        decoded = base64.b64decode(data["thumbnail"])
        assert len(decoded) > 0

        # Check bounding box matches request
        bbox = data["bounding_box"]
        assert bbox["x"] == 50
        assert bbox["y"] == 50
        assert bbox["width"] == 100
        assert bbox["height"] == 100

    def test_extract_roi_full_image(self, client, captured_image_id):
        """Test ROI extraction covering full image"""
        # Test image is 640x480
        request_data = {
            "image_id": captured_image_id,
            "roi": {"x": 0, "y": 0, "width": 640, "height": 480},
        }
        response = client.post("/api/image/extract-roi", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        bbox = data["bounding_box"]
        assert bbox["x"] == 0
        assert bbox["y"] == 0
        assert bbox["width"] == 640
        assert bbox["height"] == 480

    def test_extract_roi_clipping_beyond_bounds(self, client, captured_image_id):
        """Test ROI extraction with ROI extending beyond image bounds"""
        # Make request that we know will extend beyond typical boundaries
        # Even if image is large, request something guaranteed to extend past
        request_data = {
            "image_id": captured_image_id,
            "roi": {"x": 500, "y": 400, "width": 5000, "height": 5000},
        }
        response = client.post("/api/image/extract-roi", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Bounding box should be clipped - width/height should be less than requested
        bbox = data["bounding_box"]
        assert bbox["x"] >= 500  # May be clipped if image is smaller
        assert bbox["y"] >= 400  # May be clipped if image is smaller
        # Width/height should definitely be clipped from 5000
        assert bbox["width"] < 5000
        assert bbox["height"] < 5000
        # And should be positive
        assert bbox["width"] > 0
        assert bbox["height"] > 0

    def test_extract_roi_completely_outside_bounds(self, client, captured_image_id):
        """Test ROI extraction with ROI starting way outside image bounds"""
        # ROI starting way outside any reasonable image bounds
        # This results in clipped dimensions of 0x0, which fails validation
        request_data = {
            "image_id": captured_image_id,
            "roi": {"x": 10000, "y": 10000, "width": 100, "height": 100},
        }
        response = client.post("/api/image/extract-roi", json=request_data)

        # Should fail with 400 because clipped ROI would have zero dimensions
        assert response.status_code == 400
        data = response.json()
        assert "error" in data or "detail" in data

    def test_extract_roi_thumbnail_scaling_large(self, client, large_image_id):
        """Test thumbnail scaling for ROI wider than 320px"""
        # Extract ROI that's 400px wide (should be scaled down to 320px)
        request_data = {
            "image_id": large_image_id,
            "roi": {"x": 100, "y": 100, "width": 400, "height": 300},
        }
        response = client.post("/api/image/extract-roi", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Thumbnail should be generated (can't easily check exact dimensions without decoding)
        assert len(data["thumbnail"]) > 0

        # Bounding box should still reflect original dimensions
        bbox = data["bounding_box"]
        assert bbox["width"] == 400
        assert bbox["height"] == 300

    def test_extract_roi_thumbnail_no_scaling_small(self, client, captured_image_id):
        """Test no thumbnail scaling for ROI narrower than 320px"""
        # Extract small ROI (should not be scaled)
        request_data = {
            "image_id": captured_image_id,
            "roi": {"x": 100, "y": 100, "width": 150, "height": 100},
        }
        response = client.post("/api/image/extract-roi", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Thumbnail should be generated
        assert len(data["thumbnail"]) > 0

        # Bounding box should match request
        bbox = data["bounding_box"]
        assert bbox["width"] == 150
        assert bbox["height"] == 100

    def test_extract_roi_at_image_edge(self, client, captured_image_id):
        """Test ROI extraction at image edges"""
        # Top-left corner
        request_data = {
            "image_id": captured_image_id,
            "roi": {"x": 0, "y": 0, "width": 50, "height": 50},
        }
        response = client.post("/api/image/extract-roi", json=request_data)
        assert response.status_code == 200
        assert response.json()["success"] is True

        # Bottom-right corner (640x480 image)
        request_data = {
            "image_id": captured_image_id,
            "roi": {"x": 590, "y": 430, "width": 50, "height": 50},
        }
        response = client.post("/api/image/extract-roi", json=request_data)
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_extract_roi_invalid_image_id(self, client):
        """Test ROI extraction with non-existent image"""
        request_data = {
            "image_id": "non-existent-image-id",
            "roi": {"x": 100, "y": 100, "width": 100, "height": 100},
        }
        response = client.post("/api/image/extract-roi", json=request_data)

        assert response.status_code == 404
        data = response.json()
        assert "error" in data or "detail" in data

    def test_extract_roi_negative_coordinates(self, client, captured_image_id):
        """Test validation of negative ROI coordinates"""
        # Negative x coordinate - should fail validation
        request_data = {
            "image_id": captured_image_id,
            "roi": {"x": -10, "y": 50, "width": 100, "height": 100},
        }
        response = client.post("/api/image/extract-roi", json=request_data)

        # Should fail Pydantic validation (x must be >= 0)
        assert response.status_code == 422

    def test_extract_roi_zero_width(self, client, captured_image_id):
        """Test validation of zero width ROI"""
        # Zero width - should fail validation
        request_data = {
            "image_id": captured_image_id,
            "roi": {"x": 100, "y": 100, "width": 0, "height": 100},
        }
        response = client.post("/api/image/extract-roi", json=request_data)

        # Should fail Pydantic validation (width must be > 0)
        assert response.status_code == 422

    def test_extract_roi_zero_height(self, client, captured_image_id):
        """Test validation of zero height ROI"""
        # Zero height - should fail validation
        request_data = {
            "image_id": captured_image_id,
            "roi": {"x": 100, "y": 100, "width": 100, "height": 0},
        }
        response = client.post("/api/image/extract-roi", json=request_data)

        # Should fail Pydantic validation (height must be > 0)
        assert response.status_code == 422

    def test_extract_roi_missing_fields(self, client, captured_image_id):
        """Test validation with missing required fields"""
        # Missing roi field
        request_data = {"image_id": captured_image_id}
        response = client.post("/api/image/extract-roi", json=request_data)

        assert response.status_code == 422

        # Missing image_id field
        request_data = {"roi": {"x": 100, "y": 100, "width": 100, "height": 100}}
        response = client.post("/api/image/extract-roi", json=request_data)

        assert response.status_code == 422

    def test_extract_roi_single_pixel(self, client, captured_image_id):
        """Test ROI extraction of single pixel"""
        request_data = {
            "image_id": captured_image_id,
            "roi": {"x": 320, "y": 240, "width": 1, "height": 1},
        }
        response = client.post("/api/image/extract-roi", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Should generate thumbnail even for 1x1 image
        assert len(data["thumbnail"]) > 0

        bbox = data["bounding_box"]
        assert bbox["width"] == 1
        assert bbox["height"] == 1

    def test_extract_roi_response_structure(self, client, captured_image_id):
        """Test that response has complete expected structure"""
        request_data = {
            "image_id": captured_image_id,
            "roi": {"x": 100, "y": 100, "width": 100, "height": 100},
        }
        response = client.post("/api/image/extract-roi", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Check all required fields present
        assert "success" in data
        assert "thumbnail" in data
        assert "bounding_box" in data

        # Check field types
        assert isinstance(data["success"], bool)
        assert isinstance(data["thumbnail"], str)
        assert isinstance(data["bounding_box"], dict)

        # Check bounding_box structure
        bbox = data["bounding_box"]
        assert "x" in bbox
        assert "y" in bbox
        assert "width" in bbox
        assert "height" in bbox
        assert isinstance(bbox["x"], int)
        assert isinstance(bbox["y"], int)
        assert isinstance(bbox["width"], int)
        assert isinstance(bbox["height"], int)

    def test_extract_roi_thumbnail_is_jpeg(self, client, captured_image_id):
        """Test that thumbnail is encoded as JPEG format"""
        request_data = {
            "image_id": captured_image_id,
            "roi": {"x": 100, "y": 100, "width": 100, "height": 100},
        }
        response = client.post("/api/image/extract-roi", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Decode base64 and check JPEG signature
        decoded_bytes = base64.b64decode(data["thumbnail"])

        # JPEG files start with FF D8 (SOI marker)
        assert decoded_bytes[:2] == b"\xff\xd8"
        # JPEG files typically end with FF D9 (EOI marker)
        assert decoded_bytes[-2:] == b"\xff\xd9"
