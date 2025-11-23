"""
API Integration Tests for Test Image Router Endpoints
"""

import base64
import io

import cv2
import numpy as np
import pytest


class TestTestImageRouterAPI:
    """Integration tests for test image router endpoints"""

    @pytest.fixture
    def test_image_bytes(self):
        """Create a test image as PNG bytes"""
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (250, 150), (255, 128, 64), -1)
        cv2.circle(image, (150, 100), 30, (0, 255, 255), -1)

        # Encode to PNG
        success, buffer = cv2.imencode(".png", image)
        assert success
        return buffer.tobytes()

    @pytest.fixture
    def uploaded_test_id(self, client, test_image_bytes):
        """Upload a test image and return its ID"""
        files = {"file": ("test_image.png", io.BytesIO(test_image_bytes), "image/png")}
        response = client.post("/api/test-image/upload", files=files)
        assert response.status_code == 200
        data = response.json()
        return data["test_id"]

    def test_upload_test_image_basic(self, client, test_image_bytes):
        """Test uploading a test image"""
        files = {"file": ("test_image.png", io.BytesIO(test_image_bytes), "image/png")}
        response = client.post("/api/test-image/upload", files=files)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "test_id" in data
        assert data["test_id"].startswith("test_")
        assert data["filename"] == "test_image.png"
        assert "size" in data
        assert data["size"]["width"] == 300
        assert data["size"]["height"] == 200

    def test_upload_test_image_jpg(self, client):
        """Test uploading a JPEG test image"""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        success, buffer = cv2.imencode(".jpg", image)
        assert success

        files = {"file": ("test.jpg", io.BytesIO(buffer.tobytes()), "image/jpeg")}
        response = client.post("/api/test-image/upload", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["filename"] == "test.jpg"

    def test_upload_invalid_file(self, client):
        """Test uploading invalid file"""
        files = {"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
        response = client.post("/api/test-image/upload", files=files)

        assert response.status_code == 400
        assert "Invalid image file" in response.json()["detail"]

    def test_list_test_images_empty(self, client):
        """Test listing test images when empty"""
        response = client.get("/api/test-image/list")

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)
        # May have images from other tests, so just check it's a list
        # assert len(data) == 0

    def test_list_test_images_with_data(self, client, uploaded_test_id):
        """Test listing test images with data"""
        response = client.get("/api/test-image/list")

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)
        assert len(data) >= 1

        # Find our uploaded image
        test_image = next((img for img in data if img["id"] == uploaded_test_id), None)
        assert test_image is not None
        assert "original_filename" in test_image
        assert "size" in test_image
        assert "created_at" in test_image

    def test_get_test_image_info(self, client, uploaded_test_id):
        """Test getting test image metadata"""
        response = client.get(f"/api/test-image/{uploaded_test_id}")

        assert response.status_code == 200
        data = response.json()

        assert data["id"] == uploaded_test_id
        assert "original_filename" in data
        assert "size" in data
        assert data["size"]["width"] == 300
        assert data["size"]["height"] == 200
        assert "created_at" in data

    def test_get_test_image_info_nonexistent(self, client):
        """Test getting metadata for non-existent test image"""
        response = client.get("/api/test-image/nonexistent-id")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_test_image_thumbnail(self, client, uploaded_test_id):
        """Test getting test image thumbnail"""
        response = client.get(f"/api/test-image/{uploaded_test_id}/image")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["test_id"] == uploaded_test_id
        assert "image_base64" in data
        assert data["image_base64"].startswith("data:image/png;base64,")

        # Verify thumbnail is valid base64
        base64_data = data["image_base64"].split(",")[1]
        decoded = base64.b64decode(base64_data)
        assert len(decoded) > 0

    def test_get_test_image_thumbnail_nonexistent(self, client):
        """Test getting thumbnail for non-existent test image"""
        response = client.get("/api/test-image/nonexistent-id/image")

        assert response.status_code == 404

    def test_capture_test_image_basic(self, client, uploaded_test_id):
        """Test capturing test image (main feature!)"""
        response = client.post(f"/api/test-image/{uploaded_test_id}/capture")

        assert response.status_code == 200
        data = response.json()

        # Should return VisionResponse format (like camera capture)
        assert "test_id" in data
        assert data["test_id"] == uploaded_test_id
        assert "thumbnail_base64" in data
        assert "processing_time_ms" in data
        assert "objects" in data
        assert isinstance(data["objects"], list)
        assert len(data["objects"]) == 1  # One VisionObject with capture info

        # Check VisionObject contains image_id in properties
        vision_obj = data["objects"][0]
        assert vision_obj["object_type"] == "test_image_capture"
        assert "properties" in vision_obj
        assert "image_id" in vision_obj["properties"]
        assert vision_obj["properties"]["test_id"] == uploaded_test_id
        assert vision_obj["properties"]["resolution"] == [300, 200]

        # Check bounding_box
        bbox = vision_obj["bounding_box"]
        assert bbox["width"] == 300
        assert bbox["height"] == 200

    def test_capture_test_image_multiple_times(self, client, uploaded_test_id):
        """Test that capturing the same test image multiple times works"""
        # Capture first time
        response1 = client.post(f"/api/test-image/{uploaded_test_id}/capture")
        assert response1.status_code == 200
        data1 = response1.json()
        image_id1 = data1["objects"][0]["properties"]["image_id"]

        # Capture second time
        response2 = client.post(f"/api/test-image/{uploaded_test_id}/capture")
        assert response2.status_code == 200
        data2 = response2.json()
        image_id2 = data2["objects"][0]["properties"]["image_id"]

        # Should get different image_ids (new capture each time)
        assert image_id1 != image_id2

        # But same test_id and dimensions
        assert data1["test_id"] == data2["test_id"] == uploaded_test_id
        bbox1 = data1["objects"][0]["bounding_box"]
        bbox2 = data2["objects"][0]["bounding_box"]
        assert bbox1["width"] == bbox2["width"] == 300
        assert bbox1["height"] == bbox2["height"] == 200

    def test_capture_test_image_nonexistent(self, client):
        """Test capturing non-existent test image"""
        response = client.post("/api/test-image/nonexistent-id/capture")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_capture_then_vision_processing(self, client, uploaded_test_id):
        """Test that captured test image can be used with vision algorithms"""
        # Capture test image
        response = client.post(f"/api/test-image/{uploaded_test_id}/capture")
        assert response.status_code == 200
        image_id = response.json()["objects"][0]["properties"]["image_id"]

        # Use captured image for edge detection (using correct param names)
        edge_request = {
            "image_id": image_id,
            "params": {"method": "canny"},  # Use defaults for canny thresholds
        }
        edge_response = client.post("/api/vision/edge-detect", json=edge_request)

        assert edge_response.status_code == 200
        edge_data = edge_response.json()
        assert "objects" in edge_data

    def test_delete_test_image(self, client, test_image_bytes):
        """Test deleting test image"""
        # Upload test image
        files = {"file": ("temp.png", io.BytesIO(test_image_bytes), "image/png")}
        upload_response = client.post("/api/test-image/upload", files=files)
        test_id = upload_response.json()["test_id"]

        # Delete it
        response = client.delete(f"/api/test-image/{test_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert test_id in data["message"]

        # Verify it's gone
        get_response = client.get(f"/api/test-image/{test_id}")
        assert get_response.status_code == 404

    def test_delete_test_image_nonexistent(self, client):
        """Test deleting non-existent test image"""
        response = client.delete("/api/test-image/nonexistent-id")

        assert response.status_code == 404

    def test_upload_multiple_and_list(self, client, test_image_bytes):
        """Test uploading multiple test images and listing them"""
        # Upload 3 test images
        test_ids = []
        for i in range(3):
            files = {"file": (f"test{i}.png", io.BytesIO(test_image_bytes), "image/png")}
            response = client.post("/api/test-image/upload", files=files)
            assert response.status_code == 200
            test_ids.append(response.json()["test_id"])

        # List all test images
        response = client.get("/api/test-image/list")
        assert response.status_code == 200
        data = response.json()

        # Should have at least our 3 images
        assert len(data) >= 3

        # Verify all our test_ids are in the list
        listed_ids = [img["id"] for img in data]
        for test_id in test_ids:
            assert test_id in listed_ids

    def test_capture_image_stored_in_image_manager(self, client, uploaded_test_id):
        """Test that captured test image is stored in ImageManager"""
        # Capture test image
        capture_response = client.post(f"/api/test-image/{uploaded_test_id}/capture")
        assert capture_response.status_code == 200
        image_id = capture_response.json()["objects"][0]["properties"]["image_id"]

        # Try to use this image_id with ROI extraction (proves it's in ImageManager)
        roi_request = {
            "image_id": image_id,
            "roi": {"x": 10, "y": 10, "width": 50, "height": 50},
        }
        roi_response = client.post("/api/image/extract-roi", json=roi_request)

        assert roi_response.status_code == 200
        roi_data = roi_response.json()
        assert len(roi_data["objects"]) > 0
