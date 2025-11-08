"""
API Integration Tests for Template Endpoints
"""

import io

import cv2
import numpy as np
import pytest


class TestTemplateAPI:
    """Integration tests for template API endpoints"""

    @pytest.fixture
    def captured_image_id(self, client):
        """Capture an image for template learning"""
        response = client.post("/api/camera/capture", json={"camera_id": "test"})
        return response.json()["image_id"]

    def test_list_templates_empty(self, client):
        """Test listing templates when none exist"""
        response = client.get("/api/template/list")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_learn_template_from_roi(self, client, captured_image_id):
        """Test learning template from image ROI"""
        request_data = {
            "image_id": captured_image_id,
            "name": "Test Template",
            "description": "Template learned during integration test",
            "roi": {"x": 100, "y": 100, "width": 100, "height": 100},
        }
        response = client.post("/api/template/learn", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "template_id" in data
        assert len(data["template_id"]) > 0
        assert "thumbnail_base64" in data
        assert len(data["thumbnail_base64"]) > 0

    def test_learn_template_full_image(self, client, captured_image_id):
        """Test learning template from full image (no ROI)"""
        request_data = {
            "image_id": captured_image_id,
            "name": "Full Image Template",
            "roi": {"x": 0, "y": 0, "width": 1920, "height": 1080},
        }
        response = client.post("/api/template/learn", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_learn_template_invalid_image(self, client):
        """Test learning template from non-existent image"""
        request_data = {
            "image_id": "non-existent-id",
            "name": "Invalid Template",
            "roi": {"x": 100, "y": 100, "width": 100, "height": 100},
        }
        response = client.post("/api/template/learn", json=request_data)

        assert response.status_code == 404

    def test_learn_template_invalid_roi(self, client, captured_image_id):
        """Test learning template with invalid ROI"""
        request_data = {
            "image_id": captured_image_id,
            "name": "Invalid ROI Template",
            "roi": {"x": 5000, "y": 5000, "width": 100, "height": 100},  # Out of bounds
        }
        response = client.post("/api/template/learn", json=request_data)

        # Should return error for invalid ROI
        assert response.status_code in [400, 422]

    def test_upload_template(self, client):
        """Test uploading template image file"""
        # Create a simple test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[25:75, 25:75] = 255  # White square in center

        # Encode as PNG
        success, encoded = cv2.imencode(".png", test_image)
        assert success

        # Create file-like object
        files = {"file": ("test_template.png", io.BytesIO(encoded.tobytes()), "image/png")}
        data = {"name": "Uploaded Template", "description": "Template uploaded via API"}

        response = client.post("/api/template/upload", files=files, data=data)

        assert response.status_code == 200
        result = response.json()

        assert result["success"] is True
        assert "template_id" in result
        assert result["name"] == "Uploaded Template"
        assert "size" in result
        assert result["size"]["width"] == 100
        assert result["size"]["height"] == 100

    def test_list_templates_after_creation(self, client, captured_image_id):
        """Test listing templates after creating some"""
        # Create a few templates
        for i in range(3):
            request_data = {
                "image_id": captured_image_id,
                "name": f"Template {i+1}",
                "roi": {"x": 100 + i * 50, "y": 100, "width": 100, "height": 100},
            }
            client.post("/api/template/learn", json=request_data)

        # List templates
        response = client.get("/api/template/list")

        assert response.status_code == 200
        data = response.json()

        assert len(data) >= 3
        # Check structure of template info
        for template in data:
            assert "id" in template
            assert "name" in template
            assert "size" in template
            assert "created_at" in template

    def test_get_template_image(self, client, captured_image_id):
        """Test getting template image"""
        # First create a template
        request_data = {
            "image_id": captured_image_id,
            "name": "Template for Image Test",
            "roi": {"x": 100, "y": 100, "width": 100, "height": 100},
        }
        create_response = client.post("/api/template/learn", json=request_data)
        template_id = create_response.json()["template_id"]

        # Get template image
        response = client.get(f"/api/template/{template_id}/image")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["template_id"] == template_id
        assert "image_base64" in data
        assert len(data["image_base64"]) > 0

    def test_get_template_image_not_found(self, client):
        """Test getting image for non-existent template"""
        response = client.get("/api/template/non-existent-id/image")

        assert response.status_code == 404

    def test_delete_template(self, client, captured_image_id):
        """Test deleting a template"""
        # Create template
        request_data = {
            "image_id": captured_image_id,
            "name": "Template to Delete",
            "roi": {"x": 100, "y": 100, "width": 100, "height": 100},
        }
        create_response = client.post("/api/template/learn", json=request_data)
        template_id = create_response.json()["template_id"]

        # Delete template
        response = client.delete(f"/api/template/{template_id}")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "message" in data
        assert template_id in data["message"]

        # Verify deletion - try to get the template
        get_response = client.get(f"/api/template/{template_id}/image")
        assert get_response.status_code == 404

    def test_delete_template_not_found(self, client):
        """Test deleting non-existent template"""
        response = client.delete("/api/template/non-existent-id")

        assert response.status_code == 404

    def test_template_workflow(self, client):
        """Test complete template workflow: capture -> learn -> match -> delete"""
        # 1. Capture image
        capture_response = client.post("/api/camera/capture", json={"camera_id": "test"})
        image_id = capture_response.json()["image_id"]

        # 2. Learn template
        learn_data = {
            "image_id": image_id,
            "name": "Workflow Template",
            "roi": {"x": 100, "y": 100, "width": 100, "height": 100},
        }
        learn_response = client.post("/api/template/learn", json=learn_data)
        template_id = learn_response.json()["template_id"]

        assert learn_response.status_code == 200

        # 3. Match template
        match_data = {
            "image_id": image_id,
            "params": {"template_id": template_id, "threshold": 0.5},
        }
        match_response = client.post("/api/vision/template-match", json=match_data)

        assert match_response.status_code == 200
        data = match_response.json()
        assert "objects" in data
        assert "thumbnail_base64" in data

        # 4. Delete template
        delete_response = client.delete(f"/api/template/{template_id}")

        assert delete_response.status_code == 200

    def test_upload_invalid_image_format(self, client):
        """Test uploading invalid file format"""
        # Create invalid file content
        files = {"file": ("invalid.txt", io.BytesIO(b"not an image"), "text/plain")}
        data = {"name": "Invalid Upload"}

        response = client.post("/api/template/upload", files=files, data=data)

        # Should return error
        assert response.status_code in [400, 422]
