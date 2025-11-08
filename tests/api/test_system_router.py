"""
API Integration Tests for System Router Endpoints
"""

import time


class TestSystemRouterAPI:
    """Integration tests for system router endpoints"""

    def test_health_check(self, client):
        """Test basic health check endpoint"""
        response = client.get("/api/system/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert data["status"] == "healthy"
        assert "timestamp" in data
        # Timestamp should be ISO format
        assert "T" in data["timestamp"]

    def test_get_status(self, client):
        """Test system status endpoint"""
        response = client.get("/api/system/status")

        assert response.status_code == 200
        data = response.json()

        # Check required fields
        assert "status" in data
        assert "uptime" in data
        assert "memory_usage" in data
        assert "active_cameras" in data
        assert "buffer_usage" in data

        # Check data types
        assert isinstance(data["status"], str)
        assert isinstance(data["uptime"], (int, float))
        assert isinstance(data["active_cameras"], int)

        # Check memory_usage structure
        memory = data["memory_usage"]
        assert "process_mb" in memory
        assert "system_percent" in memory
        assert "available_mb" in memory
        assert all(isinstance(v, (int, float)) for v in memory.values())

        # Check buffer_usage is a dict
        assert isinstance(data["buffer_usage"], dict)

        # Validate status is reasonable
        assert data["status"] == "healthy"
        assert data["uptime"] >= 0

    def test_get_status_with_images(self, client):
        """Test system status with images in buffer"""
        # Capture an image to populate buffer
        client.post("/api/camera/capture", json={"camera_id": "test"})

        response = client.get("/api/system/status")

        assert response.status_code == 200
        data = response.json()

        # Buffer should have stats
        buffer_stats = data["buffer_usage"]
        assert isinstance(buffer_stats, dict)
        # Should have keys like 'count', 'size_mb', etc.
        assert len(buffer_stats) > 0

    def test_get_performance_initial(self, client):
        """Test performance metrics endpoint (history tracking removed)"""
        response = client.get("/api/system/performance")

        assert response.status_code == 200
        data = response.json()

        # Check required fields
        assert "avg_processing_time" in data
        assert "total_inspections" in data
        assert "success_rate" in data
        assert "operations_per_minute" in data

        # Check data types
        assert isinstance(data["avg_processing_time"], (int, float))
        assert isinstance(data["total_inspections"], int)
        assert isinstance(data["success_rate"], (int, float))
        assert isinstance(data["operations_per_minute"], (int, float))

        # Since history was removed, performance metrics return default values
        assert data["total_inspections"] == 0
        assert data["avg_processing_time"] == 0.0
        assert data["success_rate"] == 100.0
        assert data["operations_per_minute"] == 0.0

    def test_get_performance_with_history(self, client):
        """Test performance metrics (history tracking was removed, always returns defaults)"""
        # Capture an image
        capture_response = client.post("/api/camera/capture", json={"camera_id": "test"})
        image_id = capture_response.json()["image_id"]

        # Perform a vision operation (but history is no longer tracked)
        edge_request = {
            "image_id": image_id,
            "params": {
                "method": "canny",
                "min_area": 100,
            },
        }
        client.post("/api/vision/edge-detect", json=edge_request)

        # Get performance metrics (always returns default values now)
        response = client.get("/api/system/performance")

        assert response.status_code == 200
        data = response.json()

        # History tracking was removed, so metrics are always default values
        assert data["total_inspections"] == 0
        assert data["avg_processing_time"] == 0.0
        assert data["success_rate"] == 100.0
        assert data["operations_per_minute"] == 0.0

    def test_set_debug_mode_enable(self, client):
        """Test enabling debug mode"""
        response = client.post("/api/system/debug/true")

        assert response.status_code == 200
        data = response.json()

        # Check required fields
        assert "enabled" in data
        assert "save_images" in data
        assert "show_visualizations" in data
        assert "verbose_logging" in data

        # Check debug mode is enabled
        assert data["enabled"] is True
        assert data["verbose_logging"] is True

    def test_set_debug_mode_disable(self, client):
        """Test disabling debug mode"""
        # First enable it
        client.post("/api/system/debug/true")

        # Then disable it
        response = client.post("/api/system/debug/false")

        assert response.status_code == 200
        data = response.json()

        # Check debug mode is disabled
        assert data["enabled"] is False
        assert data["verbose_logging"] is False

    def test_set_debug_mode_toggle(self, client):
        """Test toggling debug mode multiple times"""
        # Enable
        response1 = client.post("/api/system/debug/true")
        assert response1.json()["enabled"] is True

        # Disable
        response2 = client.post("/api/system/debug/false")
        assert response2.json()["enabled"] is False

        # Enable again
        response3 = client.post("/api/system/debug/true")
        assert response3.json()["enabled"] is True

    def test_get_config(self, client):
        """Test getting configuration"""
        response = client.get("/api/system/config")

        assert response.status_code == 200
        data = response.json()

        # Should be a dictionary
        assert isinstance(data, dict)
        # Config should have some content (depends on test setup)
        # At minimum should be a valid dict (may be empty or have keys)
        assert data is not None

    def test_status_uptime_increases(self, client):
        """Test that uptime increases over time"""
        # Get initial uptime
        response1 = client.get("/api/system/status")
        uptime1 = response1.json()["uptime"]

        # Wait a bit
        time.sleep(0.1)

        # Get uptime again
        response2 = client.get("/api/system/status")
        uptime2 = response2.json()["uptime"]

        # Uptime should have increased
        assert uptime2 > uptime1

    def test_status_memory_usage_reasonable(self, client):
        """Test that memory usage values are reasonable"""
        response = client.get("/api/system/status")
        memory = response.json()["memory_usage"]

        # Memory values should be positive
        assert memory["process_mb"] > 0
        assert memory["available_mb"] > 0
        # System percent should be 0-100
        assert 0 <= memory["system_percent"] <= 100

    def test_performance_operations_per_minute(self, client):
        """Test operations per minute (history tracking removed)"""
        # Capture image and run multiple operations
        capture_response = client.post("/api/camera/capture", json={"camera_id": "test"})
        image_id = capture_response.json()["image_id"]

        # Run several edge detections (but history is no longer tracked)
        for _ in range(3):
            edge_request = {"image_id": image_id, "params": {"method": "canny"}}
            client.post("/api/vision/edge-detect", json=edge_request)

        # Get performance (always returns default values now)
        response = client.get("/api/system/performance")
        data = response.json()

        # History tracking was removed, so metrics are always default values
        assert data["operations_per_minute"] == 0.0
        assert data["total_inspections"] == 0

    def test_status_active_cameras(self, client):
        """Test active cameras count"""
        response = client.get("/api/system/status")
        data = response.json()

        # Active cameras should be non-negative integer
        assert isinstance(data["active_cameras"], int)
        assert data["active_cameras"] >= 0

    def test_debug_settings_structure(self, client):
        """Test that debug settings have complete structure"""
        response = client.post("/api/system/debug/true")
        data = response.json()

        # Check all fields are present and correct type
        assert isinstance(data["enabled"], bool)
        assert isinstance(data["save_images"], bool)
        assert isinstance(data["show_visualizations"], bool)
        assert isinstance(data["verbose_logging"], bool)

    def test_multiple_status_requests(self, client):
        """Test multiple concurrent status requests"""
        responses = []
        for _ in range(5):
            response = client.get("/api/system/status")
            responses.append(response)

        # All should succeed
        assert all(r.status_code == 200 for r in responses)
        # All should have valid data
        assert all("status" in r.json() for r in responses)
        assert all(r.json()["status"] == "healthy" for r in responses)

    def test_health_check_no_dependencies(self, client):
        """Test that health check works without manager dependencies"""
        # Health check should work even if managers are not fully initialized
        # It doesn't require any dependencies
        response = client.get("/api/system/health")

        assert response.status_code == 200
        # Should return immediately without errors
        assert "status" in response.json()
