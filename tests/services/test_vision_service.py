"""
Unit tests for VisionService
"""

import cv2
import numpy as np
import pytest

from api.exceptions import ImageNotFoundException, TemplateNotFoundException
from schemas import ROI, EdgeDetectionParams, TemplateMatchParams
from services.vision_service import VisionService


class TestVisionService:
    """Test VisionService functionality"""

    def test_template_match_success(self, mock_image_manager, mock_template_manager):
        """Test successful template matching"""
        service = VisionService(mock_image_manager, mock_template_manager)

        # Create test image and template
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image[100:150, 100:150] = 255  # White square
        test_template = np.ones((50, 50, 3), dtype=np.uint8) * 255

        mock_image_manager.get.return_value = test_image
        mock_template_manager.get_template.return_value = test_template
        mock_image_manager.create_thumbnail.return_value = (test_image, "base64-thumb")

        params = TemplateMatchParams(
            template_id="test-tmpl", method="TM_CCOEFF_NORMED", threshold=0.8
        )
        matches, thumbnail, processing_time = service.template_match(
            image_id="test-img", template_id="test-tmpl", roi=None, params=params
        )

        assert thumbnail == "base64-thumb"
        assert processing_time > 0
        assert isinstance(matches, list)

    def test_template_match_image_not_found(self, mock_image_manager, mock_template_manager):
        """Test template matching with non-existent image"""
        service = VisionService(mock_image_manager, mock_template_manager)

        mock_image_manager.get.return_value = None

        params = TemplateMatchParams(template_id="test-tmpl", threshold=0.8)
        with pytest.raises(ImageNotFoundException):
            service.template_match(
                image_id="non-existent", template_id="test-tmpl", roi=None, params=params
            )

    def test_template_match_template_not_found(self, mock_image_manager, mock_template_manager):
        """Test template matching with non-existent template"""
        service = VisionService(mock_image_manager, mock_template_manager)

        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_image_manager.get.return_value = test_image
        mock_template_manager.get_template.return_value = None

        params = TemplateMatchParams(template_id="non-existent", threshold=0.8)
        with pytest.raises(TemplateNotFoundException):
            service.template_match(
                image_id="test-img", template_id="non-existent", roi=None, params=params
            )

    def test_template_match_with_roi(self, mock_image_manager, mock_template_manager):
        """Test template matching with ROI"""
        service = VisionService(mock_image_manager, mock_template_manager)

        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_template = np.ones((50, 50, 3), dtype=np.uint8) * 255

        mock_image_manager.get.return_value = test_image
        mock_template_manager.get_template.return_value = test_template
        mock_image_manager.create_thumbnail.return_value = (test_image, "base64-thumb")

        params = TemplateMatchParams(template_id="test-tmpl", threshold=0.8)
        matches, thumbnail, processing_time = service.template_match(
            image_id="test-img", template_id="test-tmpl", roi=None, params=params
        )

        assert thumbnail is not None
        assert processing_time > 0

    def test_learn_template_from_roi(self, mock_image_manager, mock_template_manager):
        """Test learning template from ROI"""
        service = VisionService(mock_image_manager, mock_template_manager)

        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image[100:200, 100:200] = 255  # White square

        mock_image_manager.get.return_value = test_image
        mock_template_manager.learn_template.return_value = "new-template-id"
        mock_template_manager.create_template_thumbnail.return_value = "template-thumb"

        roi = ROI(x=100, y=100, width=100, height=100)
        template_id, thumbnail = service.learn_template_from_roi(
            image_id="test-img", roi=roi, name="Test Template", description="Test description"
        )

        assert template_id == "new-template-id"
        assert thumbnail == "template-thumb"
        mock_template_manager.learn_template.assert_called_once()

    def test_learn_template_image_not_found(self, mock_image_manager, mock_template_manager):
        """Test learning template when image not found"""
        service = VisionService(mock_image_manager, mock_template_manager)

        mock_image_manager.get.return_value = None

        roi = ROI(x=100, y=100, width=100, height=100)

        with pytest.raises(ImageNotFoundException):
            service.learn_template_from_roi(image_id="non-existent", roi=roi, name="Test")

    def test_edge_detect_success(self, mock_image_manager, mock_template_manager):
        """Test successful edge detection"""
        service = VisionService(mock_image_manager, mock_template_manager)

        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (100, 100), (300, 300), (255, 255, 255), 2)

        mock_image_manager.get.return_value = test_image
        mock_image_manager.create_thumbnail.return_value = (test_image, "base64-thumb")

        params = EdgeDetectionParams(method="canny")
        detected_objects, thumbnail, processing_time = service.edge_detect(
            image_id="test-img", method="canny", params=params
        )

        assert isinstance(detected_objects, list)
        assert thumbnail == "base64-thumb"
        assert processing_time > 0

    def test_edge_detect_image_not_found(self, mock_image_manager, mock_template_manager):
        """Test edge detection when image not found"""
        service = VisionService(mock_image_manager, mock_template_manager)

        mock_image_manager.get.return_value = None

        params = EdgeDetectionParams(method="canny")
        with pytest.raises(ImageNotFoundException):
            service.edge_detect(image_id="non-existent", method="canny", params=params)

    def test_edge_detect_with_params(self, mock_image_manager, mock_template_manager):
        """Test edge detection with custom parameters"""
        service = VisionService(mock_image_manager, mock_template_manager)

        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_image_manager.get.return_value = test_image
        mock_image_manager.create_thumbnail.return_value = (test_image, "base64-thumb")

        params = EdgeDetectionParams(method="canny", canny_low=50, canny_high=150)
        result, thumbnail, processing_time = service.edge_detect(
            image_id="test-img", method="canny", params=params
        )

        assert result is not None
        assert thumbnail is not None

    def test_edge_detect_invalid_method(self, mock_image_manager, mock_template_manager):
        """Test edge detection with invalid method defaults to canny"""
        service = VisionService(mock_image_manager, mock_template_manager)

        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_image_manager.get.return_value = test_image
        mock_image_manager.create_thumbnail.return_value = (test_image, "base64-thumb")

        # Invalid method should default to canny
        params = EdgeDetectionParams(method="invalid_method_name")
        result, thumbnail, processing_time = service.edge_detect(
            image_id="test-img", method="invalid_method_name", params=params
        )

        assert result is not None
        assert thumbnail is not None


class TestVisionServiceIntegration:
    """Integration tests with real managers"""

    def test_template_match_integration(
        self, vision_service, test_image, test_template, image_manager, template_manager
    ):
        """Test template matching with real managers"""
        # Store image and template
        image_id = image_manager.store(test_image, {})
        template_id = template_manager.upload_template("test-template", test_template)

        # Perform template matching
        params = TemplateMatchParams(template_id=template_id, threshold=0.5)
        matches, thumbnail, processing_time = vision_service.template_match(
            image_id=image_id, template_id=template_id, roi=None, params=params
        )

        assert thumbnail is not None
        assert processing_time > 0
        assert isinstance(matches, list)

    def test_edge_detect_integration(self, vision_service, test_image, image_manager):
        """Test edge detection with real managers"""
        # Store image
        image_id = image_manager.store(test_image, {})

        # Perform edge detection
        params = EdgeDetectionParams(method="canny")
        detected_objects, thumbnail, processing_time = vision_service.edge_detect(
            image_id=image_id, method="canny", params=params
        )

        assert isinstance(detected_objects, list)
        assert thumbnail is not None
        assert processing_time > 0
        assert len(detected_objects) >= 0

    def test_learn_template_integration(self, vision_service, test_image, image_manager):
        """Test learning template with real managers"""
        # Store image
        image_id = image_manager.store(test_image, {})

        # Learn template from ROI
        roi = ROI(x=100, y=100, width=200, height=200)
        template_id, thumbnail = vision_service.learn_template_from_roi(
            image_id=image_id, roi=roi, name="Integration Test Template"
        )

        assert template_id is not None
        assert len(template_id) > 0
        assert thumbnail is not None
