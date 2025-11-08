"""
Template API Router - Template management
"""

import logging
from typing import List

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from api.dependencies import (
    get_template_manager,  # Still needed for upload, list, delete operations
)
from api.dependencies import get_vision_service
from api.exceptions import TemplateNotFoundException, safe_endpoint
from schemas import ROI, Size, TemplateInfo, TemplateLearnRequest, TemplateUploadResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/list")
@safe_endpoint
async def list_templates(template_manager=Depends(get_template_manager)) -> List[TemplateInfo]:
    """List all templates"""
    templates = template_manager.list_templates()
    return [TemplateInfo.from_manager_dict(t) for t in templates]


@router.post("/upload")
@safe_endpoint
async def upload_template(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(None),
    template_manager=Depends(get_template_manager),
) -> TemplateUploadResponse:
    """Upload new template"""
    # Read and decode image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Upload template
    template_id = template_manager.upload_template(name, image, description)

    return TemplateUploadResponse(
        success=True,
        template_id=template_id,
        name=name,
        size=Size(width=image.shape[1], height=image.shape[0]),
    )


@router.post("/learn")
@safe_endpoint
async def learn_template(
    request: TemplateLearnRequest, vision_service=Depends(get_vision_service)
) -> dict:
    """Learn template from image region"""
    # Convert ROI from request to ROI object
    roi = ROI(x=request.roi.x, y=request.roi.y, width=request.roi.width, height=request.roi.height)

    # Service handles validation, learning, and thumbnail creation
    template_id, thumbnail_base64 = vision_service.learn_template_from_roi(
        image_id=request.image_id, roi=roi, name=request.name, description=request.description
    )

    return {"success": True, "template_id": template_id, "thumbnail_base64": thumbnail_base64}


@router.get("/{template_id}/image")
@safe_endpoint
async def get_template_image(
    template_id: str, template_manager=Depends(get_template_manager)
) -> dict:
    """Get template image"""
    thumbnail = template_manager.create_template_thumbnail(template_id, max_width=200)
    if thumbnail is None:
        raise TemplateNotFoundException(template_id)

    return {"success": True, "template_id": template_id, "image_base64": thumbnail}


@router.delete("/{template_id}")
@safe_endpoint
async def delete_template(template_id: str, template_manager=Depends(get_template_manager)) -> dict:
    """Delete template"""
    success = template_manager.delete_template(template_id)
    if not success:
        raise TemplateNotFoundException(template_id)

    return {"success": True, "message": f"Template {template_id} deleted"}
