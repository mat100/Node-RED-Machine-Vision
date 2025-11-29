"""
Template API Router - Template management
"""

import logging
from typing import List

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from dependencies import get_image_manager, get_template_manager
from exceptions import ImageNotFoundException, TemplateNotFoundException, safe_endpoint
from managers.image_manager import ImageManager
from managers.template_manager import TemplateManager
from models import Size, TemplateInfo, TemplateLearnRequest, TemplateUploadResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/list")
@safe_endpoint
async def list_templates(
    template_manager: TemplateManager = Depends(get_template_manager),
) -> List[TemplateInfo]:
    """List all templates"""
    templates = template_manager.list_templates()
    return [TemplateInfo.from_manager_dict(t) for t in templates]


@router.post("/upload")
@safe_endpoint
async def upload_template(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(None),
    template_manager: TemplateManager = Depends(get_template_manager),
) -> TemplateUploadResponse:
    """Upload new template"""
    # Read and decode image (preserve alpha channel if present)
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

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
    request: TemplateLearnRequest,
    image_manager: ImageManager = Depends(get_image_manager),
    template_manager: TemplateManager = Depends(get_template_manager),
) -> dict:
    """Learn template from image region"""
    # Get source image
    source_image = image_manager.get(request.image_id)
    if source_image is None:
        raise ImageNotFoundException(request.image_id)

    # Validate ROI
    img_height, img_width = source_image.shape[:2]
    request.roi.validate_with_constraints(image_width=img_width, image_height=img_height)

    # Learn template
    template_id = template_manager.learn_template(
        name=request.name,
        source_image=source_image,
        roi=request.roi.to_dict(),
        description=request.description,
    )

    # Get thumbnail
    thumbnail_base64 = template_manager.create_template_thumbnail(template_id)

    logger.info(f"Template learned from ROI: {template_id}")

    return {"success": True, "template_id": template_id, "thumbnail_base64": thumbnail_base64}


@router.get("/{template_id}/image")
@safe_endpoint
async def get_template_image(
    template_id: str, template_manager: TemplateManager = Depends(get_template_manager)
) -> dict:
    """Get template image"""
    thumbnail = template_manager.create_template_thumbnail(template_id, max_width=200)
    if thumbnail is None:
        raise TemplateNotFoundException(template_id)

    return {"success": True, "template_id": template_id, "image_base64": thumbnail}


@router.delete("/{template_id}")
@safe_endpoint
async def delete_template(
    template_id: str, template_manager: TemplateManager = Depends(get_template_manager)
) -> dict:
    """Delete template"""
    success = template_manager.delete_template(template_id)
    if not success:
        raise TemplateNotFoundException(template_id)

    return {"success": True, "message": f"Template {template_id} deleted"}
