"""
Template Manager - Handles template storage and management
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class TemplateManager:
    """Manages template images for pattern matching"""

    def __init__(self, storage_path: str = "./templates"):
        """
        Initialize Template Manager

        Args:
            storage_path: Path to store template files
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Template cache
        self.templates: Dict[str, Dict] = {}
        self.template_images: Dict[str, np.ndarray] = {}
        self.template_masks: Dict[str, Optional[np.ndarray]] = {}  # Alpha channel masks

        # Thread safety
        self.lock = Lock()

        # Load existing templates
        self._load_templates()

        logger.info(f"Template Manager initialized with path: {self.storage_path}")

    def _load_templates(self):
        """Load templates from storage"""
        metadata_file = self.storage_path / "templates.json"

        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    self.templates = json.load(f)

                # Load template images
                for template_id, info in self.templates.items():
                    image_path = self.storage_path / info["filename"]
                    if image_path.exists():
                        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                        if img is not None:
                            # Split into BGR and mask if alpha channel present
                            if len(img.shape) == 3 and img.shape[2] == 4:
                                bgr = img[:, :, :3]  # First 3 channels
                                alpha = img[:, :, 3]  # Alpha channel
                                self.template_images[template_id] = bgr
                                self.template_masks[template_id] = alpha
                            else:
                                # No alpha channel
                                self.template_images[template_id] = img
                                self.template_masks[template_id] = None
                        else:
                            logger.warning(f"Failed to load template image: {image_path}")

                logger.info(f"Loaded {len(self.templates)} templates")

            except Exception as e:
                logger.error(f"Failed to load templates: {e}")
                self.templates = {}

    def _save_metadata(self):
        """Save template metadata"""
        metadata_file = self.storage_path / "templates.json"

        try:
            with open(metadata_file, "w") as f:
                json.dump(self.templates, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save template metadata: {e}")

    def upload_template(
        self, name: str, image: np.ndarray, description: Optional[str] = None
    ) -> str:
        """
        Upload a new template

        Args:
            name: Template name
            image: Template image
            description: Optional description

        Returns:
            Template ID
        """
        with self.lock:
            # Generate unique ID
            template_id = f"tmpl_{uuid.uuid4().hex[:8]}"

            # Save image file
            filename = f"{template_id}.png"
            file_path = self.storage_path / filename

            try:
                # Save image (preserve alpha channel if present)
                cv2.imwrite(str(file_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 9])

                # Extract mask if alpha channel present
                mask = None
                if len(image.shape) == 3 and image.shape[2] == 4:
                    bgr = image[:, :, :3]
                    mask = image[:, :, 3]
                    stored_image = bgr
                else:
                    stored_image = image

                # Store metadata
                self.templates[template_id] = {
                    "id": template_id,
                    "name": name,
                    "description": description,
                    "filename": filename,
                    "size": {"width": image.shape[1], "height": image.shape[0]},
                    "has_alpha": mask is not None,
                    "created_at": datetime.now().isoformat(),
                }

                # Cache image and mask
                self.template_images[template_id] = stored_image
                self.template_masks[template_id] = mask

                # Save metadata
                self._save_metadata()

                logger.info(f"Template uploaded: {template_id} - {name}")
                return template_id

            except Exception as e:
                logger.error(f"Failed to upload template: {e}")
                if file_path.exists():
                    file_path.unlink()
                raise

    def learn_template(
        self,
        name: str,
        source_image: np.ndarray,
        roi: Dict[str, int],
        description: Optional[str] = None,
    ) -> str:
        """
        Learn template from a region of an image

        Args:
            name: Template name
            source_image: Source image
            roi: Region of interest (x, y, width, height)
            description: Optional description

        Returns:
            Template ID
        """
        from image.roi import extract_roi

        # Extract ROI using centralized validation
        template = extract_roi(source_image, roi, safe_mode=False)

        if template is None:
            raise ValueError("ROI exceeds image bounds")

        # Upload as new template
        return self.upload_template(name, template, description)

    def get_template(self, template_id: str) -> Optional[np.ndarray]:
        """
        Get template image

        Args:
            template_id: Template identifier

        Returns:
            Template image or None
        """
        with self.lock:
            if template_id in self.template_images:
                return self.template_images[template_id].copy()

            # Try to load from file
            if template_id in self.templates:
                file_path = self.storage_path / self.templates[template_id]["filename"]
                if file_path.exists():
                    img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        # Split into BGR and mask if alpha channel present
                        if len(img.shape) == 3 and img.shape[2] == 4:
                            bgr = img[:, :, :3]
                            alpha = img[:, :, 3]
                            self.template_images[template_id] = bgr
                            self.template_masks[template_id] = alpha
                            return bgr.copy()
                        else:
                            self.template_images[template_id] = img
                            self.template_masks[template_id] = None
                            return img.copy()

        logger.warning(f"Template not found: {template_id}")
        return None

    def get_template_mask(self, template_id: str) -> Optional[np.ndarray]:
        """
        Get template mask (alpha channel)

        Args:
            template_id: Template identifier

        Returns:
            Mask (alpha channel) as grayscale image or None if no alpha channel
        """
        with self.lock:
            # Ensure template is loaded
            if template_id not in self.template_masks:
                # Try loading it
                self.get_template(template_id)

            if template_id in self.template_masks:
                mask = self.template_masks[template_id]
                return mask.copy() if mask is not None else None

        return None

    def get_template_info(self, template_id: str) -> Optional[Dict]:
        """Get template metadata"""
        with self.lock:
            if template_id in self.templates:
                return self.templates[template_id].copy()
        return None

    def list_templates(self) -> List[Dict]:
        """List all templates"""
        with self.lock:
            return list(self.templates.values())

    def delete_template(self, template_id: str) -> bool:
        """
        Delete a template

        Args:
            template_id: Template identifier

        Returns:
            True if deleted, False if not found
        """
        with self.lock:
            if template_id not in self.templates:
                return False

            try:
                # Delete file
                file_path = self.storage_path / self.templates[template_id]["filename"]
                if file_path.exists():
                    file_path.unlink()

                # Remove from cache
                if template_id in self.template_images:
                    del self.template_images[template_id]
                if template_id in self.template_masks:
                    del self.template_masks[template_id]

                # Remove metadata
                del self.templates[template_id]

                # Save metadata
                self._save_metadata()

                logger.info(f"Template deleted: {template_id}")
                return True

            except Exception as e:
                logger.error(f"Failed to delete template {template_id}: {e}")
                return False

    def update_template(
        self, template_id: str, name: Optional[str] = None, description: Optional[str] = None
    ) -> bool:
        """Update template metadata"""
        with self.lock:
            if template_id not in self.templates:
                return False

            if name is not None:
                self.templates[template_id]["name"] = name

            if description is not None:
                self.templates[template_id]["description"] = description

            self.templates[template_id]["updated_at"] = datetime.now().isoformat()

            # Save metadata
            self._save_metadata()

            return True

    def create_template_thumbnail(self, template_id: str, max_width: int = 100) -> Optional[str]:
        """
        Create thumbnail for template using OpenCV.

        Args:
            template_id: Template identifier
            max_width: Maximum thumbnail width

        Returns:
            Base64 encoded thumbnail with data URI prefix or None
        """
        template = self.get_template(template_id)
        if template is None:
            return None

        import base64

        # Get mask if present
        mask = self.get_template_mask(template_id)

        # Calculate size maintaining aspect ratio
        aspect = template.shape[0] / template.shape[1]
        width = min(max_width, template.shape[1])
        height = int(width * aspect)

        # Resize using Lanczos interpolation for quality
        thumbnail = cv2.resize(template, (width, height), interpolation=cv2.INTER_LANCZOS4)

        # If mask exists, resize and combine with thumbnail
        if mask is not None:
            mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LANCZOS4)
            # Combine BGR + alpha into BGRA
            thumbnail = cv2.merge(
                [thumbnail[:, :, 0], thumbnail[:, :, 1], thumbnail[:, :, 2], mask_resized]
            )

        # Encode to PNG
        success, buffer = cv2.imencode(".png", thumbnail)

        if not success:
            return None

        base64_str = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/png;base64,{base64_str}"
