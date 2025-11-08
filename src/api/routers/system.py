"""
System API Router - Status and performance monitoring
"""

import logging
import time
from datetime import datetime

from fastapi import APIRouter, Depends, Request

from api.dependencies import get_camera_manager, get_image_manager
from api.exceptions import safe_endpoint
from schemas import DebugSettings, PerformanceMetrics, SystemStatus

logger = logging.getLogger(__name__)

router = APIRouter()

# Track start time
START_TIME = time.time()


@router.get("/status")
@safe_endpoint
async def get_status(
    image_manager=Depends(get_image_manager), camera_manager=Depends(get_camera_manager)
) -> SystemStatus:
    """Get system status (simplified without psutil dependency)"""
    # Count active cameras
    active_cameras = len([cam for cam_id, cam in camera_manager.cameras.items() if cam.connected])

    # Get buffer usage
    buffer_stats = image_manager.get_stats()

    return SystemStatus(
        status="healthy",
        uptime=time.time() - START_TIME,
        memory_usage={
            "buffer_mb": buffer_stats.get("memory_mb", 0),
            "buffer_images": buffer_stats.get("count", 0),
        },
        active_cameras=active_cameras,
        buffer_usage=buffer_stats,
    )


@router.get("/performance")
@safe_endpoint
async def get_performance() -> PerformanceMetrics:
    """Get performance metrics"""
    # Return default metrics since history tracking was removed
    return PerformanceMetrics(
        avg_processing_time=0.0,
        total_inspections=0,
        success_rate=100.0,
        operations_per_minute=0.0,
    )


@router.post("/debug/{enable}")
@safe_endpoint
async def set_debug_mode(enable: bool, request: Request) -> DebugSettings:
    """Enable or disable debug mode"""
    # Access config from app state
    config = request.app.state.config

    # Update debug settings if available
    if "debug" in config:
        config["debug"]["save_debug_images"] = enable
        config["debug"]["show_overlays"] = enable

    # Configure logging level
    log_level = logging.DEBUG if enable else logging.INFO
    logging.getLogger().setLevel(log_level)

    logger.info(f"Debug mode {'enabled' if enable else 'disabled'}")

    return DebugSettings(
        enabled=enable,
        save_images=config.get("debug", {}).get("save_debug_images", enable),
        show_visualizations=config.get("debug", {}).get("show_overlays", enable),
        verbose_logging=enable,
    )


@router.get("/config")
@safe_endpoint
async def get_config(request: Request) -> dict:
    """Get current configuration"""
    return request.app.state.config


@router.get("/health")
async def health_check() -> dict:
    """Simple health check"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}
