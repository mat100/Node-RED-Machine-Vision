"""
System-related API models.

This module contains models for system status and monitoring:
- System status information
- Performance metrics
- Debug settings
"""

from typing import Any, Dict

from pydantic import BaseModel


class SystemStatus(BaseModel):
    """System status information"""

    status: str
    uptime: float
    memory_usage: Dict[str, float]
    active_cameras: int
    buffer_usage: Dict[str, Any]


class PerformanceMetrics(BaseModel):
    """Performance metrics"""

    avg_processing_time: float
    total_inspections: int
    success_rate: float
    operations_per_minute: float


class DebugSettings(BaseModel):
    """Debug settings"""

    enabled: bool
    save_images: bool
    show_visualizations: bool
    verbose_logging: bool
