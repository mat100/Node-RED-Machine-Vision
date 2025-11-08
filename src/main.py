"""
Machine Vision Flow - Main FastAPI Application
"""

import asyncio
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from api.exceptions import register_exception_handlers  # noqa: E402

# Import routers  # noqa: E402
from api.routers import camera, image, system, template, vision  # noqa: E402

# Import configuration and exception handlers  # noqa: E402
from config import get_settings  # noqa: E402
from core.camera_manager import CameraManager  # noqa: E402

# Import core components  # noqa: E402
from core.image_manager import ImageManager  # noqa: E402
from core.template_manager import TemplateManager  # noqa: E402

# Get configuration
settings = get_settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.system.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress watchfiles debug messages (if available in dev mode)
try:
    logging.getLogger("watchfiles").setLevel(logging.WARNING)
except Exception:
    pass  # watchfiles not installed (production mode)

# Initialize managers
image_manager = None
camera_manager = None
template_manager = None
shutdown_event = asyncio.Event()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global image_manager, camera_manager, template_manager

    # Startup
    logger.info("Starting Machine Vision Flow server...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.system.debug}")

    # Initialize managers with Pydantic config
    image_manager = ImageManager(
        max_size_mb=settings.image.max_memory_mb,
        max_images=settings.image.max_images,
        thumbnail_width=settings.image.thumbnail_width,
    )

    camera_manager = CameraManager(
        capture_timeout_ms=settings.camera.capture_timeout_ms,
    )

    template_manager = TemplateManager(storage_path=settings.template.storage_path)

    logger.info("All managers initialized successfully")

    # Store managers in app state for access by routers
    app.state.image_manager = image_manager
    app.state.camera_manager = camera_manager
    app.state.template_manager = template_manager
    app.state.config = settings.to_dict()
    app.state.debug = settings.system.debug

    yield

    # Shutdown
    logger.info("Shutting down Machine Vision Flow server...")

    # Cleanup managers - ensure camera is properly released
    try:
        if camera_manager:
            logger.info("Releasing all cameras...")
            await camera_manager.cleanup()
            # Give cameras time to fully release
            await asyncio.sleep(0.5)
        if image_manager:
            logger.info("Cleaning up image manager...")
            image_manager.cleanup()
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

    logger.info("Server shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Machine Vision Flow",
    description="Modular Machine Vision System inspired by Keyence and Cognex",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS for Node-RED
if settings.api.cors_enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Register exception handlers
register_exception_handlers(app)

# Include routers
app.include_router(camera.router, prefix="/api/camera", tags=["Camera"])
app.include_router(vision.router, prefix="/api/vision", tags=["Vision"])
app.include_router(template.router, prefix="/api/template", tags=["Template"])
app.include_router(image.router, prefix="/api/image", tags=["Image"])
app.include_router(system.router, prefix="/api/system", tags=["System"])


# Root endpoint
@app.get("/")
async def root():
    return {
        "name": "Machine Vision Flow",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "camera": "/api/camera",
            "vision": "/api/vision",
            "template": "/api/template",
            "image": "/api/image",
            "system": "/api/system",
            "docs": "/docs",
        },
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "image_manager": hasattr(app.state, "image_manager")
            and app.state.image_manager is not None,
            "camera_manager": hasattr(app.state, "camera_manager")
            and app.state.camera_manager is not None,
            "template_manager": hasattr(app.state, "template_manager")
            and app.state.template_manager is not None,
        },
    }


# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": f"Internal server error: {str(exc)}"})


# Note: Managers are set in lifespan handler above


def handle_signal(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_event.set()
    # Force exit after timeout
    import threading

    def force_exit():
        import time

        time.sleep(10)  # Give 10 seconds for graceful shutdown
        logger.warning("Forcing exit after timeout")
        sys.exit(1)

    threading.Thread(target=force_exit, daemon=True).start()


if __name__ == "__main__":
    # Write PID file for process management
    run_dir = os.getenv("RUN_DIR", os.path.join(Path(__file__).parent.parent, "var", "run"))
    pid_file = os.path.join(run_dir, "backend.pid")

    # Ensure run directory exists
    os.makedirs(run_dir, exist_ok=True)

    # Write PID
    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))
    logger.info(f"PID {os.getpid()} written to {pid_file}")

    # Register signal handlers
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    reload_excludes = (
        ["*.log", "*.pyc", "__pycache__", ".git", ".venv", "venv"]
        if settings.system.debug
        else None
    )

    # Configure server with proper shutdown handling
    server_config = uvicorn.Config(
        "main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.system.debug,
        reload_excludes=reload_excludes,
        log_level="info",
        loop="asyncio",
    )

    server = uvicorn.Server(server_config)

    # Run server with proper signal handling
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        # Clean up PID file
        try:
            if os.path.exists(pid_file):
                os.remove(pid_file)
                logger.info(f"Removed PID file: {pid_file}")
        except Exception as e:
            logger.warning(f"Failed to remove PID file: {e}")
        logger.info("Server exiting...")
