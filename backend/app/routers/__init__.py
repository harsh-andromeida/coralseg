"""
FastAPI Routers for ReefVision Backend.
"""
from app.routers.system import router as system_router
from app.routers.images import router as images_router
from app.routers.batch import router as batch_router
from app.routers.video import router as video_router

__all__ = ["system_router", "images_router", "batch_router", "video_router"]
