"""
ReefVision FastAPI Engine - Autonomous multi-model coral reef instance segmentation,
taxonomy classification, and bleaching assessment.
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure backend root is on sys.path so 'core' can be imported cleanly
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from core.models import check_models_download_status
from app import state
from app.routers import system_router, images_router, batch_router, video_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager:
    Initializes processor singletons on startup, and cleans up resources on shutdown.
    """
    # Startup: ensure processors are initialized
    state.init_processors()
    yield
    # Shutdown: cancel active background tasks and jobs
    if state.batch_processor:
        state.batch_processor.cancel_batch()
    if state.video_processor:
        active_job = state.video_processor.get_active_job()
        if active_job and active_job.get("state") == "running":
            state.video_processor.cancel_job(active_job["job_id"])


app = FastAPI(
    title="Andromeida Reef Vision Studio API",
    version="2.0.0",
    description="FastAPI Backend for Coral Reef Segmentation, BioCLIP Taxonomy, and Bleaching Detection",
    lifespan=lifespan,
)

# Enable CORS for Tauri desktop and Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount modular API routers
app.include_router(system_router)
app.include_router(images_router)
app.include_router(batch_router)
app.include_router(video_router)

# Mount frontend static distribution or fallback route
DIST_DIR = os.path.join(BACKEND_DIR.parent, "dist")
if os.path.isdir(DIST_DIR):
    app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="static_frontend")
else:
    @app.get("/")
    def index_fallback():
        return {
            "status": "online",
            "message": "ReefVision Studio API Engine is running. Run 'npm run build' to bundle the web UI.",
            "docs": "/docs",
            "api_health": "/api/health",
        }
