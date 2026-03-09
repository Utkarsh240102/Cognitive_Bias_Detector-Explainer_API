"""
FastAPI application entry point.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import API_TITLE, API_DESCRIPTION, API_VERSION, LLM_ENABLED
from app.logger import get_logger
from app.services.inference import load_model
from app.services.llm_explainer import load_llm

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    logger.info("Starting up Cognitive Bias Detector API...")
    load_model()
    if LLM_ENABLED:
        load_llm()
    yield
    logger.info("Shutting down Cognitive Bias Detector API...")


app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
)

# ── CORS Middleware ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Exception Handlers ──────────────────────────────────────────
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle preprocessing validation errors (text too short/long)."""
    logger.warning("Validation error: %s", exc)
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    """Handle model-not-loaded or other runtime errors."""
    logger.error("Runtime error: %s", exc)
    return JSONResponse(status_code=503, content={"detail": str(exc)})


# ── Routes ──────────────────────────────────────────────────────
app.include_router(router)

# ── Frontend Static Files ───────────────────────────────────────
if STATIC_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="assets")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(str(STATIC_DIR / "index.html"))

