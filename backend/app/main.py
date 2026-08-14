import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.responses import HTMLResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.v1.router import router as api_router
from app.config.settings import get_settings
from app.guardrails.rate_limiter import limiter
from app.services.blog_service import cleanup_expired_jobs
from app.utils.exceptions import (
    BlogWriterBaseException,
    blog_writer_exception_handler,
    rate_limit_exception_handler,
    unhandled_exception_handler,
)
from app.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)

_BASE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager — runs code at startup and shutdown.

    Startup:
    - Logs app start with environment and version.
    - Starts the periodic job cleanup background task.

    Shutdown:
    - Cancels the cleanup task cleanly.
    """
    logger.info(
        "App starting",
        extra={"env": settings.APP_ENV, "version": settings.APP_VERSION},
    )

    # Start periodic cleanup — runs every JOB_CLEANUP_INTERVAL_MINUTES
    cleanup_task = asyncio.create_task(_run_cleanup_loop())

    yield  # App is running — handle requests

    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    logger.info("App shutdown complete")


async def _run_cleanup_loop() -> None:
    """
    Background loop that periodically cleans up expired jobs.
    Interval is read from settings — not hardcoded.
    """
    interval_seconds = settings.JOB_CLEANUP_INTERVAL_MINUTES * 60
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            removed = await cleanup_expired_jobs()
            if removed:
                logger.info("Cleanup ran", extra={"removed": removed})
        except Exception as e:
            logger.error("Cleanup loop error", extra={"error": str(e)})


app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Rate limiter ──────────────────────────────────────────────────────────────
# Attach the limiter singleton (from guardrails/rate_limiter.py) to the app.
# SlowAPI reads app.state.limiter to apply @limiter.limit() decorators.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exception_handler)

# ── Exception handlers ────────────────────────────────────────────────────────
# All handlers already written in utils/exceptions.py — registered here.
app.add_exception_handler(BlogWriterBaseException, blog_writer_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── API routes ────────────────────────────────────────────────────────────────
app.include_router(api_router)

# ── Static files + Templates ──────────────────────────────────────────────────
_static_dir = _BASE_DIR / "static"
_templates_dir = _BASE_DIR / "templates"

app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
templates = Jinja2Templates(directory=str(_templates_dir))


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui(request: Request) -> HTMLResponse:
    """
    Serves the frontend SPA at the root URL.
    All frontend-backend communication happens via /api/v1/* endpoints.
    """
    return templates.TemplateResponse("index.html", {"request": request})