from fastapi import APIRouter, Depends, Request

from app.api.dependencies import verify_api_key
from app.guardrails.rate_limiter import limiter
from app.schemas.blog import BlogRequest, BlogResponse, JobStatus, BlogResult, ErrorResponse, JobStatusEnum
from app.services import blog_service
from app.config.settings import get_settings
from app.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)

router = APIRouter(prefix="/blog", tags=["Blog Generation"])

_ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Invalid topic — failed input validation."},
    401: {"model": ErrorResponse, "description": "Missing or invalid API key."},
    404: {"model": ErrorResponse, "description": "Job not found or expired."},
    429: {"model": ErrorResponse, "description": "Rate limit exceeded."},
    500: {"model": ErrorResponse, "description": "Pipeline or server error."},
}


@router.post(
    "/generate",
    response_model=BlogResponse,
    status_code=202,
    responses=_ERROR_RESPONSES,
    summary="Start blog generation",
    description=(
        "Accepts a topic, validates it, and starts the multi-agent blog "
        "generation pipeline in the background. Returns a `job_id` immediately. "
        "Poll `/status/{job_id}` to track progress."
    ),
)
@limiter.limit(settings.RATE_LIMIT_GENERATE)
async def generate_blog(
    request: Request,
    body: BlogRequest,
    _: None = Depends(verify_api_key),
) -> BlogResponse:
    """
    POST /api/v1/blog/generate

    `request` is required by SlowAPI's rate limiter — it reads the client IP.
    All business logic is delegated to blog_service.generate_blog().
    """
    job_id = await blog_service.generate_blog(body.topic)

    return BlogResponse(
        job_id=job_id,
        status=JobStatusEnum.QUEUED,
        message=(
            f"Blog generation started. "
            f"Poll /api/v1/blog/status/{job_id} for progress updates."
        ),
    )


@router.get(
    "/status/{job_id}",
    response_model=JobStatus,
    responses=_ERROR_RESPONSES,
    summary="Poll job status",
    description=(
        "Returns the current state of a blog generation job. "
        "Poll every 2-3 seconds until `status` is `completed` or `failed`."
    ),
)
@limiter.limit(settings.RATE_LIMIT_STATUS)
async def get_status(
    request: Request,
    job_id: str,
    _: None = Depends(verify_api_key),
) -> dict:
    """
    GET /api/v1/blog/status/{job_id}

    Returns the raw dict from job_store — FastAPI validates it against
    JobStatus automatically via response_model.
    """
    return await blog_service.get_job_status(job_id)


@router.get(
    "/result/{job_id}",
    responses=_ERROR_RESPONSES,
    summary="Fetch completed blog result",
    description=(
        "Returns the final blog post, evaluation report, and metadata. "
        "Only call this after `/status/{job_id}` returns `status: completed`."
    ),
)
@limiter.limit(settings.RATE_LIMIT_RESULT)
async def get_result(
    request: Request,
    job_id: str,
    _: None = Depends(verify_api_key),
) -> dict:
    """
    GET /api/v1/blog/result/{job_id}

    Returns the raw dict from job_store — FastAPI validates it against
    BlogResult automatically via response_model.
    """
    return await blog_service.get_job_result(job_id)
