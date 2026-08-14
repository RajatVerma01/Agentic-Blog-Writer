from fastapi import APIRouter
from app.config.settings import get_settings

settings = get_settings()

router = APIRouter(tags=["Health"])


@router.get("/health", summary="Health check")
async def health_check() -> dict:
    """
    Returns the app version and environment.
    Used by Docker, load balancers, and monitoring tools to confirm
    the service is alive and accepting requests.
    """
    return {
        "status":  "ok",
        "version": settings.APP_VERSION,
        "env":     settings.APP_ENV,
    }
