from fastapi import Header, HTTPException
from app.config.settings import get_settings

settings = get_settings()


async def verify_api_key(x_api_key: str = Header(default="")) -> None:
    """
    FastAPI dependency — verifies the X-API-Key header when the app
    is configured to require one (API_SECRET_KEY is set in .env).

    If API_SECRET_KEY is empty in settings, auth is disabled and this
    dependency is a no-op. This makes local development friction-free.

    Usage in any endpoint:
        @router.post("/generate")
        async def generate(request: ..., _: None = Depends(verify_api_key)):
            ...

    Raises:
        HTTPException 401: If key is required but missing or wrong.
    """
    if not settings.is_api_key_required:
        return

    if x_api_key != settings.API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")
