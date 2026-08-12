from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config.settings import get_settings

settings = get_settings()

# Module-level singleton — created once, shared across all endpoints.
# get_remote_address extracts the client IP from the request for per-IP limiting.
limiter = Limiter(key_func=get_remote_address)