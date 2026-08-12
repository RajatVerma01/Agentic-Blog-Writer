import re

from app.config.settings import get_settings
from app.utils.exceptions import InputValidationError
from app.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)

# Pre-compiled patterns — compiled ONCE at module load, not on every call.
# This is the key latency optimization: regex compilation is expensive.
_PROMPT_INJECTION_PATTERN = re.compile(
    r"(ignore (previous|all|above)|system prompt|you are now|"
    r"disregard|forget (everything|all)|act as|jailbreak)",
    flags=re.IGNORECASE,
)
_PII_PATTERN = re.compile(
    r"(\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"   # email
    r"|\b\d{10}\b"                                               # 10-digit phone
    r"|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b)",                       # formatted phone
)


def validate_topic(topic: str) -> str:
    """
    Validates the user-provided blog topic against all safety and format rules.

    Checks run IN ORDER — cheapest first to fail fast:
    1. Type check     → ensures topic is a non-empty string
    2. Length check   → TOPIC_MIN_LENGTH to TOPIC_MAX_LENGTH (from settings)
    3. Blocked words  → checks against settings.BLOCKED_KEYWORDS list
    4. Prompt inject  → detects LLM manipulation attempts
    5. PII check      → rejects topics containing emails or phone numbers

    Args:
        topic: Raw topic string from the API request body.

    Returns:
        Stripped, validated topic string.

    Raises:
        InputValidationError: On any validation failure. FastAPI's exception
        handler converts this to a 400 JSON response automatically.
    """
    if not isinstance(topic, str) or not topic.strip():
        raise InputValidationError("Topic must be a non-empty string.")

    topic = topic.strip()

    if len(topic) < settings.TOPIC_MIN_LENGTH:
        raise InputValidationError(
            f"Topic is too short (min {settings.TOPIC_MIN_LENGTH} characters).",
            detail=f"provided_length={len(topic)}",
        )

    if len(topic) > settings.TOPIC_MAX_LENGTH:
        raise InputValidationError(
            f"Topic is too long (max {settings.TOPIC_MAX_LENGTH} characters).",
            detail=f"provided_length={len(topic)}",
        )

    topic_lower = topic.lower()

    # Single pass over blocked keywords — avoids multiple .lower() calls
    matched = next(
        (kw for kw in settings.BLOCKED_KEYWORDS if kw.lower() in topic_lower),
        None,
    )
    if matched:
        raise InputValidationError(
            "Topic contains prohibited content.",
            detail=f"matched_keyword={matched}",
        )

    if _PROMPT_INJECTION_PATTERN.search(topic):
        raise InputValidationError(
            "Topic contains a prompt injection attempt.",
            detail="prompt_injection_detected",
        )

    if _PII_PATTERN.search(topic):
        raise InputValidationError(
            "Topic must not contain personal information (email, phone number).",
            detail="pii_detected",
        )

    logger.info("Topic validated successfully", extra={"topic_length": len(topic)})
    return topic