from app.config.settings import get_settings
from app.utils.exceptions import OutputValidationError
from app.utils.logger import get_logger
from app.tools.text_utils import (
    count_words,
    has_h1_heading,
    count_h2_headings,
    has_conclusion,
    get_text_stats,
)

settings = get_settings()
logger = get_logger(__name__)

_MIN_H2_SECTIONS = 3


def validate_blog_output(blog: str, job_id: str = "") -> str:
    """
    Validates the final blog post produced by the Writer agent before
    it is stored in the job_store and returned to the user.

    All text analysis functions are imported from tools/text_utils.py —
    NOT re-implemented here. This file only applies the business rules.

    Checks run IN ORDER — cheapest/most likely to fail first:
    1. Empty check      → catches completely empty output
    2. Word count       → within BLOG_MIN_WORDS to BLOG_MAX_WORDS (settings)
    3. H1 heading       → blog must have a title
    4. H2 sections      → minimum 3 sections
    5. Conclusion       → must have a conclusion section

    Args:
        blog:   The full Markdown blog string from state["final_blog"].
        job_id: Used for logging context (optional).

    Returns:
        The blog string unchanged if all checks pass.

    Raises:
        OutputValidationError: On any structural quality failure.
        Maps to HTTP 500 — this is a server-side content failure.
    """
    log = logger.getChild(job_id) if job_id else logger

    if not blog or not blog.strip():
        raise OutputValidationError("Blog output is empty.")

    # get_text_stats() calls count_words, count_h2_headings, has_h1_heading,
    # has_conclusion, and extract_urls in a SINGLE pass — not called separately.
    stats = get_text_stats(blog)

    log.info("Blog output stats", extra=stats)

    if stats["word_count"] < settings.BLOG_MIN_WORDS:
        raise OutputValidationError(
            f"Blog is too short ({stats['word_count']} words). "
            f"Minimum is {settings.BLOG_MIN_WORDS} words.",
            detail=f"word_count={stats['word_count']}",
        )

    if stats["word_count"] > settings.BLOG_MAX_WORDS:
        raise OutputValidationError(
            f"Blog is too long ({stats['word_count']} words). "
            f"Maximum is {settings.BLOG_MAX_WORDS} words.",
            detail=f"word_count={stats['word_count']}",
        )

    if stats["h1_count"] == 0:
        raise OutputValidationError(
            "Blog is missing an H1 title (# Heading).",
            detail="no_h1_found",
        )

    if stats["h2_count"] < _MIN_H2_SECTIONS:
        raise OutputValidationError(
            f"Blog has only {stats['h2_count']} sections. "
            f"Minimum is {_MIN_H2_SECTIONS} H2 sections (## Heading).",
            detail=f"h2_count={stats['h2_count']}",
        )

    if not stats["has_conclusion"]:
        raise OutputValidationError(
            "Blog is missing a conclusion section.",
            detail="no_conclusion_found",
        )

    log.info(
        "Blog output validated successfully",
        extra={
            "word_count": stats["word_count"],
            "h2_count": stats["h2_count"],
            "reading_ease": stats["reading_ease_score"],
        },
    )
    return blog
