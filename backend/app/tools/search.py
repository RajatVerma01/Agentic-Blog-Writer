import asyncio
import os
from typing import Optional

from langchain_community.tools.tavily_search import TavilySearchResults

from app.config.settings import get_settings
from app.utils.exceptions import ToolExecutionError
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Ensure TAVILY_API_KEY is in os.environ — TavilySearchResults reads it
# directly from the environment, not from our settings object.
os.environ.setdefault("TAVILY_API_KEY", settings.TAVILY_API_KEY)


def build_tavily_tool() -> TavilySearchResults:
    """
    Creates and returns a configured TavilySearchResults tool instance.

    Configuration:
        max_results=5           — fetch top 5 results per query
        search_depth="advanced" — deeper search, more relevant results
        include_answer=True     — Tavily also returns a synthesized answer
        include_raw_content=False — we only need snippets, not full HTML
    """
    return TavilySearchResults(
        max_results=settings.TAVILY_MAX_RESULTS,
        search_depth=settings.TAVILY_SEARCH_DEPTH,
        include_answer=True,
        include_raw_content=False,
        include_images=False,
        tavily_api_key=settings.TAVILY_API_KEY,
    )


# Module-level singleton — created once, reused across all calls
tavily_tool: TavilySearchResults = build_tavily_tool()


def search_web(query: str, max_results: Optional[int] = None) -> list[dict]:
    """
    Performs a Tavily web search and returns structured results.

    This is a synchronous wrapper. Agent nodes call search_web_async()
    to avoid blocking the event loop. Use this only in synchronous contexts.

    Args:
        query:       The search query string.
        max_results: Override the default max results count (optional).

    Returns:
        A list of result dicts: [{"title": str, "url": str, "content": str}]

    Raises:
        ToolExecutionError: If Tavily returns an error or times out.
    """
    logger.info("Tavily search started", extra={"query": query})

    try:
        raw_results = tavily_tool.invoke(query)
    except Exception as e:
        logger.error(
            "Tavily search failed",
            extra={"query": query, "error": str(e)},
            exc_info=True,
        )
        raise ToolExecutionError(tool_name="tavily_search", reason=str(e)) from e

    structured: list[dict] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        structured.append({
            "title":   item.get("title",   "Untitled"),
            "url":     item.get("url",     ""),
            "content": item.get("content", item.get("snippet", "")),
        })

    if max_results is not None:
        structured = structured[:max_results]

    logger.info(
        "Tavily search complete",
        extra={"query": query, "results_count": len(structured)},
    )
    return structured


async def search_web_async(
    query: str,
    max_results: Optional[int] = None,
) -> list[dict]:
    """
    Async version of search_web. Runs in a thread pool so it doesn't
    block the FastAPI event loop.

    Used by researcher/tools.py:
        results = await search_web_async("AI in healthcare")

    Args:
        query:       The search query string.
        max_results: Optional result count override.

    Returns:
        Same structure as search_web().
    """
    return await asyncio.to_thread(search_web, query, max_results)