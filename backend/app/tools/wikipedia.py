import asyncio
from typing import Optional

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from app.utils.exceptions import ToolExecutionError
from app.utils.logger import get_logger

logger = get_logger(__name__)


def build_wikipedia_tool() -> WikipediaQueryRun:
    """
    Creates and returns a configured WikipediaQueryRun tool instance.

    Configuration:
        top_k_results=2            — fetch top 2 matching articles
        doc_content_chars_max=4000 — limit chars per article
        lang="en"                  — English Wikipedia
    """
    wrapper = WikipediaAPIWrapper(
        top_k_results=2,
        doc_content_chars_max=4000,
        lang="en",
    )
    return WikipediaQueryRun(api_wrapper=wrapper)


# Module-level singleton — created once, reused across all calls
wikipedia_tool: WikipediaQueryRun = build_wikipedia_tool()


def search_wikipedia(query: str) -> Optional[dict]:
    """
    Searches Wikipedia for the given query and returns the result.

    Use search_wikipedia_async() inside async agent nodes to avoid
    blocking the event loop.

    Args:
        query: The search query (usually the blog topic or a sub-topic).

    Returns:
        A dict: {"title": str, "url": str, "content": str, "source": "wikipedia"}
        or None if no results found.

    Raises:
        ToolExecutionError: If Wikipedia API call fails.
    """
    logger.info("Wikipedia search started", extra={"query": query})

    try:
        raw_content: str = wikipedia_tool.run(query)
    except Exception as e:
        logger.error(
            "Wikipedia search failed",
            extra={"query": query, "error": str(e)},
            exc_info=True,
        )
        raise ToolExecutionError(tool_name="wikipedia", reason=str(e)) from e

    if not raw_content or "No good Wikipedia Search Result" in raw_content:
        logger.warning("Wikipedia returned no results", extra={"query": query})
        return None

    normalized_query = query.strip().replace(" ", "_")
    wiki_url = f"https://en.wikipedia.org/wiki/{normalized_query}"

    lines = raw_content.strip().split("\n")
    title = lines[0].strip() if lines else query
    content = "\n".join(lines[1:]).strip() if len(lines) > 1 else raw_content

    result = {
        "title":   title,
        "url":     wiki_url,
        "content": content[:3000],
        "source":  "wikipedia",
    }

    logger.info(
        "Wikipedia search complete",
        extra={"query": query, "title": title, "content_length": len(content)},
    )
    return result


async def search_wikipedia_async(query: str) -> Optional[dict]:
    """
    Async version of search_wikipedia. Runs in a thread pool to avoid
    blocking the FastAPI event loop.

    Used by researcher/tools.py:
        wiki_result = await search_wikipedia_async("AI in healthcare")

    Args:
        query: The search query string.

    Returns:
        Same structure as search_wikipedia(), or None if no result found.
    """
    return await asyncio.to_thread(search_wikipedia, query)