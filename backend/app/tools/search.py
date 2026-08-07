

import asyncio
from typing import Optional

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import BaseTool

from app.config.settings import get_settings
from app.utils.exceptions import ToolExecutionError
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()



def build_tavily_tool() -> TavilySearchResults:
    """
    Creates and returns a configured TavilySearchResults tool instance.

    Configuration:
        max_results=5     — fetch top 5 results per query
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

    This is a synchronous wrapper. The Researcher agent calls this function
    directly inside its (async) node function using asyncio.to_thread() to
    avoid blocking the event loop.

    Args:
        query:       The search query string.
        max_results: Override the default max results count (optional).

    Returns:
        A list of result dicts, each containing:
        {
            "title":   "Article Title",
            "url":     "https://example.com/article",
            "content": "Key excerpt from the article...",
        }

    Raises:
        ToolExecutionError: If Tavily returns an error or times out.

    Example:
        results = search_web("AI applications in healthcare 2024")
        # [
        #   {"title": "Stanford AI Study", "url": "https://...", "content": "..."},
        #   ...
        # ]
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

    # Normalize the results into a consistent structure
    structured: list[dict] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        structured.append({
            "title":   item.get("title",   "Untitled"),
            "url":     item.get("url",     ""),
            "content": item.get("content", item.get("snippet", "")),
        })

    # Respect max_results override
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
    Async version of search_web. Runs the synchronous Tavily call in a
    thread pool so it doesn't block the FastAPI event loop.

    Used by the researcher_node (which is an async function):
        results = await search_web_async("AI in healthcare")

    Args:
        query:       The search query string.
        max_results: Optional result count override.

    Returns:
        Same structure as search_web().
    """
    return await asyncio.to_thread(search_web, query, max_results)


def search_multiple_queries(queries: list[str]) -> list[dict]:
    """
    Runs multiple search queries and merges results, deduplicating by URL.

    Used when the researcher needs to cover multiple angles of a topic.
    For example, topic "AI in Healthcare" might generate queries:
        - "AI healthcare applications 2024"
        - "machine learning medical diagnosis"
        - "AI healthcare risks and challenges"

    Args:
        queries: List of search query strings.

    Returns:
        Deduplicated list of result dicts sorted by relevance (order preserved).
    """
    seen_urls: set[str] = set()
    all_results: list[dict] = []

    for query in queries:
        try:
            results = search_web(query)
            for result in results:
                url = result.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(result)
        except ToolExecutionError as e:
            logger.warning(
                "Skipping failed search query",
                extra={"query": query, "error": e.message},
            )
            continue

    logger.info(
        "Multi-query search complete",
        extra={"queries_count": len(queries), "total_results": len(all_results)},
    )
    return all_results