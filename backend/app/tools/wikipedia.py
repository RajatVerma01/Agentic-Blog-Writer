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
        top_k_results=2            — fetch top 2 matching Wikipedia articles
        doc_content_chars_max=4000 — limit to 4000 chars per article
                                     (prevents overwhelming the LLM context window)
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

    This is a synchronous function. Use search_wikipedia_async() inside
    async agent nodes to avoid blocking the event loop.

    Args:
        query: The search query (usually the blog topic or a sub-topic).

    Returns:
        A dict with Wikipedia content, or None if no results found:
        {
            "title":   "Artificial intelligence in healthcare",
            "url":     "https://en.wikipedia.org/wiki/Artificial_intelligence_in_healthcare",
            "content": "Artificial intelligence in healthcare is the use of...",
            "source":  "wikipedia"
        }

    Raises:
        ToolExecutionError: If Wikipedia API call fails.

    Example:
        result = search_wikipedia("AI in healthcare")
        if result:
            print(result["content"][:500])
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

    # Wikipedia tool returns an empty string or "No good Wikipedia Search Result was found"
    # when nothing matches
    if not raw_content or "No good Wikipedia Search Result" in raw_content:
        logger.warning(
            "Wikipedia returned no results",
            extra={"query": query},
        )
        return None

    # Build a structured result consistent with ResearchSource in state.py
    # Wikipedia URLs follow a predictable pattern based on the query
    normalized_query = query.strip().replace(" ", "_")
    wiki_url = f"https://en.wikipedia.org/wiki/{normalized_query}"

    # Extract the first line as the title (Wikipedia tool prepends the page title)
    lines = raw_content.strip().split("\n")
    title = lines[0].strip() if lines else query
    content = "\n".join(lines[1:]).strip() if len(lines) > 1 else raw_content

    result = {
        "title":   title,
        "url":     wiki_url,
        "content": content[:3000],  # Trim to 3000 chars for the state
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

    Used by the researcher_node (async function):
        wiki_result = await search_wikipedia_async("AI in healthcare")
        if wiki_result:
            sources.append(wiki_result)

    Args:
        query: The search query string.

    Returns:
        Same structure as search_wikipedia(), or None if no result found.
    """
    return await asyncio.to_thread(search_wikipedia, query)


def get_wikipedia_summary(topic: str) -> str:
    """
    Returns a plain text Wikipedia summary for a topic, or an empty string
    if nothing is found. This is a convenience wrapper that never raises —
    making it safe to call even if Wikipedia is unreachable.

    Used as a fallback in the researcher_node when Tavily has limited results.

    Args:
        topic: The blog topic string.

    Returns:
        Wikipedia content as a plain string, or "" if unavailable.
    """
    try:
        result = search_wikipedia(topic)
        return result["content"] if result else ""
    except ToolExecutionError:
        logger.warning(
            "Wikipedia unavailable, continuing without it",
            extra={"topic": topic},
        )
        return ""