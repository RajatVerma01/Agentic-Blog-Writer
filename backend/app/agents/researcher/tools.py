"""
app/agents/researcher/tools.py
==============================
Tool wrappers specific to the Researcher agent.

Converts raw tool outputs from app/tools/ into ResearchSource TypedDicts
that BlogState["research_data"] expects. Handles partial failures
gracefully so a single tool failure never stops the whole pipeline.
"""

import asyncio
from typing import Optional

from app.agents.state import ResearchSource
from app.tools.search import search_web_async
from app.tools.wikipedia import search_wikipedia_async
from app.tools.text_utils import deduplicate_by_url
from app.utils.exceptions import ToolExecutionError
from app.utils.logger import get_agent_logger

# Max characters to store per source snippet in state.
# Keeps state size manageable and LLM context windows from overflowing.
MAX_SNIPPET_CHARS = 800


async def fetch_tavily_sources(
    queries: list[str],
    job_id: str = "",
) -> list[ResearchSource]:
    """
    Runs multiple Tavily search queries concurrently and returns results
    as a list of ResearchSource dicts ready for BlogState["research_data"].

    Runs all queries in PARALLEL using asyncio.gather() for speed.
    If one query fails, others still succeed (error is logged and skipped).

    Args:
        queries: List of search query strings (from LLM query generation).
        job_id:  Job UUID for logging context.

    Returns:
        Deduplicated list of ResearchSource dicts.
    """
    logger = get_agent_logger("researcher", job_id=job_id)
    logger.info(
        "Starting Tavily searches",
        extra={"query_count": len(queries), "queries": queries},
    )

    # Run all queries in parallel
    tasks = [search_web_async(query) for query in queries]
    results_per_query = await asyncio.gather(*tasks, return_exceptions=True)

    # Flatten all results into one list
    all_raw: list[dict] = []
    for query, result in zip(queries, results_per_query):
        if isinstance(result, Exception):
            logger.warning(
                "Tavily query failed, skipping",
                extra={"query": query, "error": str(result)},
            )
            continue
        all_raw.extend(result)

    # Deduplicate by URL using the shared utility (Issue 1 fix)
    unique_raw = deduplicate_by_url(all_raw)

    sources: list[ResearchSource] = [
        ResearchSource(
            title=item.get("title", "Untitled"),
            url=item.get("url", ""),
            snippet=item.get("content", item.get("snippet", ""))[:MAX_SNIPPET_CHARS],
            source="tavily",
        )
        for item in unique_raw
    ]

    logger.info("Tavily search complete", extra={"sources_found": len(sources)})
    return sources


async def fetch_wikipedia_source(
    topic: str,
    job_id: str = "",
) -> Optional[ResearchSource]:
    """
    Fetches the Wikipedia article for the topic and returns it as a
    ResearchSource dict, or None if no article is found.

    Args:
        topic:  The blog topic (used as the Wikipedia search query).
        job_id: Job UUID for logging.

    Returns:
        A ResearchSource dict or None.
    """
    logger = get_agent_logger("researcher", job_id=job_id)
    logger.info("Fetching Wikipedia source", extra={"topic": topic})

    try:
        result = await search_wikipedia_async(topic)
        if not result:
            logger.warning("Wikipedia returned no results", extra={"topic": topic})
            return None

        source = ResearchSource(
            title=result.get("title", f"Wikipedia: {topic}"),
            url=result.get("url", f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"),
            snippet=result.get("content", "")[:MAX_SNIPPET_CHARS],
            source="wikipedia",
        )
        logger.info("Wikipedia source fetched", extra={"title": source["title"]})
        return source

    except ToolExecutionError as e:
        logger.warning(
            "Wikipedia fetch failed, continuing without it",
            extra={"topic": topic, "error": e.message},
        )
        return None


async def gather_all_sources(
    topic: str,
    queries: list[str],
    job_id: str = "",
    max_sources: int = 7,
) -> list[ResearchSource]:
    """
    Orchestrates all research tools in parallel and returns a clean,
    deduplicated list of ResearchSource dicts limited to max_sources.

    Strategy:
    1. Run all Tavily queries concurrently.
    2. Fetch Wikipedia article concurrently.
    3. Merge: Tavily first (more specific), Wikipedia appended.
    4. Trim to max_sources.

    Args:
        topic:       The blog topic.
        queries:     LLM-generated search queries.
        job_id:      Job UUID for logging.
        max_sources: Maximum number of sources to return.

    Returns:
        List of ResearchSource dicts, max length = max_sources.
    """
    logger = get_agent_logger("researcher", job_id=job_id)
    logger.info("Gathering all research sources", extra={"topic": topic})

    tavily_task = fetch_tavily_sources(queries, job_id=job_id)
    wikipedia_task = fetch_wikipedia_source(topic, job_id=job_id)

    tavily_sources, wiki_source = await asyncio.gather(
        tavily_task, wikipedia_task, return_exceptions=False
    )

    all_sources: list[ResearchSource] = list(tavily_sources)
    if wiki_source:
        all_sources.append(wiki_source)

    trimmed = all_sources[:max_sources]

    logger.info(
        "Research complete",
        extra={
            "total_sources": len(all_sources),
            "returned_sources": len(trimmed),
        },
    )
    return trimmed
