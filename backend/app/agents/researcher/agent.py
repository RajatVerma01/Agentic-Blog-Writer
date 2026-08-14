"""
app/agents/researcher/agent.py
==============================
The Researcher agent node function for the LangGraph pipeline.

What this agent does:
1. Receives the blog topic from BlogState.
2. Asks the LLM (Groq/Llama3) to generate focused search queries.
3. Runs those queries against Tavily Search + Wikipedia in parallel.
4. Returns the gathered sources as state["research_data"].

This function is registered as a node in graph.py:
    graph.add_node("researcher", researcher_node)

LangGraph contract:
- Input:  full BlogState dict
- Output: partial dict {"research_data": [...], "error": None}
  LangGraph merges this partial update into the full state.
"""

import json
import asyncio
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.agents.state import BlogState, ResearchSource
from app.agents.researcher.prompts import (
    RESEARCHER_SYSTEM_PROMPT,
    RESEARCHER_USER_PROMPT,
)
from app.agents.researcher.tools import gather_all_sources
from app.config.settings import get_settings
from app.schemas.blog import JobStatusEnum, AgentNameEnum
from app.storage.job_store import get_job_store
from app.utils.logger import get_agent_logger

settings = get_settings()


def _build_llm() -> ChatGroq:
    """
    Creates and returns a Groq LLM client configured for the researcher.
    Lower temperature (0.2) means more consistent, focused query generation.
    """
    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model_name=settings.GROQ_MODEL_NAME,
        temperature=0.2,
        max_tokens=512,
    )


# Module-level LLM singleton
llm = _build_llm()


# =============================================================================
# Query generation helper
# =============================================================================

async def _generate_search_queries(topic: str, job_id: str) -> list[str]:
    """
    Uses the LLM to generate 3-5 focused search queries for the topic.

    The LLM is instructed to return a JSON array of strings.
    We parse and validate the output, falling back to a basic default
    query list if the LLM response cannot be parsed.

    Args:
        topic:  The blog topic.
        job_id: For logging context.

    Returns:
        List of search query strings (3-5 items).
    """
    logger = get_agent_logger("researcher", job_id=job_id)
    logger.info("Generating search queries via LLM", extra={"topic": topic})

    messages = [
        SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
        HumanMessage(content=RESEARCHER_USER_PROMPT.format(topic=topic)),
    ]

    try:
        response = await asyncio.to_thread(llm.invoke, messages)
        raw_text = response.content.strip()

        # Extract JSON array from response
        # Sometimes LLM wraps it in markdown code fences: ```json [...]```
        if "```" in raw_text:
            start = raw_text.find("[")
            end = raw_text.rfind("]") + 1
            raw_text = raw_text[start:end]

        queries: list[str] = json.loads(raw_text)

        # Validate: must be a list of non-empty strings
        if not isinstance(queries, list) or len(queries) == 0:
            raise ValueError("LLM returned empty or non-list query response")

        queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]

        # Enforce limits: 3–5 queries
        queries = queries[:5]
        if len(queries) < 3:
            raise ValueError(f"LLM returned only {len(queries)} queries (need 3+)")

        logger.info(
            "Search queries generated",
            extra={"query_count": len(queries), "queries": queries},
        )
        return queries

    except (json.JSONDecodeError, ValueError) as e:
        # Graceful fallback — use basic queries derived from the topic
        logger.warning(
            "LLM query generation failed, using fallback queries",
            extra={"error": str(e), "topic": topic},
        )
        return _fallback_queries(topic)


def _fallback_queries(topic: str) -> list[str]:
    """
    Generates basic fallback search queries when the LLM fails to respond
    correctly. Ensures the pipeline never stops due to a query generation failure.

    Args:
        topic: The blog topic.

    Returns:
        3 simple queries based on the topic string.
    """
    return [
        f"{topic} overview 2024",
        f"{topic} applications and examples",
        f"{topic} challenges and future trends",
    ]


# =============================================================================
# Researcher Node — the LangGraph node function
# =============================================================================

async def researcher_node(state: BlogState) -> dict[str, Any]:
    """
    LangGraph node function for the Researcher agent.

    This is the entry point called by LangGraph when the graph execution
    reaches the "researcher" node.

    Steps:
    1. Extract topic and job_id from state.
    2. Generate search queries using the LLM.
    3. Gather sources from Tavily + Wikipedia in parallel.
    4. Return partial state update with research_data populated.

    Args:
        state: The current full BlogState dict.

    Returns:
        Partial state dict: {"research_data": [...]} on success.
        Partial state dict: {"research_data": [], "error": "..."} on failure.
        LangGraph merges this into the full state.
    """
    topic = state["topic"]
    job_id = state["metadata"]["job_id"]
    logger = get_agent_logger("researcher", job_id=job_id)

    try:
        await get_job_store().update_status(
            job_id, JobStatusEnum.RUNNING, AgentNameEnum.RESEARCHER
        )
    except Exception:
        pass

    logger.info("Researcher agent started", extra={"topic": topic})

    try:
        # Step 1: Generate search queries
        queries = await _generate_search_queries(topic, job_id)

        # Step 2: Gather sources from all tools in parallel
        sources: list[ResearchSource] = await gather_all_sources(
            topic=topic,
            queries=queries,
            job_id=job_id,
            max_sources=settings.RESEARCHER_MAX_RESULTS,
        )

        # Step 3: Validate — we need at least 1 source to continue
        if not sources:
            logger.error(
                "No research sources found",
                extra={"topic": topic, "queries": queries},
            )
            return {
                "research_data": [],
                "error": (
                    f"Researcher found no sources for topic: '{topic}'. "
                    "Try a different or more specific topic."
                ),
            }

        logger.info(
            "Researcher agent complete",
            extra={
                "sources_count": len(sources),
                "sources": [s["title"] for s in sources],
            },
        )

        return {
            "research_data": sources,
            "error": None,
        }

    except Exception as e:
        error_msg = f"Researcher agent failed unexpectedly: {str(e)}"
        logger.error(error_msg, extra={"topic": topic}, exc_info=True)
        return {
            "research_data": [],
            "error": error_msg,
        }
