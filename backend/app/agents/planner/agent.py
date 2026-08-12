
import json
import asyncio
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.agents.state import BlogState, BlogOutline, OutlineSection
from app.agents.planner.prompts import (
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_PROMPT,
    format_research_for_prompt,
)
from app.config.settings import get_settings
from app.utils.logger import get_agent_logger

settings = get_settings()

# Module-level LLM singleton.
# temperature=0.4 — slightly creative for title/outline generation,
# but constrained enough to stay factual and structured.
_llm = ChatGroq(
    api_key=settings.GROQ_API_KEY,
    model_name=settings.GROQ_MODEL_NAME,
    temperature=0.4,
    max_tokens=2048,
)


def _parse_outline(raw_text: str, topic: str) -> BlogOutline:
    """
    Parses the LLM's raw JSON response into a BlogOutline TypedDict.
    Falls back to a minimal default outline if parsing fails, so the
    pipeline never halts due to a malformed LLM response.

    Args:
        raw_text: Raw string from the LLM response.
        topic:    The original topic (used in fallback outline title).

    Returns:
        A valid BlogOutline TypedDict.
    """
    # Strip markdown code fences if LLM wrapped the JSON
    if "```" in raw_text:
        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1
        raw_text = raw_text[start:end]

    try:
        data = json.loads(raw_text.strip())

        sections = [
            OutlineSection(
                heading=sec.get("heading", f"Section {i+1}"),
                key_points=sec.get("key_points", []),
                word_target=int(sec.get("word_target", 200)),
            )
            for i, sec in enumerate(data.get("sections", []))
        ]

        return BlogOutline(
            title=data.get("title", topic),
            sections=sections,
            keywords=data.get("keywords", []),
            total_word_target=int(data.get("total_word_target", 1000)),
        )

    except (json.JSONDecodeError, KeyError, ValueError):
        # Fallback — minimal outline so the Writer can still proceed
        return _fallback_outline(topic)


def _fallback_outline(topic: str) -> BlogOutline:
    """
    Returns a minimal but valid BlogOutline when the LLM fails to
    produce parseable JSON. Ensures the pipeline never stops due to
    a planner parsing failure.

    Args:
        topic: The blog topic string.

    Returns:
        A simple 3-section BlogOutline.
    """
    return BlogOutline(
        title=f"A Complete Guide to {topic}",
        sections=[
            OutlineSection(
                heading="Introduction",
                key_points=[f"Overview of {topic}", "Why it matters today"],
                word_target=200,
            ),
            OutlineSection(
                heading="Key Concepts and Applications",
                key_points=["Core concepts", "Real-world applications", "Current trends"],
                word_target=500,
            ),
            OutlineSection(
                heading="Conclusion",
                key_points=["Summary of key points", "Future outlook"],
                word_target=200,
            ),
        ],
        keywords=[topic.lower(), f"{topic} guide", f"{topic} overview"],
        total_word_target=900,
    )


async def planner_node(state: BlogState) -> dict[str, Any]:
    """
    LangGraph node function for the Planner agent.

    Reads topic and research_data from state, calls the LLM to produce
    a structured blog outline, and returns it as a partial state update.

    Args:
        state: The current full BlogState dict.

    Returns:
        {"outline": BlogOutline}  on success.
        {"outline": fallback,  "error": "..."}  on failure.
    """
    topic = state["topic"]
    job_id = state["metadata"]["job_id"]
    logger = get_agent_logger("planner", job_id=job_id)

    logger.info("Planner agent started", extra={"topic": topic})

    # Guard: if researcher failed, error is already in state — propagate
    if state.get("error"):
        logger.warning(
            "Skipping planner — error already in state",
            extra={"error": state["error"]},
        )
        return {}

    try:
        research_summary = format_research_for_prompt(state["research_data"])

        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(
                content=PLANNER_USER_PROMPT.format(
                    topic=topic,
                    research_summary=research_summary,
                )
            ),
        ]

        response = await asyncio.to_thread(_llm.invoke, messages)
        outline = _parse_outline(response.content, topic)

        logger.info(
            "Planner agent complete",
            extra={
                "title": outline["title"],
                "sections": len(outline["sections"]),
                "total_word_target": outline["total_word_target"],
            },
        )

        return {"outline": outline, "error": None}

    except Exception as e:
        error_msg = f"Planner agent failed: {str(e)}"
        logger.error(error_msg, extra={"topic": topic}, exc_info=True)

        # Use fallback outline so Writer can still attempt a blog
        fallback = _fallback_outline(topic)
        logger.warning("Using fallback outline", extra={"title": fallback["title"]})

        return {"outline": fallback, "error": error_msg}
