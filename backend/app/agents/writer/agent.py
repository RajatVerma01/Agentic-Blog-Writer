import asyncio
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.agents.state import BlogState
from app.agents.writer.prompts import (
    WRITER_SYSTEM_PROMPT,
    WRITER_USER_PROMPT,
    WRITER_REVISION_PROMPT,
    format_sections_for_prompt,
)
from app.agents.planner.prompts import format_research_for_prompt
from app.config.settings import get_settings
from app.schemas.blog import JobStatusEnum, AgentNameEnum
from app.storage.job_store import get_job_store
from app.utils.logger import get_agent_logger

settings = get_settings()

# Module-level LLM singleton.
# temperature=0.6 — enough creativity for engaging prose while staying factual.
_llm = ChatGroq(
    api_key=settings.GROQ_API_KEY,
    model_name=settings.GROQ_MODEL_NAME,
    temperature=0.6,
    max_tokens=settings.GROQ_MAX_TOKENS,
)


def _is_first_draft(state: BlogState) -> bool:
    """Returns True if no draft exists yet (first run).
    Checks state['draft'] content, NOT revision_count — because
    revision_count stays 0 after the first draft, so checking it
    would always return True even on revision cycles.
    """
    return not state.get("draft", "").strip()


def _build_messages(state: BlogState) -> list:
    """
    Builds the LLM message list for either a first draft or a revision.

    First draft  → uses WRITER_USER_PROMPT with outline + research.
    Revision     → uses WRITER_REVISION_PROMPT with draft + evaluation feedback.

    Centralising this logic here keeps writer_node clean and avoids
    duplicating the if/else branch in multiple places.

    Args:
        state: Full BlogState dict.

    Returns:
        List of [SystemMessage, HumanMessage] for the LLM call.
    """
    outline = state["outline"]
    topic = state["topic"]

    if _is_first_draft(state):
        user_content = WRITER_USER_PROMPT.format(
            topic=topic,
            title=outline["title"],
            total_word_target=outline["total_word_target"],
            sections_text=format_sections_for_prompt(outline["sections"]),
            research_summary=format_research_for_prompt(state["research_data"]),
        )
    else:
        # Revision — inject evaluation feedback so the Writer knows what to fix.
        # evaluation_result["feedback_text"] is set by evaluation_to_state_dict.
        eval_result = state["evaluation_result"]
        evaluation_context = eval_result.get("feedback_text", eval_result.get("feedback", ""))

        user_content = WRITER_REVISION_PROMPT.format(
            topic=topic,
            draft=state["draft"],
            evaluation_context=evaluation_context,
        )

    return [
        SystemMessage(content=WRITER_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]


async def writer_node(state: BlogState) -> dict[str, Any]:
    """
    LangGraph node function for the Writer agent.

    On first call:  writes the initial draft from outline + research.
    On revision:    rewrites the draft guided by evaluation feedback.

    The `revision_count` is incremented here so graph.py can enforce
    the MAX_REVISION_CYCLES safety limit.

    Args:
        state: The current full BlogState dict.

    Returns:
        {"draft": str, "revision_count": int}  on success.
        {"draft": "", "error": str}            on failure.
    """
    topic = state["topic"]
    job_id = state["metadata"]["job_id"]
    revision = state["revision_count"]
    logger = get_agent_logger("writer", job_id=job_id)

    try:
        await get_job_store().update_status(
            job_id, JobStatusEnum.RUNNING, AgentNameEnum.WRITER
        )
    except Exception:
        pass

    mode = "first draft" if _is_first_draft(state) else f"revision {revision}"
    logger.info("Writer agent started", extra={"topic": topic, "mode": mode})

    # Guard: propagate upstream error without wasting an LLM call
    if state.get("error"):
        logger.warning(
            "Skipping writer — error already in state",
            extra={"error": state["error"]},
        )
        return {"error": state["error"]}

    try:
        messages = _build_messages(state)
        response = await asyncio.to_thread(_llm.invoke, messages)
        draft = response.content.strip()

        if not draft:
            raise ValueError("LLM returned an empty draft.")

        logger.info(
            "Writer agent complete",
            extra={
                "mode": mode,
                "draft_length": len(draft),
                "revision_count": revision + 1 if not _is_first_draft(state) else 0,
            },
        )

        return {
            "draft": draft,
            "revision_count": revision if _is_first_draft(state) else revision + 1,
            "error": None,
        }

    except Exception as e:
        error_msg = f"Writer agent failed ({mode}): {str(e)}"
        logger.error(error_msg, extra={"topic": topic}, exc_info=True)
        return {"draft": "", "error": error_msg}
