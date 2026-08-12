import json
import asyncio
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import ValidationError

from app.agents.state import BlogState
from app.agents.evaluator.prompts import (
    EVALUATOR_SYSTEM_PROMPT,
    EVALUATOR_USER_PROMPT,
    format_research_for_evaluation,
)
from app.schemas.evaluation import EvaluationResult, evaluation_to_state_dict
from app.config.settings import get_settings
from app.utils.logger import get_agent_logger

settings = get_settings()

# Module-level LLM singleton.
# temperature=0.1 — as deterministic as possible for consistent, fair scoring.
# Same topic evaluated twice should yield nearly identical scores.
_llm = ChatGroq(
    api_key=settings.GROQ_API_KEY,
    model_name=settings.GROQ_MODEL_NAME,
    temperature=0.1,
    max_tokens=2048,
)


def _parse_evaluation(raw_text: str, revision_number: int) -> EvaluationResult:
    """
    Parses the LLM's raw JSON response into a validated EvaluationResult.

    Strips markdown fences if present, then validates with Pydantic.
    All scoring rules (0-10 range, all 5 dimensions present, no duplicates)
    are enforced by EvaluationResult's validators — not repeated here.

    Args:
        raw_text:        Raw string content from the LLM response.
        revision_number: Current revision cycle (injected into the result).

    Returns:
        A fully validated EvaluationResult instance.

    Raises:
        ValueError:        If JSON cannot be parsed.
        ValidationError:   If Pydantic validation fails (re-raised as ValueError).
    """
    if "```" in raw_text:
        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1
        raw_text = raw_text[start:end]

    try:
        data = json.loads(raw_text.strip())
        data["revision_number"] = revision_number
        return EvaluationResult(**data)
    except ValidationError as e:
        raise ValueError(f"EvaluationResult validation failed: {e}") from e


async def evaluator_node(state: BlogState) -> dict[str, Any]:
    """
    LangGraph node function for the Evaluator agent.

    Scores the current draft across 5 rubric dimensions, determines
    whether it is approved, and formats improvement feedback for the
    Writer on the next revision cycle.

    Key state writes:
    - "evaluation_result": scored dict with score, approved, improvements.
    - "feedback_text":     formatted string for Writer's revision prompt.
    - "final_blog":        set to state["draft"] only when approved=True.

    Args:
        state: The current full BlogState dict.

    Returns:
        Partial state dict with evaluation_result, feedback_text,
        and optionally final_blog.
    """
    topic = state["topic"]
    job_id = state["metadata"]["job_id"]
    revision_number = state["revision_count"]
    logger = get_agent_logger("evaluator", job_id=job_id)

    logger.info(
        "Evaluator agent started",
        extra={"topic": topic, "revision_number": revision_number},
    )

    # Guard: propagate upstream error without wasting an LLM call
    if state.get("error"):
        logger.warning(
            "Skipping evaluator — error already in state",
            extra={"error": state["error"]},
        )
        return {}

    draft = state["draft"]
    if not draft:
        error_msg = "Evaluator received an empty draft — cannot evaluate."
        logger.error(error_msg)
        return {"error": error_msg}

    try:
        messages = [
            SystemMessage(content=EVALUATOR_SYSTEM_PROMPT),
            HumanMessage(
                content=EVALUATOR_USER_PROMPT.format(
                    topic=topic,
                    draft=draft,
                    research_summary=format_research_for_evaluation(state["research_data"]),
                    revision_number=revision_number,
                )
            ),
        ]

        response = await asyncio.to_thread(_llm.invoke, messages)
        result = _parse_evaluation(response.content, revision_number)

        # evaluation_to_state_dict() adds computed fields (score, approved,
        # scores_by_dimension) that Pydantic properties don't serialize.
        eval_dict = evaluation_to_state_dict(result)

        # feedback_text is the formatted revision instructions for the Writer.
        # Stored separately in state so writer/agent.py can inject it directly
        # into WRITER_REVISION_PROMPT without re-formatting.
        feedback_text = result.to_writer_prompt_context()

        logger.info(
            "Evaluator agent complete",
            extra={
                "score": result.score,
                "approved": result.approved,
                "revision_number": revision_number,
                "improvements_count": len(result.improvements),
            },
        )

        return {
            "evaluation_result": eval_dict,
            "feedback_text": feedback_text,
            "final_blog": draft if result.approved else None,
            "error": None,
        }

    except Exception as e:
        error_msg = f"Evaluator agent failed: {str(e)}"
        logger.error(error_msg, extra={"topic": topic}, exc_info=True)
        return {"error": error_msg}
