import asyncio
from typing import Optional

from app.agents.graph import get_graph
from app.agents.state import create_initial_state
from app.guardrails.input_validator import validate_topic
from app.guardrails.output_validator import validate_blog_output
from app.schemas.blog import JobStatusEnum, AgentNameEnum
from app.storage.job_store import get_job_store, JobRecord
from app.utils.exceptions import JobNotFoundError, AgentExecutionError, OutputValidationError
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def generate_blog(raw_topic: str) -> str:
    """
    Entry point called by the API endpoint for POST /blog/generate.

    Validates the topic, creates a job record, starts the LangGraph
    pipeline as a background task, and returns the job_id immediately.
    The caller does NOT wait for blog generation to finish.

    Args:
        raw_topic: Raw topic string from the API request body.

    Returns:
        job_id string — the UUID the user polls for status/result.

    Raises:
        InputValidationError: If the topic fails guardrail validation (400).
    """
    # validate_topic raises InputValidationError if topic is invalid.
    # FastAPI exception handler converts it to 400 JSON automatically.
    topic = validate_topic(raw_topic)

    job_store = get_job_store()
    record = await job_store.create(topic)

    logger.info(
        "Blog generation queued",
        extra={"job_id": record.job_id, "topic": topic},
    )

    # Fire-and-forget: start pipeline in background, return job_id immediately
    asyncio.create_task(_run_pipeline(record.job_id, topic))

    return record.job_id


async def _run_pipeline(job_id: str, topic: str) -> None:
    """
    Runs the full LangGraph pipeline for one blog generation job.

    Lifecycle:
        QUEUED → RUNNING → COMPLETED (on success)
                        → FAILED    (on any error)

    This function is intentionally isolated — it catches ALL exceptions
    and stores them in the job record rather than propagating them.
    Unhandled propagation would crash the background task silently.

    Args:
        job_id: UUID of the job to run.
        topic:  Validated topic string.
    """
    job_store = get_job_store()

    await job_store.update_status(
        job_id,
        status=JobStatusEnum.RUNNING,
        current_agent=AgentNameEnum.RESEARCHER,
    )

    try:
        graph = get_graph()
        initial_state = create_initial_state(topic=topic, job_id=job_id)

        logger.info("Pipeline started", extra={"job_id": job_id})

        final_state = await graph.ainvoke(initial_state)

        # Check if any agent wrote an error into state
        if final_state.get("error"):
            raise AgentExecutionError(
                agent_name="pipeline",
                reason=final_state["error"],
            )

        final_blog = final_state.get("final_blog") or final_state.get("draft", "")

        # Output guardrail — validates word count, structure, headings
        validate_blog_output(final_blog, job_id=job_id)

        # Build a compact evaluation summary for the result response
        eval_result = final_state.get("evaluation_result", {})
        evaluation_summary = {
            "score":            eval_result.get("score"),
            "approved":         eval_result.get("approved"),
            "revision_count":   final_state.get("revision_count", 0),
            "scores_by_dimension": eval_result.get("scores_by_dimension", {}),
        }

        await job_store.complete(
            job_id=job_id,
            final_blog=final_blog,
            evaluation_summary=evaluation_summary,
        )

        logger.info(
            "Pipeline completed",
            extra={
                "job_id": job_id,
                "score": eval_result.get("score"),
                "revisions": final_state.get("revision_count", 0),
            },
        )

    except OutputValidationError as e:
        error_msg = f"Output validation failed: {e.message}"
        logger.error(error_msg, extra={"job_id": job_id})
        await job_store.fail(job_id, error_msg)

    except AgentExecutionError as e:
        logger.error(e.message, extra={"job_id": job_id, "agent": e.agent_name})
        await job_store.fail(job_id, e.message)

    except Exception as e:
        error_msg = f"Unexpected pipeline error: {str(e)}"
        logger.error(error_msg, extra={"job_id": job_id}, exc_info=True)
        await job_store.fail(job_id, error_msg)


async def get_job_status(job_id: str) -> dict:
    """
    Returns the current status of a blog generation job.

    Called by GET /blog/status/{job_id}.

    Args:
        job_id: The UUID to look up.

    Returns:
        dict from JobRecord.to_status_dict() — job_id, status, current_agent,
        error, created_at, completed_at.

    Raises:
        JobNotFoundError: If job_id doesn't exist or has expired (404).
    """
    record = await get_job_store().get(job_id)
    return record.to_status_dict()


async def get_job_result(job_id: str) -> dict:
    """
    Returns the full result of a completed blog generation job.

    Called by GET /blog/result/{job_id}.
    Should only be called after status == COMPLETED.

    Args:
        job_id: The UUID to look up.

    Returns:
        dict from JobRecord.to_result_dict() — includes final_blog and
        evaluation_summary in addition to status fields.

    Raises:
        JobNotFoundError: If job_id doesn't exist or has expired (404).
    """
    record = await get_job_store().get(job_id)
    return record.to_result_dict()


async def cleanup_expired_jobs() -> int:
    """
    Deletes all jobs older than JOB_TTL_HOURS from the job store.

    Intended to be called periodically by a background scheduler
    registered in main.py (e.g., every JOB_CLEANUP_INTERVAL_MINUTES).

    Returns:
        Number of jobs removed.
    """
    removed = await get_job_store().cleanup_expired()
    logger.info("Cleanup complete", extra={"removed": removed})
    return removed