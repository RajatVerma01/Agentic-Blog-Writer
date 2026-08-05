from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from app.schemas.evaluation import EvaluationResult




class JobStatusEnum(str, Enum):
    """
    The four possible lifecycle states of a blog generation job.
    Using str Enum means values serialize as plain strings in JSON.

    Lifecycle:
        queued → running → completed
                        → failed
    """
    QUEUED    = "queued"     # Job created, pipeline not started yet
    RUNNING   = "running"    # Agents are actively executing
    COMPLETED = "completed"  # Final blog approved and ready
    FAILED    = "failed"     # An agent or tool raised an unhandled error


class AgentNameEnum(str, Enum):
    """
    Names of the four agents — used in JobStatus.current_agent so the
    frontend can show which step is currently running.
    """
    RESEARCHER = "researcher"
    PLANNER    = "planner"
    WRITER     = "writer"
    EVALUATOR  = "evaluator"
    DONE       = "done"       # pipeline finished
    NONE       = "none"       # not started yet



class BlogRequest(BaseModel):
    """
    The only input the user provides: a blog topic.

    FastAPI automatically validates this before your endpoint code runs.
    If topic is missing or fails validation → 422 Unprocessable Entity.

    Example request body:
        { "topic": "The Future of Artificial Intelligence in Healthcare" }
    """

    topic: str = Field(
        min_length=10,
        max_length=200,
        description=(
            "The blog topic to write about. "
            "Must be descriptive (min 10 chars, max 200 chars)."
        ),
        examples=[
            "The Future of Artificial Intelligence in Healthcare",
            "How Blockchain is Transforming Supply Chain Management",
        ],
    )

    @field_validator("topic", mode="before")
    @classmethod
    def strip_and_normalize(cls, v: str) -> str:
        """
        Strip leading/trailing whitespace and collapse internal whitespace.
        Prevents topics like "  AI   in   Healthcare  " from passing through.
        """
        if not isinstance(v, str):
            raise ValueError("Topic must be a string.")
        # Strip outer whitespace
        v = v.strip()
        # Collapse multiple internal spaces
        import re
        v = re.sub(r"\s+", " ", v)
        return v

    @field_validator("topic")
    @classmethod
    def topic_must_have_words(cls, v: str) -> str:
        """Ensure the topic has at least 2 words (not just a single character repeated)."""
        words = v.split()
        if len(words) < 2:
            raise ValueError(
                "Topic must contain at least 2 words. "
                "Example: 'AI in Healthcare' not just 'AI'."
            )
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "topic": "The Future of Artificial Intelligence in Healthcare"
            }
        }
    }




class BlogResponse(BaseModel):
    """
    Returned immediately (< 100ms) after POST /blog/generate.

    The blog generation pipeline runs in the background. The user must
    use job_id to poll /status and then fetch /result.

    Example response:
        {
            "job_id":  "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "status":  "queued",
            "message": "Blog generation started. Poll /api/v1/blog/status/{job_id} for updates."
        }
    """

    job_id: str = Field(
        description="Unique identifier for this blog generation job. Use this to poll for status.",
        examples=["a1b2c3d4-e5f6-7890-abcd-ef1234567890"],
    )

    status: JobStatusEnum = Field(
        default=JobStatusEnum.QUEUED,
        description="Initial job status. Always 'queued' on creation.",
    )

    message: str = Field(
        description="Human-readable description of what to do next.",
        examples=["Blog generation started. Poll /api/v1/blog/status/{job_id} for updates."],
    )


class JobStatus(BaseModel):
    """
    Returned by GET /blog/status/{job_id} — the frontend polls this
    every 2 seconds to show a live progress indicator.

    The current_agent field powers the animated progress bar:
        researcher → planner → writer → evaluator → (writer → evaluator)* → done

    Example response (mid-generation):
        {
            "job_id":        "a1b2c3d4-...",
            "status":        "running",
            "current_agent": "writer",
            "revision_count": 1,
            "created_at":    "2024-01-15T10:30:00Z",
            "updated_at":    "2024-01-15T10:30:45Z"
        }
    """

    job_id: str = Field(
        description="Unique job identifier.",
    )

    status: JobStatusEnum = Field(
        description="Current lifecycle state of the job.",
    )

    current_agent: AgentNameEnum = Field(
        default=AgentNameEnum.NONE,
        description="Which agent is currently executing. Used to power the progress bar.",
    )

    revision_count: int = Field(
        default=0,
        ge=0,
        le=3,
        description=(
            "How many writer ↔ evaluator revision cycles have completed. "
            "Maximum is 3 (configured via MAX_REVISION_CYCLES)."
        ),
    )

    created_at: datetime = Field(
        description="ISO 8601 timestamp when the job was created.",
    )

    updated_at: datetime = Field(
        description="ISO 8601 timestamp of the last status change.",
    )

    error: Optional[str] = Field(
        default=None,
        description=(
            "Error message if status is 'failed'. "
            "None for all other statuses."
        ),
    )

    @property
    def elapsed_seconds(self) -> float:
        """How long this job has been running (in seconds)."""
        return (self.updated_at - self.created_at).total_seconds()

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_id":         "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "status":         "running",
                "current_agent":  "writer",
                "revision_count": 1,
                "created_at":     "2024-01-15T10:30:00Z",
                "updated_at":     "2024-01-15T10:30:45Z",
                "error":          None,
            }
        }
    }


class BlogMetadata(BaseModel):
    """
    Metadata about the blog generation process — attached to BlogResult.
    Useful for debugging, analytics, and showing the user how long it took.

    Example:
        {
            "job_id":          "a1b2c3d4-...",
            "started_at":      "2024-01-15T10:30:00Z",
            "completed_at":    "2024-01-15T10:31:05Z",
            "elapsed_seconds": 65.2,
            "total_revisions": 1,
            "sources_used":    5,
            "word_count":      1247
        }
    """

    job_id: str
    started_at: datetime
    completed_at: datetime
    elapsed_seconds: float = Field(ge=0.0)
    total_revisions: int   = Field(ge=0, le=3)
    sources_used: int      = Field(ge=0, description="Number of research sources used.")
    word_count: int        = Field(ge=0, description="Final word count of the approved blog.")


class BlogResult(BaseModel):
    """
    The final response returned by GET /blog/result/{job_id} when the
    job status is 'completed'.

    Contains:
    - The full blog post in Markdown format (render on frontend)
    - The evaluation report (score, improvements, dimension breakdown)
    - Metadata (timing, revision count, word count)

    Example response:
        {
            "job_id":     "a1b2c3d4-...",
            "topic":      "AI in Healthcare",
            "final_blog": "# AI in Healthcare\\n\\nArtificial intelligence...",
            "evaluation": {
                "score":    8.4,
                "approved": true,
                "improvements": [],
                ...
            },
            "metadata": {
                "elapsed_seconds": 65.2,
                "total_revisions": 1,
                ...
            }
        }
    """

    job_id: str = Field(
        description="Unique job identifier.",
    )

    topic: str = Field(
        description="The original topic provided by the user.",
    )

    final_blog: str = Field(
        description=(
            "The approved blog post in Markdown format. "
            "Render this on the frontend using a Markdown library."
        ),
    )

    evaluation: EvaluationResult = Field(
        description=(
            "The final evaluation report from the Evaluator agent. "
            "Includes score, dimension breakdown, and improvement points "
            "(empty list if blog was approved on first pass)."
        ),
    )

    metadata: BlogMetadata = Field(
        description="Timing, revision count, and content statistics.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_id":     "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "topic":      "The Future of AI in Healthcare",
                "final_blog": "# The Future of AI in Healthcare\n\nArtificial intelligence...",
                "evaluation": {
                    "dimension_scores": [
                        {"dimension": "grammar_clarity",  "score": 8.5, "rationale": "Clear prose throughout."},
                        {"dimension": "factual_accuracy",  "score": 8.0, "rationale": "All claims sourced."},
                        {"dimension": "citation_quality",  "score": 7.5, "rationale": "6 out of 7 citations present."},
                        {"dimension": "structure_flow",    "score": 9.0, "rationale": "Follows outline perfectly."},
                        {"dimension": "seo_optimization",  "score": 8.0, "rationale": "Keywords used naturally."},
                    ],
                    "improvements": [],
                    "feedback": "Excellent blog. Well-structured with strong citations.",
                    "revision_number": 1,
                },
                "metadata": {
                    "job_id":          "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "started_at":      "2024-01-15T10:30:00Z",
                    "completed_at":    "2024-01-15T10:31:05Z",
                    "elapsed_seconds": 65.2,
                    "total_revisions": 1,
                    "sources_used":    5,
                    "word_count":      1247,
                },
            }
        }
    }




class ErrorResponse(BaseModel):
    """
    Consistent error response shape for ALL API errors.
    Registered as the response model for error status codes in the router.

    This is what the frontend always receives when something goes wrong,
    regardless of which layer raised the error (guardrail, agent, storage).

    Example:
        {
            "error_code":  "JOB_NOT_FOUND",
            "message":     "Job 'abc-123' not found. It may have expired.",
            "status_code": 404,
            "detail":      "job_id=abc-123"
        }
    """

    error_code: str = Field(
        description="Machine-readable SCREAMING_SNAKE_CASE error identifier.",
        examples=["JOB_NOT_FOUND", "INPUT_VALIDATION_ERROR", "RATE_LIMIT_EXCEEDED"],
    )

    message: str = Field(
        description="Human-readable explanation of what went wrong.",
    )

    status_code: int = Field(
        description="HTTP status code of this error.",
        examples=[400, 404, 429, 500],
    )

    detail: Optional[str] = Field(
        default=None,
        description="Optional additional context (e.g., the offending field or value).",
    )