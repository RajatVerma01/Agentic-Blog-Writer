

from typing import Optional


class BlogWriterBaseException(Exception):
    """
    Root exception for all custom errors in this application.
    All other custom exceptions inherit from this.

    Attributes:
        message    : Human-readable description of what went wrong.
        error_code : SCREAMING_SNAKE_CASE identifier (used by frontend).
        status_code: HTTP status code this error maps to.
        detail     : Optional extra context (e.g., the offending field name).
    """

    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int,
        detail: Optional[str] = None,
    ) -> None:
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.detail = detail
        super().__init__(message)

    def to_dict(self) -> dict:
        """
        Converts the exception to a dict suitable for a JSON HTTP response.
        Used by the FastAPI exception handler in main.py.
        """
        payload = {
            "error_code": self.error_code,
            "message": self.message,
            "status_code": self.status_code,
        }
        if self.detail:
            payload["detail"] = self.detail
        return payload

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"error_code={self.error_code!r}, "
            f"message={self.message!r}, "
            f"status_code={self.status_code})"
        )




class InputValidationError(BlogWriterBaseException):
    """
    Raised by the input_validator guardrail when the user-provided
    topic fails any safety or format check.

    Triggers:
    - Topic is too short or too long
    - Topic contains a blocked keyword
    - Prompt injection pattern detected
    - PII (email, phone number) detected in topic

    HTTP Status: 400 Bad Request
    """

    def __init__(self, message: str, detail: Optional[str] = None) -> None:
        super().__init__(
            message=message,
            error_code="INPUT_VALIDATION_ERROR",
            status_code=400,
            detail=detail,
        )




class JobNotFoundError(BlogWriterBaseException):
    """
    Raised when a job_id is looked up in the JobStore but doesn't exist.
    This happens if:
    - The job_id was never created (user sent a made-up ID)
    - The job expired (TTL elapsed, job was cleaned up)

    HTTP Status: 404 Not Found
    """

    def __init__(self, job_id: str) -> None:
        super().__init__(
            message=f"Job '{job_id}' not found. It may have expired or never existed.",
            error_code="JOB_NOT_FOUND",
            status_code=404,
            detail=f"job_id={job_id}",
        )



class RateLimitError(BlogWriterBaseException):
    """
    Raised when a client exceeds the configured rate limit for an endpoint.
    SlowAPI raises its own exception (RateLimitExceeded) which is caught in
    main.py and re-raised as this custom exception for consistent formatting.

    HTTP Status: 429 Too Many Requests
    """

    def __init__(self, limit: str, endpoint: str) -> None:
        super().__init__(
            message=f"Rate limit exceeded: {limit}. Please slow down and try again.",
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            detail=f"endpoint={endpoint}, limit={limit}",
        )



class UnauthorizedError(BlogWriterBaseException):
    """
    Raised when a request is missing a valid API key header,
    and the app is configured to require one (API_SECRET_KEY is set).

    HTTP Status: 401 Unauthorized
    """

    def __init__(self) -> None:
        super().__init__(
            message="Missing or invalid API key. Include 'X-API-Key' header.",
            error_code="UNAUTHORIZED",
            status_code=401,
        )




class AgentExecutionError(BlogWriterBaseException):
    """
    Raised when an agent node crashes unexpectedly during the LangGraph
    pipeline execution (e.g., Groq API error, tool timeout, parsing failure).

    This is caught by blog_service.py which updates the job status to
    'failed' and stores the error message.

    HTTP Status: 500 Internal Server Error

    Attributes:
        agent_name: Which agent failed ('researcher', 'planner', 'writer', 'evaluator')
    """

    def __init__(self, agent_name: str, reason: str) -> None:
        self.agent_name = agent_name
        super().__init__(
            message=f"Agent '{agent_name}' failed during execution: {reason}",
            error_code="AGENT_EXECUTION_ERROR",
            status_code=500,
            detail=f"agent={agent_name}",
        )


class OutputValidationError(BlogWriterBaseException):
    """
    Raised by the output_validator guardrail when the final blog content
    produced by the Writer agent fails safety or quality checks:
    - Blog is too short (< BLOG_MIN_WORDS)
    - Blog is too long  (> BLOG_MAX_WORDS)
    - Hallucinated URLs detected (URL exists in text but returns 4xx/5xx)
    - Harmful content detected in output
    - Blog is missing structural elements (no H1, no sections)

    HTTP Status: 500 Internal Server Error
    (This is a server-side content quality failure, not a user mistake)
    """

    def __init__(self, message: str, detail: Optional[str] = None) -> None:
        super().__init__(
            message=message,
            error_code="OUTPUT_VALIDATION_ERROR",
            status_code=500,
            detail=detail,
        )


class GraphExecutionError(BlogWriterBaseException):
    """
    Raised when the LangGraph graph itself fails to execute
    (e.g., graph compilation error, state corruption, infinite loop guard).
    Distinct from AgentExecutionError which is per-agent.

    HTTP Status: 500 Internal Server Error
    """

    def __init__(self, reason: str) -> None:
        super().__init__(
            message=f"Blog generation pipeline failed: {reason}",
            error_code="GRAPH_EXECUTION_ERROR",
            status_code=500,
            detail=reason,
        )


class ToolExecutionError(BlogWriterBaseException):
    """
    Raised when a LangChain tool (Tavily search, Wikipedia) fails or times out.
    The researcher agent catches this and can fall back gracefully.

    HTTP Status: 500 Internal Server Error

    Attributes:
        tool_name: Which tool failed ('tavily_search', 'wikipedia')
    """

    def __init__(self, tool_name: str, reason: str) -> None:
        self.tool_name = tool_name
        super().__init__(
            message=f"Tool '{tool_name}' failed: {reason}",
            error_code="TOOL_EXECUTION_ERROR",
            status_code=500,
            detail=f"tool={tool_name}, reason={reason}",
        )




from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded


async def blog_writer_exception_handler(
    request: Request,
    exc: BlogWriterBaseException,
) -> JSONResponse:
    """
    Catches any BlogWriterBaseException (and subclasses) and returns a
    consistent JSON error response.

    Example response:
    {
        "error_code":  "JOB_NOT_FOUND",
        "message":     "Job 'abc-123' not found. It may have expired.",
        "status_code": 404
    }
    """
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(),
    )


async def rate_limit_exception_handler(
    request: Request,
    exc: RateLimitExceeded,
) -> JSONResponse:
    """
    Catches SlowAPI's RateLimitExceeded and converts it to our standard
    JSON error format.
    """
    custom_exc = RateLimitError(
        limit=str(exc.detail),
        endpoint=str(request.url.path),
    )
    return JSONResponse(
        status_code=429,
        content=custom_exc.to_dict(),
    )


async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """
    Catches ANY unhandled exception (last resort handler).
    Returns a generic 500 response so we never leak internal stack traces.

    The actual exception is logged (via logger) in blog_service.py before
    this handler fires.
    """
    return JSONResponse(
        status_code=500,
        content={
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred. Please try again later.",
            "status_code": 500,
        },
    )
