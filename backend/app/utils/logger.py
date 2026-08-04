import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any



class JsonFormatter(logging.Formatter):
    """
    Formats every log record as a single-line JSON object.

    Every log line will look like:
    {
        "timestamp": "2024-01-15T10:30:00.123456Z",
        "level":     "INFO",
        "logger":    "app.agents.researcher.agent",
        "message":   "Research complete",
        "agent":     "researcher",
        "job_id":    "abc-123-xyz",
        "sources":   5
    }

    Any keyword args passed to logger.info(..., extra={...}) are merged
    directly into the JSON object at the top level.
    """

    
    RESERVED_ATTRS = {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "message", "module",
        "msecs", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName",
        "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        # Format the exception traceback if present
        if record.exc_info:
            record.exc_text = self.formatException(record.exc_info)

        # Build the base log payload
        log_object: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level":   record.levelname,
            "logger":  record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_text:
            log_object["exception"] = record.exc_text

        # Merge any extra fields passed via extra={...}
        for key, value in record.__dict__.items():
            if key not in self.RESERVED_ATTRS:
                log_object[key] = value

        return json.dumps(log_object, default=str)



class DevFormatter(logging.Formatter):
    """
    Formats log records in a colorized, human-readable format for the terminal.

    Example output:
    2024-01-15 10:30:00 | INFO     | app.agents.researcher | Research complete | job_id=abc-123
    2024-01-15 10:30:05 | ERROR    | app.tools.search      | Tavily timed out  | tool=tavily
    """

    LEVEL_COLORS = {
        "DEBUG":    "\033[36m",   # Cyan
        "INFO":     "\033[32m",   # Green
        "WARNING":  "\033[33m",   # Yellow
        "ERROR":    "\033[31m",   # Red
        "CRITICAL": "\033[35m",   # Magenta
    }
    RESET = "\033[0m"
    RESERVED_ATTRS = JsonFormatter.RESERVED_ATTRS

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelname, self.RESET)
        timestamp = datetime.fromtimestamp(
            record.created, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S")

        # Collect extra fields
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in self.RESERVED_ATTRS
        }
        extras_str = " | " + " ".join(
            f"{k}={v}" for k, v in extras.items()
        ) if extras else ""

        # Format exception if present
        exc_str = ""
        if record.exc_info:
            exc_str = "\n" + self.formatException(record.exc_info)

        return (
            f"{timestamp} | "
            f"{color}{record.levelname:<8}{self.RESET} | "
            f"{record.name:<35} | "
            f"{record.getMessage()}"
            f"{extras_str}"
            f"{exc_str}"
        )




def _is_development() -> bool:
    """
    Checks APP_ENV without creating a circular import with settings.py.
    Falls back to 'development' if the env var is not set.
    """
    import os
    return os.getenv("APP_ENV", "development") == "development"


def _get_log_level() -> int:
    """
    Reads LOG_LEVEL from environment. Defaults to INFO.
    Maps string → logging constant (e.g., "DEBUG" → logging.DEBUG).
    """
    import os
    level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_str, logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a named logger configured for either production (JSON)
    or development (colorized human-readable) output.

    This function is idempotent — calling it twice with the same name
    returns the same logger without adding duplicate handlers.

    Args:
        name: Usually __name__ of the calling module.
              e.g., "app.agents.researcher.agent"

    Returns:
        A configured logging.Logger instance.

    Example:
        logger = get_logger(__name__)
        logger.info("Blog generation started", extra={"job_id": "abc-123"})
        logger.warning("Low score, triggering revision", extra={"score": 6.2})
        logger.error("Tavily search failed", extra={"tool": "tavily"}, exc_info=True)
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if get_logger is called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(_get_log_level())

    # Create stdout handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(_get_log_level())

    # Choose formatter based on environment
    if _is_development():
        handler.setFormatter(DevFormatter())
    else:
        handler.setFormatter(JsonFormatter())

    logger.addHandler(handler)

    # Prevent log records from propagating to the root logger
    # (avoids duplicate log lines)
    logger.propagate = False

    return logger



app_logger = get_logger("app")


class AgentLoggerAdapter(logging.LoggerAdapter):
    """
    A LoggerAdapter that automatically injects agent context into every
    log message, so you don't have to pass extra={} on every call.

    Usage inside any agent node:
        from app.utils.logger import get_agent_logger
        logger = get_agent_logger("researcher", job_id="abc-123")
        logger.info("Starting web search")
        # Produces: {"agent": "researcher", "job_id": "abc-123", "message": "Starting web search"}
    """

    def process(
        self, msg: str, kwargs: dict
    ) -> tuple[str, dict]:
        # Merge the adapter's context into every log call's extra dict
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs


def get_agent_logger(agent_name: str, job_id: str = "") -> AgentLoggerAdapter:
    """
    Returns a logger pre-loaded with agent_name and job_id context.
    Every log line from this logger will automatically include those fields.

    Args:
        agent_name: "researcher" | "planner" | "writer" | "evaluator"
        job_id:     The UUID of the current blog generation job.

    Example:
        logger = get_agent_logger("writer", job_id="abc-123")
        logger.info("Draft complete")
        # JSON output: {"agent": "writer", "job_id": "abc-123", "message": "Draft complete"}
    """
    base_logger = get_logger(f"app.agents.{agent_name}")
    return AgentLoggerAdapter(
        base_logger,
        extra={"agent": agent_name, "job_id": job_id},
    )