import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from typing import Optional

from app.schemas.blog import JobStatusEnum, AgentNameEnum
from app.config.settings import get_settings
from app.utils.exceptions import JobNotFoundError
from app.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class JobRecord:
    """
    In-memory record for a single blog generation job.

    Attributes are updated in-place by blog_service.py throughout the
    pipeline lifecycle. Using a class (not a dict) gives type safety and
    prevents missing-key bugs.

    Lifecycle:
        created       → status=QUEUED,   current_agent=NONE
        pipeline runs → status=RUNNING,  current_agent=researcher/planner/writer/evaluator
        approved      → status=COMPLETED, final_blog set
        error         → status=FAILED,   error set
    """

    __slots__ = (
        "job_id", "topic", "status", "current_agent",
        "final_blog", "evaluation_summary", "error",
        "created_at", "updated_at", "completed_at",
        "revision_count",
    )

    def __init__(self, job_id: str, topic: str) -> None:
        self.job_id: str                       = job_id
        self.topic: str                        = topic
        self.status: JobStatusEnum             = JobStatusEnum.QUEUED
        self.current_agent: AgentNameEnum      = AgentNameEnum.NONE
        self.final_blog: Optional[str]         = None
        self.evaluation_summary: Optional[dict]= None
        self.error: Optional[str]              = None
        self.revision_count: int               = 0
        self.created_at: datetime              = datetime.now(tz=timezone.utc)
        self.updated_at: datetime              = datetime.now(tz=timezone.utc)
        self.completed_at: Optional[datetime]  = None

    def is_expired(self, ttl_hours: int) -> bool:
        """Returns True if this job is older than ttl_hours."""
        age = datetime.now(tz=timezone.utc) - self.created_at
        return age > timedelta(hours=ttl_hours)

    def to_status_dict(self) -> dict:
        """
        Serializes the job record to a plain dict for the /status endpoint.
        Only fields relevant to a status poll — NOT the full blog content.
        """
        return {
            "job_id":         self.job_id,
            "status":         self.status.value,
            "current_agent":  self.current_agent.value,
            "revision_count": self.revision_count,
            "error":          self.error,
            "created_at":     self.created_at.isoformat(),
            "updated_at":     self.updated_at.isoformat(),
            "completed_at":   self.completed_at.isoformat() if self.completed_at else None,
        }

    def to_result_dict(self) -> dict:
        """
        Serializes the full job result for the /result endpoint.
        Includes final_blog and evaluation_summary.
        """
        return {
            **self.to_status_dict(),
            "topic":               self.topic,
            "final_blog":          self.final_blog,
            "evaluation_summary":  self.evaluation_summary,
        }


class JobStore:
    """
    Thread-safe in-memory store for all active blog generation jobs.

    Uses asyncio.Lock to prevent race conditions when multiple async
    coroutines read/write the same job simultaneously.

    Storage: plain dict { job_id: JobRecord }
    No database, no Redis — all in process memory. Jobs are lost on restart.
    TTL cleanup runs periodically via blog_service.py background task.
    """

    def __init__(self) -> None:
        self._store: dict[str, JobRecord] = {}
        self._lock = asyncio.Lock()

    async def create(self, topic: str) -> JobRecord:
        """
        Creates a new job record with a fresh UUID and returns it.

        Args:
            topic: The validated blog topic string.

        Returns:
            The newly created JobRecord (status=QUEUED).
        """
        job_id = str(uuid.uuid4())
        record = JobRecord(job_id=job_id, topic=topic)

        async with self._lock:
            self._store[job_id] = record

        logger.info("Job created", extra={"job_id": job_id, "topic_len": len(topic)})
        return record

    async def get(self, job_id: str) -> JobRecord:
        """
        Retrieves a job record by ID.

        Args:
            job_id: The UUID string to look up.

        Returns:
            The JobRecord for this job_id.

        Raises:
            JobNotFoundError: If job_id doesn't exist or has been cleaned up.
        """
        async with self._lock:
            record = self._store.get(job_id)

        if record is None:
            raise JobNotFoundError(job_id)

        return record

    async def update_status(
        self,
        job_id: str,
        status: JobStatusEnum,
        current_agent: AgentNameEnum = AgentNameEnum.NONE,
    ) -> None:
        """
        Updates the job status and current running agent.

        Called by blog_service.py at each pipeline transition:
            await job_store.update_status(job_id, RUNNING, AgentNameEnum.RESEARCHER)

        Args:
            job_id:        The job to update.
            status:        New JobStatusEnum value.
            current_agent: Which agent is now running (optional).
        """
        async with self._lock:
            record = self._store.get(job_id)
            if record is None:
                return  # Job may have been cleaned up — silently ignore

        record.status = status
        record.current_agent = current_agent
        record.updated_at = datetime.now(tz=timezone.utc)
        logger.info(
            "Job status updated",
            extra={"job_id": job_id, "status": status.value, "agent": current_agent.value},
        )

    async def complete(
        self,
        job_id: str,
        final_blog: str,
        evaluation_summary: Optional[dict] = None,
    ) -> None:
        """
        Marks a job as COMPLETED and stores the final blog content.

        Args:
            job_id:              The job to complete.
            final_blog:          The approved Markdown blog post.
            evaluation_summary:  Optional dict of scores for the result response.
        """
        async with self._lock:
            record = self._store.get(job_id)
            if record is None:
                return

        record.status             = JobStatusEnum.COMPLETED
        record.current_agent      = AgentNameEnum.DONE
        record.final_blog         = final_blog
        record.evaluation_summary = evaluation_summary
        record.completed_at       = datetime.now(tz=timezone.utc)
        record.updated_at         = record.completed_at

        logger.info("Job completed", extra={"job_id": job_id})

    async def fail(self, job_id: str, error: str) -> None:
        """
        Marks a job as FAILED and stores the error message.

        Args:
            job_id: The job to fail.
            error:  Human-readable error string stored in the record.
        """
        async with self._lock:
            record = self._store.get(job_id)
            if record is None:
                return

        record.status        = JobStatusEnum.FAILED
        record.current_agent = AgentNameEnum.NONE
        record.error         = error
        record.completed_at  = datetime.now(tz=timezone.utc)
        record.updated_at    = record.completed_at

        logger.error("Job failed", extra={"job_id": job_id, "error": error})

    async def cleanup_expired(self) -> int:
        """
        Removes all jobs older than JOB_TTL_HOURS (from settings).

        Called periodically by a background task in blog_service.py.
        Returns the count of deleted records for logging.

        Returns:
            Number of jobs removed.
        """
        ttl = settings.JOB_TTL_HOURS
        expired_ids = []

        async with self._lock:
            expired_ids = [
                jid for jid, rec in self._store.items()
                if rec.is_expired(ttl)
            ]
            for jid in expired_ids:
                del self._store[jid]

        if expired_ids:
            logger.info("Expired jobs cleaned up", extra={"count": len(expired_ids)})

        return len(expired_ids)

    @property
    def total_jobs(self) -> int:
        """Returns the current number of active jobs in memory."""
        return len(self._store)


@lru_cache(maxsize=1)
def get_job_store() -> JobStore:
    """
    Returns the JobStore singleton.

    @lru_cache ensures only ONE JobStore instance exists in the process.
    All modules that import get_job_store() share the same in-memory dict.

    In tests, call get_job_store.cache_clear() to get a fresh empty store.
    """
    return JobStore()