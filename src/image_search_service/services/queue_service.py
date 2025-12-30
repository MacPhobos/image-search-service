"""Service for RQ queue monitoring and introspection."""

from datetime import UTC, datetime

from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from rq import Queue
from rq.job import Job
from rq.registry import FailedJobRegistry, FinishedJobRegistry, StartedJobRegistry
from rq.worker import Worker

from image_search_service.api.queue_schemas import (
    CurrentJobInfo,
    JobDetailResponse,
    JobInfo,
    JobStatus,
    QueueDetailResponse,
    QueuesOverviewResponse,
    QueueSummary,
    WorkerInfo,
    WorkersResponse,
    WorkerState,
)
from image_search_service.core.logging import get_logger
from image_search_service.queue.worker import (
    QUEUE_DEFAULT,
    QUEUE_HIGH,
    QUEUE_LOW,
    QUEUE_NORMAL,
    get_redis,
)

logger = get_logger(__name__)

QUEUE_NAMES = [QUEUE_HIGH, QUEUE_NORMAL, QUEUE_LOW, QUEUE_DEFAULT]


class QueueService:
    """Service for RQ queue monitoring (synchronous - RQ uses sync Redis)."""

    def __init__(self) -> None:
        """Initialize the queue service."""
        self._redis: Redis | None = None

    def _get_redis(self) -> Redis:
        """Get cached Redis connection."""
        if self._redis is None:
            self._redis = get_redis()
        return self._redis

    def _is_redis_connected(self) -> bool:
        """Check Redis connectivity."""
        try:
            self._get_redis().ping()
            return True
        except (RedisConnectionError, Exception):
            return False

    def _get_queue(self, name: str) -> Queue:
        """Get Queue instance."""
        return Queue(name, connection=self._get_redis())

    def _map_job_status(self, rq_status: str | None) -> JobStatus:
        """Map RQ status string to enum."""
        mapping = {
            "queued": JobStatus.QUEUED,
            "started": JobStatus.STARTED,
            "deferred": JobStatus.DEFERRED,
            "finished": JobStatus.FINISHED,
            "stopped": JobStatus.STOPPED,
            "scheduled": JobStatus.SCHEDULED,
            "canceled": JobStatus.CANCELED,
            "failed": JobStatus.FAILED,
        }
        return mapping.get(rq_status or "", JobStatus.QUEUED)

    def _map_worker_state(self, state: str | None) -> WorkerState:
        """Map RQ worker state to enum."""
        mapping = {
            "idle": WorkerState.IDLE,
            "busy": WorkerState.BUSY,
            "suspended": WorkerState.SUSPENDED,
        }
        return mapping.get(state or "", WorkerState.IDLE)

    def _job_to_info(self, job: Job, queue_name: str | None = None) -> JobInfo:
        """Convert RQ Job to JobInfo schema."""
        args = [str(a) for a in (job.args or [])]
        kwargs = {k: str(v) for k, v in (job.kwargs or {}).items()}

        error_msg = None
        if job.exc_info:
            lines = str(job.exc_info).strip().split("\n")
            error_msg = lines[-1] if lines else str(job.exc_info)

        return JobInfo(
            id=job.id,
            funcName=job.func_name or "unknown",
            status=self._map_job_status(job.get_status()),
            queueName=queue_name or job.origin or "unknown",
            args=args,
            kwargs=kwargs,
            createdAt=job.created_at,
            enqueuedAt=job.enqueued_at,
            startedAt=job.started_at,
            endedAt=job.ended_at,
            timeout=int(job.timeout) if job.timeout else None,
            result=str(job.result) if job.result is not None else None,
            errorMessage=error_msg,
            workerName=getattr(job, "worker_name", None),
        )

    def get_queues_overview(self) -> QueuesOverviewResponse:
        """Get overview of all queues."""
        if not self._is_redis_connected():
            return QueuesOverviewResponse(
                queues=[],
                totalJobs=0,
                totalWorkers=0,
                workersBusy=0,
                redisConnected=False,
            )

        redis = self._get_redis()
        queues: list[QueueSummary] = []
        total_jobs = 0

        for queue_name in QUEUE_NAMES:
            queue = self._get_queue(queue_name)

            started_registry = StartedJobRegistry(queue_name, connection=redis)
            failed_registry = FailedJobRegistry(queue_name, connection=redis)
            finished_registry = FinishedJobRegistry(queue_name, connection=redis)

            count = queue.count
            total_jobs += count

            queues.append(
                QueueSummary(
                    name=queue_name,
                    count=count,
                    isEmpty=queue.is_empty(),
                    startedCount=len(started_registry),
                    failedCount=len(failed_registry),
                    finishedCount=len(finished_registry),
                    scheduledCount=0,
                )
            )

        workers = Worker.all(connection=redis)
        total_workers = len(workers)
        workers_busy = sum(1 for w in workers if w.get_state() == "busy")

        return QueuesOverviewResponse(
            queues=queues,
            totalJobs=total_jobs,
            totalWorkers=total_workers,
            workersBusy=workers_busy,
            redisConnected=True,
        )

    def get_queue_detail(
        self, name: str, page: int = 1, page_size: int = 50
    ) -> QueueDetailResponse | None:
        """Get queue details with paginated jobs."""
        if name not in QUEUE_NAMES:
            return None

        redis = self._get_redis()
        queue = self._get_queue(name)

        offset = (page - 1) * page_size
        job_ids = queue.get_job_ids(offset=offset, length=page_size)
        jobs = []
        for jid in job_ids:
            if Job.exists(jid, connection=redis):
                job = Job.fetch(jid, connection=redis)
                jobs.append(self._job_to_info(job, name))

        started_registry = StartedJobRegistry(name, connection=redis)
        started_count = len(started_registry)

        failed_registry = FailedJobRegistry(name, connection=redis)
        failed_count = len(failed_registry)

        total_count = queue.count
        has_more = offset + page_size < total_count

        return QueueDetailResponse(
            name=name,
            count=total_count,
            isEmpty=queue.is_empty(),
            jobs=jobs,
            startedJobs=started_count,
            failedJobs=failed_count,
            page=page,
            pageSize=page_size,
            hasMore=has_more,
        )

    def get_job_detail(self, job_id: str) -> JobDetailResponse | None:
        """Get detailed job information."""
        redis = self._get_redis()

        if not Job.exists(job_id, connection=redis):
            return None

        job = Job.fetch(job_id, connection=redis)
        base = self._job_to_info(job)

        return JobDetailResponse(
            **base.model_dump(by_alias=True),
            excInfo=str(job.exc_info) if job.exc_info else None,
            meta=job.meta if job.meta else {},
            retryCount=getattr(job, "retries_left", 0),
            origin=job.origin,
        )

    def get_workers(self) -> WorkersResponse:
        """Get all workers with status."""
        if not self._is_redis_connected():
            return WorkersResponse(workers=[], total=0, active=0, idle=0)

        redis = self._get_redis()
        rq_workers = Worker.all(connection=redis)

        workers: list[WorkerInfo] = []
        active = 0
        idle = 0

        for w in rq_workers:
            state = self._map_worker_state(w.get_state())

            if state == WorkerState.BUSY:
                active += 1
            elif state == WorkerState.IDLE:
                idle += 1

            current_job = None
            rq_job = w.get_current_job()
            if rq_job and rq_job.started_at:
                current_job = CurrentJobInfo(
                    jobId=rq_job.id,
                    funcName=rq_job.func_name or "unknown",
                    startedAt=rq_job.started_at,
                )

            workers.append(
                WorkerInfo(
                    name=w.name,
                    state=state,
                    queues=[q.name for q in w.queues],
                    currentJob=current_job,
                    successfulJobCount=w.successful_job_count,
                    failedJobCount=w.failed_job_count,
                    totalWorkingTime=w.total_working_time,
                    birthDate=w.birth_date or datetime.now(UTC),
                    lastHeartbeat=w.last_heartbeat or datetime.now(UTC),
                    pid=w.pid or 0,
                    hostname=w.hostname or "unknown",
                )
            )

        return WorkersResponse(
            workers=workers,
            total=len(workers),
            active=active,
            idle=idle,
        )
