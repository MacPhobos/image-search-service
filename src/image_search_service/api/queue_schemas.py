"""Pydantic schemas for queue monitoring API."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class JobStatus(str, Enum):
    """RQ job status enumeration."""

    QUEUED = "queued"
    STARTED = "started"
    DEFERRED = "deferred"
    FINISHED = "finished"
    STOPPED = "stopped"
    SCHEDULED = "scheduled"
    CANCELED = "canceled"
    FAILED = "failed"


class WorkerState(str, Enum):
    """RQ worker state enumeration."""

    IDLE = "idle"
    BUSY = "busy"
    SUSPENDED = "suspended"


class QueueSummary(BaseModel):
    """Summary information for a single queue."""

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(description="Queue name")
    count: int = Field(description="Total number of jobs in queue")
    is_empty: bool = Field(alias="isEmpty", description="Whether queue is empty")
    started_count: int = Field(
        alias="startedCount", description="Number of started jobs"
    )
    failed_count: int = Field(alias="failedCount", description="Number of failed jobs")
    finished_count: int = Field(
        alias="finishedCount", description="Number of finished jobs"
    )
    scheduled_count: int = Field(
        alias="scheduledCount", description="Number of scheduled jobs"
    )


class QueuesOverviewResponse(BaseModel):
    """Overview of all queues and workers."""

    model_config = ConfigDict(populate_by_name=True)

    queues: list[QueueSummary] = Field(description="List of queue summaries")
    total_jobs: int = Field(alias="totalJobs", description="Total jobs across all queues")
    total_workers: int = Field(alias="totalWorkers", description="Total number of workers")
    workers_busy: int = Field(alias="workersBusy", description="Number of busy workers")
    redis_connected: bool = Field(
        alias="redisConnected", description="Redis connection status"
    )


class JobInfo(BaseModel):
    """Basic job information."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(description="Job ID")
    func_name: str = Field(alias="funcName", description="Function name")
    status: JobStatus = Field(description="Current job status")
    queue_name: str = Field(alias="queueName", description="Queue name")
    args: list[str] = Field(default_factory=list, description="Job arguments")
    kwargs: dict[str, str] = Field(default_factory=dict, description="Job keyword arguments")
    created_at: datetime | None = Field(
        None, alias="createdAt", description="Job creation timestamp"
    )
    enqueued_at: datetime | None = Field(
        None, alias="enqueuedAt", description="Job enqueue timestamp"
    )
    started_at: datetime | None = Field(
        None, alias="startedAt", description="Job start timestamp"
    )
    ended_at: datetime | None = Field(
        None, alias="endedAt", description="Job end timestamp"
    )
    timeout: int | None = Field(None, description="Job timeout in seconds")
    result: str | None = Field(None, description="Job result (serialized)")
    error_message: str | None = Field(
        None, alias="errorMessage", description="Error message if failed"
    )
    worker_name: str | None = Field(
        None, alias="workerName", description="Worker name that processed job"
    )


class QueueDetailResponse(BaseModel):
    """Detailed information about a specific queue."""

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(description="Queue name")
    count: int = Field(description="Total number of jobs in queue")
    is_empty: bool = Field(alias="isEmpty", description="Whether queue is empty")
    jobs: list[JobInfo] = Field(description="List of jobs in queue")
    started_jobs: int = Field(
        alias="startedJobs", description="Number of started jobs"
    )
    failed_jobs: int = Field(alias="failedJobs", description="Number of failed jobs")
    page: int = Field(description="Current page number")
    page_size: int = Field(alias="pageSize", description="Page size")
    has_more: bool = Field(alias="hasMore", description="Whether more jobs exist")


class JobDetailResponse(JobInfo):
    """Extended job information with additional details."""

    model_config = ConfigDict(populate_by_name=True)

    exc_info: str | None = Field(
        None, alias="excInfo", description="Exception traceback if failed"
    )
    meta: dict[str, str] = Field(
        default_factory=dict, description="Job metadata"
    )
    retry_count: int = Field(
        0, alias="retryCount", description="Number of retry attempts"
    )
    origin: str | None = Field(None, description="Queue origin")


class CurrentJobInfo(BaseModel):
    """Information about a worker's current job."""

    model_config = ConfigDict(populate_by_name=True)

    job_id: str = Field(alias="jobId", description="Job ID")
    func_name: str = Field(alias="funcName", description="Function name")
    started_at: datetime = Field(alias="startedAt", description="Job start timestamp")


class WorkerInfo(BaseModel):
    """Information about a worker."""

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(description="Worker name")
    state: WorkerState = Field(description="Worker state")
    queues: list[str] = Field(description="Queues worker is listening to")
    current_job: CurrentJobInfo | None = Field(
        None, alias="currentJob", description="Currently executing job"
    )
    successful_job_count: int = Field(
        alias="successfulJobCount", description="Number of successful jobs"
    )
    failed_job_count: int = Field(
        alias="failedJobCount", description="Number of failed jobs"
    )
    total_working_time: float = Field(
        alias="totalWorkingTime", description="Total working time in seconds"
    )
    birth_date: datetime = Field(alias="birthDate", description="Worker start time")
    last_heartbeat: datetime = Field(
        alias="lastHeartbeat", description="Last heartbeat timestamp"
    )
    pid: int = Field(description="Process ID")
    hostname: str = Field(description="Hostname")


class WorkersResponse(BaseModel):
    """List of workers with summary statistics."""

    model_config = ConfigDict(populate_by_name=True)

    workers: list[WorkerInfo] = Field(description="List of workers")
    total: int = Field(description="Total number of workers")
    active: int = Field(description="Number of active (busy) workers")
    idle: int = Field(description="Number of idle workers")
