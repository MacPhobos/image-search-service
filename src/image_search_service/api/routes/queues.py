"""Queue monitoring endpoints."""

from fastapi import APIRouter, HTTPException, Query, status

from image_search_service.api.queue_schemas import (
    JobDetailResponse,
    QueueDetailResponse,
    QueuesOverviewResponse,
    WorkersResponse,
)
from image_search_service.core.logging import get_logger
from image_search_service.services.queue_service import QueueService

logger = get_logger(__name__)

# Main queues router
router = APIRouter(prefix="/queues", tags=["queues"])

# Separate routers for jobs and workers (they have different prefixes)
jobs_router = APIRouter(prefix="/jobs", tags=["jobs"])
workers_router = APIRouter(prefix="/workers", tags=["workers"])


def get_queue_service() -> QueueService:
    """Dependency for QueueService."""
    return QueueService()


@router.get("", response_model=QueuesOverviewResponse)
def get_queues_overview() -> QueuesOverviewResponse:
    """Get overview of all queues.

    Returns summary counts for all RQ queues including jobs in queue,
    started/failed/finished counts, worker totals, and Redis connection status.
    """
    service = get_queue_service()
    return service.get_queues_overview()


@router.get("/{queue_name}", response_model=QueueDetailResponse)
def get_queue_detail(
    queue_name: str,
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(50, ge=1, le=100, alias="pageSize", description="Items per page"),
) -> QueueDetailResponse:
    """Get detailed information for a specific queue.

    Valid queues: training-high, training-normal, training-low, default
    """
    service = get_queue_service()
    result = service.get_queue_detail(queue_name, page, page_size)

    if result is None:
        valid_queues = "training-high, training-normal, training-low, default"
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "QUEUE_NOT_FOUND",
                "message": f"Queue '{queue_name}' not found. Valid queues: {valid_queues}",
            },
        )

    return result


@jobs_router.get("/{job_id}", response_model=JobDetailResponse)
def get_job_detail(job_id: str) -> JobDetailResponse:
    """Get detailed information for a specific RQ job."""
    service = get_queue_service()
    result = service.get_job_detail(job_id)

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "JOB_NOT_FOUND",
                "message": f"Job '{job_id}' not found",
            },
        )

    return result


@workers_router.get("", response_model=WorkersResponse)
def get_workers() -> WorkersResponse:
    """Get information about all RQ workers."""
    service = get_queue_service()
    return service.get_workers()
