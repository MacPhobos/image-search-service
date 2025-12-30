"""Test queue monitoring endpoints."""

from datetime import datetime, timezone

import pytest
from httpx import AsyncClient

from image_search_service.api.queue_schemas import (
    CurrentJobInfo,
    JobDetailResponse,
    JobInfo,
    JobStatus,
    QueueDetailResponse,
    QueueSummary,
    QueuesOverviewResponse,
    WorkerInfo,
    WorkersResponse,
    WorkerState,
)


# Mock Fixtures


@pytest.fixture
def mock_queue_service(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock QueueService for testing queue endpoints.

    Creates a mock service that returns predefined responses without
    requiring Redis or RQ to be running.
    """

    class MockQueueService:
        """Mock implementation of QueueService."""

        def get_queues_overview(self) -> QueuesOverviewResponse:
            """Return mock queues overview."""
            return QueuesOverviewResponse(
                queues=[
                    QueueSummary(
                        name="training-high",
                        count=5,
                        isEmpty=False,
                        startedCount=1,
                        failedCount=0,
                        finishedCount=100,
                        scheduledCount=0,
                    ),
                    QueueSummary(
                        name="training-normal",
                        count=10,
                        isEmpty=False,
                        startedCount=2,
                        failedCount=1,
                        finishedCount=200,
                        scheduledCount=0,
                    ),
                    QueueSummary(
                        name="training-low",
                        count=0,
                        isEmpty=True,
                        startedCount=0,
                        failedCount=0,
                        finishedCount=50,
                        scheduledCount=0,
                    ),
                    QueueSummary(
                        name="default",
                        count=3,
                        isEmpty=False,
                        startedCount=0,
                        failedCount=0,
                        finishedCount=10,
                        scheduledCount=1,
                    ),
                ],
                totalJobs=18,
                totalWorkers=2,
                workersBusy=1,
                redisConnected=True,
            )

        def get_queue_detail(
            self, name: str, page: int, page_size: int
        ) -> QueueDetailResponse | None:
            """Return mock queue detail or None if queue not found."""
            valid_queues = {
                "training-high",
                "training-normal",
                "training-low",
                "default",
            }

            if name not in valid_queues:
                return None

            # Mock jobs for the queue
            jobs = []
            if name == "training-normal":
                # Create sample jobs for pagination testing
                total_jobs = 10
                start_idx = (page - 1) * page_size
                end_idx = min(start_idx + page_size, total_jobs)

                for i in range(start_idx, end_idx):
                    jobs.append(
                        JobInfo(
                            id=f"job-{name}-{i}",
                            funcName="train_embeddings",
                            status=JobStatus.QUEUED,
                            queueName=name,
                            args=["arg1", "arg2"],
                            kwargs={"session_id": "123"},
                            createdAt=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                            enqueuedAt=datetime(
                                2024, 1, 1, 12, 0, 1, tzinfo=timezone.utc
                            ),
                            startedAt=None,
                            endedAt=None,
                            timeout=600,
                            result=None,
                            errorMessage=None,
                            workerName=None,
                        )
                    )

                return QueueDetailResponse(
                    name=name,
                    count=total_jobs,
                    isEmpty=False,
                    jobs=jobs,
                    startedJobs=2,
                    failedJobs=1,
                    page=page,
                    pageSize=page_size,
                    hasMore=end_idx < total_jobs,
                )
            elif name == "training-high":
                # Empty queue for testing
                return QueueDetailResponse(
                    name=name,
                    count=5,
                    isEmpty=False,
                    jobs=[
                        JobInfo(
                            id="job-high-1",
                            funcName="train_embeddings_high_priority",
                            status=JobStatus.STARTED,
                            queueName=name,
                            args=[],
                            kwargs={},
                            createdAt=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                            enqueuedAt=datetime(
                                2024, 1, 1, 12, 0, 1, tzinfo=timezone.utc
                            ),
                            startedAt=datetime(
                                2024, 1, 1, 12, 5, 0, tzinfo=timezone.utc
                            ),
                            endedAt=None,
                            timeout=3600,
                            result=None,
                            errorMessage=None,
                            workerName="worker1",
                        )
                    ],
                    startedJobs=1,
                    failedJobs=0,
                    page=page,
                    pageSize=page_size,
                    hasMore=False,
                )
            else:
                # Empty queue
                return QueueDetailResponse(
                    name=name,
                    count=0,
                    isEmpty=True,
                    jobs=[],
                    startedJobs=0,
                    failedJobs=0,
                    page=page,
                    pageSize=page_size,
                    hasMore=False,
                )

        def get_job_detail(self, job_id: str) -> JobDetailResponse | None:
            """Return mock job detail or None if job not found."""
            if job_id == "nonexistent":
                return None

            # Mock job detail
            if job_id == "failed-job":
                return JobDetailResponse(
                    id=job_id,
                    funcName="train_embeddings",
                    status=JobStatus.FAILED,
                    queueName="training-normal",
                    args=["session123"],
                    kwargs={"force": "true"},
                    createdAt=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
                    enqueuedAt=datetime(2024, 1, 1, 10, 0, 1, tzinfo=timezone.utc),
                    startedAt=datetime(2024, 1, 1, 10, 5, 0, tzinfo=timezone.utc),
                    endedAt=datetime(2024, 1, 1, 10, 10, 0, tzinfo=timezone.utc),
                    timeout=600,
                    result=None,
                    errorMessage="Division by zero",
                    workerName="worker1",
                    excInfo="Traceback (most recent call last):\n  File ...",
                    meta={"retry": "1"},
                    retryCount=1,
                    origin="training-normal",
                )

            # Default successful job
            return JobDetailResponse(
                id=job_id,
                funcName="train_embeddings",
                status=JobStatus.FINISHED,
                queueName="training-normal",
                args=[],
                kwargs={},
                createdAt=datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
                enqueuedAt=datetime(2024, 1, 1, 9, 0, 1, tzinfo=timezone.utc),
                startedAt=datetime(2024, 1, 1, 9, 5, 0, tzinfo=timezone.utc),
                endedAt=datetime(2024, 1, 1, 9, 15, 0, tzinfo=timezone.utc),
                timeout=600,
                result="success",
                errorMessage=None,
                workerName="worker1",
                excInfo=None,
                meta={},
                retryCount=0,
                origin="training-normal",
            )

        def get_workers(self) -> WorkersResponse:
            """Return mock workers information."""
            return WorkersResponse(
                workers=[
                    WorkerInfo(
                        name="worker1.hostname1.12345",
                        state=WorkerState.BUSY,
                        queues=["training-high", "training-normal", "default"],
                        currentJob=CurrentJobInfo(
                            jobId="current-job-1",
                            funcName="train_embeddings",
                            startedAt=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                        ),
                        successfulJobCount=150,
                        failedJobCount=5,
                        totalWorkingTime=36000.5,
                        birthDate=datetime(2024, 1, 1, 8, 0, 0, tzinfo=timezone.utc),
                        lastHeartbeat=datetime(2024, 1, 1, 12, 30, 0, tzinfo=timezone.utc),
                        pid=12345,
                        hostname="hostname1",
                    ),
                    WorkerInfo(
                        name="worker2.hostname2.67890",
                        state=WorkerState.IDLE,
                        queues=["training-low", "default"],
                        currentJob=None,
                        successfulJobCount=80,
                        failedJobCount=2,
                        totalWorkingTime=18000.2,
                        birthDate=datetime(2024, 1, 1, 8, 0, 0, tzinfo=timezone.utc),
                        lastHeartbeat=datetime(2024, 1, 1, 12, 29, 0, tzinfo=timezone.utc),
                        pid=67890,
                        hostname="hostname2",
                    ),
                ],
                total=2,
                active=1,
                idle=1,
            )

    # Patch the get_queue_service function
    monkeypatch.setattr(
        "image_search_service.api.routes.queues.get_queue_service",
        lambda: MockQueueService(),
    )


@pytest.fixture
def mock_queue_service_redis_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock QueueService with Redis connection failure."""

    class MockQueueServiceRedisDown:
        """Mock service simulating Redis disconnection."""

        def get_queues_overview(self) -> QueuesOverviewResponse:
            """Return overview with redisConnected=False."""
            return QueuesOverviewResponse(
                queues=[],
                totalJobs=0,
                totalWorkers=0,
                workersBusy=0,
                redisConnected=False,
            )

        def get_queue_detail(
            self, name: str, page: int, page_size: int
        ) -> QueueDetailResponse | None:
            """Return None when Redis is down."""
            return None

        def get_job_detail(self, job_id: str) -> JobDetailResponse | None:
            """Return None when Redis is down."""
            return None

        def get_workers(self) -> WorkersResponse:
            """Return empty workers when Redis is down."""
            return WorkersResponse(workers=[], total=0, active=0, idle=0)

    monkeypatch.setattr(
        "image_search_service.api.routes.queues.get_queue_service",
        lambda: MockQueueServiceRedisDown(),
    )


# Queues Overview Tests


@pytest.mark.asyncio
async def test_get_queues_overview_returns_all_queues(
    test_client: AsyncClient,
    mock_queue_service: None,
) -> None:
    """Test GET /api/v1/queues returns 200 with all queues."""
    response = await test_client.get("/api/v1/queues")

    assert response.status_code == 200
    data = response.json()

    # Verify structure
    assert "queues" in data
    assert "totalJobs" in data
    assert "totalWorkers" in data
    assert "workersBusy" in data
    assert "redisConnected" in data

    # Verify values
    assert len(data["queues"]) == 4
    assert data["totalJobs"] == 18
    assert data["totalWorkers"] == 2
    assert data["workersBusy"] == 1
    assert data["redisConnected"] is True

    # Verify queue names
    queue_names = {q["name"] for q in data["queues"]}
    assert queue_names == {"training-high", "training-normal", "training-low", "default"}


@pytest.mark.asyncio
async def test_get_queues_overview_includes_queue_details(
    test_client: AsyncClient,
    mock_queue_service: None,
) -> None:
    """Test queue overview includes detailed queue information."""
    response = await test_client.get("/api/v1/queues")

    assert response.status_code == 200
    data = response.json()

    # Find training-normal queue
    training_normal = next(q for q in data["queues"] if q["name"] == "training-normal")

    assert training_normal["count"] == 10
    assert training_normal["isEmpty"] is False
    assert training_normal["startedCount"] == 2
    assert training_normal["failedCount"] == 1
    assert training_normal["finishedCount"] == 200
    assert training_normal["scheduledCount"] == 0


@pytest.mark.asyncio
async def test_get_queues_overview_redis_disconnected(
    test_client: AsyncClient,
    mock_queue_service_redis_down: None,
) -> None:
    """Test queues overview when Redis is disconnected returns 200 with flag."""
    response = await test_client.get("/api/v1/queues")

    assert response.status_code == 200
    data = response.json()

    assert data["redisConnected"] is False
    assert data["queues"] == []
    assert data["totalJobs"] == 0
    assert data["totalWorkers"] == 0


# Queue Detail Tests


@pytest.mark.asyncio
async def test_get_queue_detail_valid_queue(
    test_client: AsyncClient,
    mock_queue_service: None,
) -> None:
    """Test GET /api/v1/queues/{name} returns 200 for valid queue."""
    response = await test_client.get("/api/v1/queues/training-normal")

    assert response.status_code == 200
    data = response.json()

    # Verify structure
    assert data["name"] == "training-normal"
    assert "count" in data
    assert "isEmpty" in data
    assert "jobs" in data
    assert "startedJobs" in data
    assert "failedJobs" in data
    assert "page" in data
    assert "pageSize" in data
    assert "hasMore" in data

    # Verify values
    assert data["count"] == 10
    assert data["isEmpty"] is False
    assert isinstance(data["jobs"], list)
    assert data["page"] == 1
    assert data["pageSize"] == 50


@pytest.mark.asyncio
async def test_get_queue_detail_invalid_queue(
    test_client: AsyncClient,
    mock_queue_service: None,
) -> None:
    """Test GET /api/v1/queues/{name} returns 404 for invalid queue."""
    response = await test_client.get("/api/v1/queues/invalid-queue")

    assert response.status_code == 404
    data = response.json()

    assert "detail" in data
    assert "code" in data["detail"]
    assert "message" in data["detail"]
    assert data["detail"]["code"] == "QUEUE_NOT_FOUND"
    assert "invalid-queue" in data["detail"]["message"]


@pytest.mark.asyncio
async def test_get_queue_detail_pagination_first_page(
    test_client: AsyncClient,
    mock_queue_service: None,
) -> None:
    """Test queue detail pagination first page."""
    response = await test_client.get("/api/v1/queues/training-normal?page=1&pageSize=5")

    assert response.status_code == 200
    data = response.json()

    assert data["page"] == 1
    assert data["pageSize"] == 5
    assert len(data["jobs"]) == 5
    assert data["hasMore"] is True


@pytest.mark.asyncio
async def test_get_queue_detail_pagination_last_page(
    test_client: AsyncClient,
    mock_queue_service: None,
) -> None:
    """Test queue detail pagination last page."""
    response = await test_client.get("/api/v1/queues/training-normal?page=2&pageSize=5")

    assert response.status_code == 200
    data = response.json()

    assert data["page"] == 2
    assert data["pageSize"] == 5
    assert len(data["jobs"]) == 5
    assert data["hasMore"] is False


@pytest.mark.asyncio
async def test_get_queue_detail_includes_job_details(
    test_client: AsyncClient,
    mock_queue_service: None,
) -> None:
    """Test queue detail includes job information."""
    response = await test_client.get("/api/v1/queues/training-high")

    assert response.status_code == 200
    data = response.json()

    assert len(data["jobs"]) == 1
    job = data["jobs"][0]

    assert job["id"] == "job-high-1"
    assert job["funcName"] == "train_embeddings_high_priority"
    assert job["status"] == "started"
    assert job["queueName"] == "training-high"
    assert job["workerName"] == "worker1"
    assert job["timeout"] == 3600


@pytest.mark.asyncio
async def test_get_queue_detail_empty_queue(
    test_client: AsyncClient,
    mock_queue_service: None,
) -> None:
    """Test queue detail for empty queue."""
    response = await test_client.get("/api/v1/queues/training-low")

    assert response.status_code == 200
    data = response.json()

    assert data["name"] == "training-low"
    assert data["count"] == 0
    assert data["isEmpty"] is True
    assert data["jobs"] == []
    assert data["hasMore"] is False


# Job Detail Tests


@pytest.mark.asyncio
async def test_get_job_detail_valid_job(
    test_client: AsyncClient,
    mock_queue_service: None,
) -> None:
    """Test GET /api/v1/jobs/{id} returns 200 for valid job."""
    response = await test_client.get("/api/v1/jobs/test-job-123")

    assert response.status_code == 200
    data = response.json()

    # Verify structure
    assert data["id"] == "test-job-123"
    assert "funcName" in data
    assert "status" in data
    assert "queueName" in data
    assert "args" in data
    assert "kwargs" in data
    assert "createdAt" in data
    assert "excInfo" in data
    assert "meta" in data
    assert "retryCount" in data
    assert "origin" in data

    # Verify values
    assert data["status"] == "finished"
    assert data["queueName"] == "training-normal"
    assert data["result"] == "success"
    assert data["errorMessage"] is None
    assert data["retryCount"] == 0


@pytest.mark.asyncio
async def test_get_job_detail_failed_job(
    test_client: AsyncClient,
    mock_queue_service: None,
) -> None:
    """Test job detail for failed job includes error information."""
    response = await test_client.get("/api/v1/jobs/failed-job")

    assert response.status_code == 200
    data = response.json()

    assert data["id"] == "failed-job"
    assert data["status"] == "failed"
    assert data["errorMessage"] == "Division by zero"
    assert data["excInfo"] is not None
    assert "Traceback" in data["excInfo"]
    assert data["retryCount"] == 1
    assert data["meta"]["retry"] == "1"


@pytest.mark.asyncio
async def test_get_job_detail_not_found(
    test_client: AsyncClient,
    mock_queue_service: None,
) -> None:
    """Test GET /api/v1/jobs/{id} returns 404 for nonexistent job."""
    response = await test_client.get("/api/v1/jobs/nonexistent")

    assert response.status_code == 404
    data = response.json()

    assert "detail" in data
    assert "code" in data["detail"]
    assert "message" in data["detail"]
    assert data["detail"]["code"] == "JOB_NOT_FOUND"
    assert "nonexistent" in data["detail"]["message"]


# Workers Tests


@pytest.mark.asyncio
async def test_get_workers_returns_worker_list(
    test_client: AsyncClient,
    mock_queue_service: None,
) -> None:
    """Test GET /api/v1/workers returns 200 with worker information."""
    response = await test_client.get("/api/v1/workers")

    assert response.status_code == 200
    data = response.json()

    # Verify structure
    assert "workers" in data
    assert "total" in data
    assert "active" in data
    assert "idle" in data

    # Verify values
    assert len(data["workers"]) == 2
    assert data["total"] == 2
    assert data["active"] == 1
    assert data["idle"] == 1


@pytest.mark.asyncio
async def test_get_workers_includes_worker_details(
    test_client: AsyncClient,
    mock_queue_service: None,
) -> None:
    """Test workers endpoint includes detailed worker information."""
    response = await test_client.get("/api/v1/workers")

    assert response.status_code == 200
    data = response.json()

    # Find busy worker
    busy_worker = next(w for w in data["workers"] if w["state"] == "busy")

    assert busy_worker["name"] == "worker1.hostname1.12345"
    assert busy_worker["state"] == "busy"
    assert "training-high" in busy_worker["queues"]
    assert busy_worker["currentJob"] is not None
    assert busy_worker["currentJob"]["jobId"] == "current-job-1"
    assert busy_worker["successfulJobCount"] == 150
    assert busy_worker["failedJobCount"] == 5
    assert busy_worker["pid"] == 12345
    assert busy_worker["hostname"] == "hostname1"


@pytest.mark.asyncio
async def test_get_workers_includes_idle_worker(
    test_client: AsyncClient,
    mock_queue_service: None,
) -> None:
    """Test workers endpoint includes idle worker without current job."""
    response = await test_client.get("/api/v1/workers")

    assert response.status_code == 200
    data = response.json()

    # Find idle worker
    idle_worker = next(w for w in data["workers"] if w["state"] == "idle")

    assert idle_worker["name"] == "worker2.hostname2.67890"
    assert idle_worker["state"] == "idle"
    assert idle_worker["currentJob"] is None
    assert idle_worker["successfulJobCount"] == 80
    assert idle_worker["failedJobCount"] == 2


@pytest.mark.asyncio
async def test_get_workers_redis_down(
    test_client: AsyncClient,
    mock_queue_service_redis_down: None,
) -> None:
    """Test workers endpoint when Redis is down returns empty list."""
    response = await test_client.get("/api/v1/workers")

    assert response.status_code == 200
    data = response.json()

    assert data["workers"] == []
    assert data["total"] == 0
    assert data["active"] == 0
    assert data["idle"] == 0
