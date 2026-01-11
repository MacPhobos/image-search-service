"""Job progress and status endpoints."""

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from redis import Redis

from image_search_service.core.config import get_settings

logger = logging.getLogger(__name__)

# Use distinct router name to avoid conflict with queues.jobs_router
job_progress_router = APIRouter(prefix="/job-progress", tags=["job-progress"])


def get_redis() -> Redis:
    """Get Redis client (lazy initialization)."""
    settings = get_settings()
    return Redis.from_url(settings.redis_url)


@job_progress_router.get(
    "/events",
    summary="Stream job progress via Server-Sent Events",
    description="""
    Opens an SSE connection that streams progress updates for a background job.

    Pass the `progress_key` returned from the job creation endpoint.

    Events:
    - `progress`: Periodic progress updates
    - `complete`: Final result when job finishes
    - `error`: Error message if job fails

    The stream automatically closes when the job completes or fails.
    """,
)
async def stream_job_events(progress_key: str) -> StreamingResponse:
    """Stream real-time progress updates for a background job."""

    redis = get_redis()

    # Verify the key exists (job was started)
    if not redis.exists(progress_key):
        raise HTTPException(status_code=404, detail="Job not found or expired")

    async def event_generator() -> Any:
        """Generate SSE events from Redis progress updates."""
        last_data = None
        poll_interval = 1.0  # seconds
        max_polls = 600  # 10 minutes max

        for _ in range(max_polls):
            # Get current progress
            data = redis.get(progress_key)

            if data:
                data_str = data.decode() if isinstance(data, bytes) else data
                assert isinstance(data_str, str), "Expected string from Redis"

                # Only send if changed
                if data_str != last_data:
                    last_data = data_str
                    progress: dict[str, Any] = json.loads(data_str)

                    # Determine event type
                    phase = progress.get("phase", "")
                    if phase == "completed":
                        yield f"event: complete\ndata: {data_str}\n\n"
                        return
                    elif phase == "failed":
                        yield f"event: error\ndata: {data_str}\n\n"
                        return
                    else:
                        yield f"event: progress\ndata: {data_str}\n\n"

            await asyncio.sleep(poll_interval)

        # Timeout
        yield 'event: error\ndata: {"error": "Timeout waiting for job"}\n\n'

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@job_progress_router.get(
    "/status",
    summary="Get current job status (non-streaming)",
)
async def get_job_status(progress_key: str) -> dict[str, Any]:
    """Get current status of a background job."""

    redis = get_redis()

    data = redis.get(progress_key)
    if not data:
        raise HTTPException(status_code=404, detail="Job not found or expired")

    data_str = data.decode() if isinstance(data, bytes) else str(data)
    result: dict[str, Any] = json.loads(data_str)
    return result
