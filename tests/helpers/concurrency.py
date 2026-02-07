"""Concurrency test helpers for race condition testing.

This module provides utilities for testing concurrent operations and detecting
race conditions in the image-search-service codebase.

Key utilities:
- race_requests(): Execute coroutines concurrently to widen race windows
- OperationLogger: Thread-safe operation tracking for analyzing interleaving
- DelayedMockSession: Mock DB session with configurable I/O delays
"""

import asyncio
from collections.abc import Awaitable, Sequence
from typing import Any


async def race_requests(
    coroutines: Sequence[Awaitable[Any]],
    stagger_ms: float = 0,
) -> list[Any]:
    """Execute multiple coroutines concurrently to test race conditions.

    Launches all coroutines as close to simultaneously as possible to maximize
    the race window for detecting concurrency bugs.

    Args:
        coroutines: List of async callables to execute
        stagger_ms: Optional delay between launches in milliseconds (default: 0)

    Returns:
        List of results or exceptions (uses return_exceptions=True internally)

    Example:
        results = await race_requests([
            client.post("/endpoint", json=data1),
            client.post("/endpoint", json=data2),
        ])
    """
    if stagger_ms > 0:
        tasks: list[asyncio.Task[Any]] = []
        for coro in coroutines:
            task: asyncio.Task[Any] = asyncio.create_task(coro)  # type: ignore[arg-type]
            tasks.append(task)
            await asyncio.sleep(stagger_ms / 1000)
        return await asyncio.gather(*tasks, return_exceptions=True)
    else:
        # Maximum race window: launch all at once
        return await asyncio.gather(*coroutines, return_exceptions=True)


class OperationLogger:
    """Thread-safe logger for tracking operation ordering in race condition tests.

    Records operations with timestamps to analyze interleaving patterns
    and verify race condition behavior.

    Example:
        logger = OperationLogger()
        await logger.log("request_1_started")
        await logger.log("request_2_started")
        # Later:
        assert logger.count("started") == 2
    """

    def __init__(self) -> None:
        """Initialize operation logger with empty log."""
        self.operations: list[tuple[float, str]] = []
        self._lock = asyncio.Lock()

    async def log(self, operation: str) -> None:
        """Log an operation with timestamp.

        Args:
            operation: Operation description string
        """
        async with self._lock:
            timestamp = asyncio.get_event_loop().time()
            self.operations.append((timestamp, operation))

    def get_log(self) -> list[str]:
        """Get operations in chronological order.

        Returns:
            List of operation strings sorted by timestamp
        """
        return [op for _, op in sorted(self.operations)]

    def count(self, pattern: str) -> int:
        """Count operations matching a pattern.

        Args:
            pattern: Substring to search for in operation names

        Returns:
            Number of operations containing the pattern
        """
        return sum(1 for _, op in self.operations if pattern in op)


class DelayedMockSession:
    """Mock DB session that adds configurable delays to simulate real I/O.

    Wraps a base session and injects artificial delays to widen race windows
    for testing concurrent database operations.

    Usage:
        session = DelayedMockSession(
            base_session,
            read_delay_ms=10,
            commit_delay_ms=20
        )
    """

    def __init__(
        self,
        base_session: Any,
        read_delay_ms: float = 0,
        commit_delay_ms: float = 0,
    ) -> None:
        """Initialize delayed mock session.

        Args:
            base_session: Underlying session to wrap
            read_delay_ms: Delay for read operations (get, execute) in milliseconds
            commit_delay_ms: Delay for commit operations in milliseconds
        """
        self._base = base_session
        self._read_delay = read_delay_ms / 1000
        self._commit_delay = commit_delay_ms / 1000

    async def get(self, model: type, pk: Any, **kwargs: Any) -> Any:
        """Get entity by primary key with optional delay.

        Args:
            model: SQLAlchemy model class
            pk: Primary key value
            **kwargs: Additional arguments passed to base session

        Returns:
            Entity instance or None
        """
        if self._read_delay:
            await asyncio.sleep(self._read_delay)
        return await self._base.get(model, pk, **kwargs)

    async def commit(self) -> None:
        """Commit transaction with optional delay."""
        if self._commit_delay:
            await asyncio.sleep(self._commit_delay)
        await self._base.commit()

    # Delegate all other attributes to base session
    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to base session.

        Args:
            name: Attribute name

        Returns:
            Attribute value from base session
        """
        return getattr(self._base, name)
