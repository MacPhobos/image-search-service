"""Failure injection helpers for testing Qdrant-PostgreSQL desync scenarios.

This module provides mock Qdrant clients that fail in controlled ways to verify
proper error handling and cross-system consistency.
"""

from typing import Any
from uuid import UUID


class FailAfterNCallsQdrant:
    """Mock Qdrant client that fails after N successful calls.

    Used to test partial sync failures in batch operations.

    Usage:
        qdrant = FailAfterNCallsQdrant(
            succeed_count=2,
            error=ConnectionError("timeout")
        )
        # First 2 calls succeed, subsequent calls raise ConnectionError
    """

    def __init__(
        self, succeed_count: int = 0, error: Exception | None = None
    ) -> None:
        """Initialize failure injector.

        Args:
            succeed_count: Number of successful calls before failing
            error: Exception to raise (default: ConnectionError)
        """
        self._succeed_count = succeed_count
        self._error = error or ConnectionError("Qdrant unavailable")
        self._call_count = 0
        self.calls: list[dict[str, Any]] = []

    def update_person_ids(
        self, point_ids: list[UUID], person_id: UUID
    ) -> None:
        """Mock update_person_ids that may fail after N calls.

        Args:
            point_ids: List of Qdrant point IDs
            person_id: Person UUID to assign

        Raises:
            Exception: After succeed_count calls
        """
        self._call_count += 1
        self.calls.append(
            {
                "method": "update_person_ids",
                "point_ids": point_ids,
                "person_id": person_id,
            }
        )
        if self._call_count > self._succeed_count:
            raise self._error

    def upsert_centroid(self, **kwargs: Any) -> None:
        """Mock upsert_centroid that may fail after N calls.

        Args:
            **kwargs: Arbitrary keyword arguments

        Raises:
            Exception: After succeed_count calls
        """
        self._call_count += 1
        self.calls.append({"method": "upsert_centroid", **kwargs})
        if self._call_count > self._succeed_count:
            raise self._error


class SelectiveFailureQdrant:
    """Mock Qdrant client that fails for specific person IDs.

    Used to test partial batch sync failures where some items succeed
    and others fail.

    Usage:
        qdrant = SelectiveFailureQdrant(fail_person_ids={"person-b"})
        qdrant.update_person_ids([...], "person-a")  # succeeds
        qdrant.update_person_ids([...], "person-b")  # raises
    """

    def __init__(
        self, fail_person_ids: set[str] | None = None
    ) -> None:
        """Initialize selective failure injector.

        Args:
            fail_person_ids: Set of person ID strings that should fail
        """
        self._fail_ids = fail_person_ids or set()
        self.successful_calls: list[dict[str, Any]] = []
        self.failed_calls: list[dict[str, Any]] = []

    def update_person_ids(
        self, point_ids: list[UUID], person_id: UUID
    ) -> None:
        """Mock update_person_ids that fails for specific person IDs.

        Args:
            point_ids: List of Qdrant point IDs
            person_id: Person UUID to assign

        Raises:
            ConnectionError: If person_id is in fail set
        """
        call_info = {"point_ids": point_ids, "person_id": str(person_id)}
        if str(person_id) in self._fail_ids:
            self.failed_calls.append(call_info)
            raise ConnectionError(f"Qdrant timeout for person {person_id}")
        self.successful_calls.append(call_info)
