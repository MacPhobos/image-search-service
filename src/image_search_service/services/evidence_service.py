"""Evidence service for managing training evidence records."""

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.core.logging import get_logger
from image_search_service.db.models import TrainingEvidence

logger = get_logger(__name__)


class EvidenceService:
    """Service for managing training evidence and metadata."""

    async def create_evidence(
        self,
        db: AsyncSession,
        asset_id: int,
        session_id: int,
        model_name: str,
        model_version: str,
        embedding_checksum: str | None,
        device: str,
        processing_time_ms: int,
        error_message: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> TrainingEvidence:
        """Create a training evidence record.

        Args:
            db: Database session
            asset_id: Image asset ID
            session_id: Training session ID
            model_name: Model name (e.g., "OpenCLIP")
            model_version: Model version
            embedding_checksum: MD5 or SHA256 checksum of embedding
            device: Device used (cuda:0, cpu, etc.)
            processing_time_ms: Processing time in milliseconds
            error_message: Optional error message for failures
            metadata: Optional metadata dictionary

        Returns:
            Created evidence record
        """
        evidence = TrainingEvidence(
            asset_id=asset_id,
            session_id=session_id,
            model_name=model_name,
            model_version=model_version,
            embedding_checksum=embedding_checksum,
            device=device,
            processing_time_ms=processing_time_ms,
            error_message=error_message,
            metadata_json=metadata,
        )

        db.add(evidence)
        await db.commit()
        await db.refresh(evidence)

        logger.debug(f"Created evidence {evidence.id} for asset {asset_id}")
        return evidence

    async def get_evidence(
        self, db: AsyncSession, evidence_id: int
    ) -> TrainingEvidence | None:
        """Get evidence by ID.

        Args:
            db: Database session
            evidence_id: Evidence record ID

        Returns:
            Evidence record or None if not found
        """
        query = select(TrainingEvidence).where(TrainingEvidence.id == evidence_id)
        result = await db.execute(query)
        return result.scalar_one_or_none()

    async def list_evidence(
        self,
        db: AsyncSession,
        session_id: int | None = None,
        asset_id: int | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[TrainingEvidence], int]:
        """List evidence records with filtering and pagination.

        Args:
            db: Database session
            session_id: Optional session ID filter
            asset_id: Optional asset ID filter
            page: Page number (1-indexed)
            page_size: Number of items per page

        Returns:
            Tuple of (evidence list, total count)
        """
        # Build base query
        query = select(TrainingEvidence)

        # Apply filters
        if session_id is not None:
            query = query.where(TrainingEvidence.session_id == session_id)
        if asset_id is not None:
            query = query.where(TrainingEvidence.asset_id == asset_id)

        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar_one()

        # Add ordering and pagination
        query = query.order_by(TrainingEvidence.created_at.desc())
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)

        # Execute query
        result = await db.execute(query)
        evidence_list = list(result.scalars().all())

        return evidence_list, total

    async def get_asset_history(
        self, db: AsyncSession, asset_id: int
    ) -> list[TrainingEvidence]:
        """Get all training evidence for an asset (training history).

        Args:
            db: Database session
            asset_id: Image asset ID

        Returns:
            List of evidence records ordered by creation time
        """
        query = (
            select(TrainingEvidence)
            .where(TrainingEvidence.asset_id == asset_id)
            .order_by(TrainingEvidence.created_at.desc())
        )

        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_session_evidence_stats(
        self, db: AsyncSession, session_id: int
    ) -> dict[str, object]:
        """Get aggregate statistics for a session's evidence.

        Args:
            db: Database session
            session_id: Training session ID

        Returns:
            Dictionary with aggregate stats:
            - total: Total evidence records
            - successful: Records without errors
            - failed: Records with errors
            - avg_processing_time_ms: Average processing time
            - min_processing_time_ms: Minimum processing time
            - max_processing_time_ms: Maximum processing time
            - devices_used: List of unique devices
            - model_versions: List of unique model versions
        """
        # Get all evidence for session
        query = select(TrainingEvidence).where(TrainingEvidence.session_id == session_id)
        result = await db.execute(query)
        evidence_list = list(result.scalars().all())

        if not evidence_list:
            return {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "avg_processing_time_ms": 0.0,
                "min_processing_time_ms": 0,
                "max_processing_time_ms": 0,
                "devices_used": [],
                "model_versions": [],
            }

        # Calculate statistics
        successful = [e for e in evidence_list if e.error_message is None]
        failed = [e for e in evidence_list if e.error_message is not None]
        processing_times = [e.processing_time_ms for e in evidence_list]

        # Collect unique devices and model versions
        devices_used = sorted(set(e.device for e in evidence_list))
        model_versions = sorted(set(e.model_version for e in evidence_list))

        return {
            "total": len(evidence_list),
            "successful": len(successful),
            "failed": len(failed),
            "avg_processing_time_ms": sum(processing_times) / len(processing_times),
            "min_processing_time_ms": min(processing_times),
            "max_processing_time_ms": max(processing_times),
            "devices_used": devices_used,
            "model_versions": model_versions,
        }

    async def delete_evidence(self, db: AsyncSession, evidence_id: int) -> bool:
        """Delete evidence record.

        Args:
            db: Database session
            evidence_id: Evidence record ID

        Returns:
            True if deleted, False if not found
        """
        evidence = await self.get_evidence(db, evidence_id)
        if not evidence:
            return False

        await db.delete(evidence)
        await db.commit()

        logger.info(f"Deleted evidence {evidence_id}")
        return True

    async def delete_session_evidence(self, db: AsyncSession, session_id: int) -> int:
        """Delete all evidence for a session.

        Args:
            db: Database session
            session_id: Training session ID

        Returns:
            Number of evidence records deleted
        """
        query = select(TrainingEvidence).where(TrainingEvidence.session_id == session_id)
        result = await db.execute(query)
        evidence_list = list(result.scalars().all())

        count = len(evidence_list)
        for evidence in evidence_list:
            await db.delete(evidence)

        await db.commit()

        logger.info(f"Deleted {count} evidence records for session {session_id}")
        return count
