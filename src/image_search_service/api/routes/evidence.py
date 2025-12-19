"""Evidence endpoints for training metadata and history."""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from image_search_service.api.schemas import PaginatedResponse
from image_search_service.api.training_schemas import (
    EvidenceStatsResponse,
    TrainingEvidenceDetailResponse,
    TrainingEvidenceResponse,
)
from image_search_service.core.logging import get_logger
from image_search_service.db.models import TrainingEvidence
from image_search_service.db.session import get_db
from image_search_service.services.evidence_service import EvidenceService

logger = get_logger(__name__)
router = APIRouter(tags=["evidence"])


@router.get("/evidence", response_model=PaginatedResponse[TrainingEvidenceResponse])
async def list_evidence(
    session_id: int | None = Query(None, description="Filter by session ID"),
    asset_id: int | None = Query(None, description="Filter by asset ID"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> PaginatedResponse[TrainingEvidenceResponse]:
    """List training evidence records.

    Can filter by session_id, asset_id, or both.

    Args:
        session_id: Optional session ID filter
        asset_id: Optional asset ID filter
        page: Page number (1-indexed)
        page_size: Number of items per page (max 100)
        db: Database session

    Returns:
        Paginated list of evidence records
    """
    service = EvidenceService()
    evidence_list, total = await service.list_evidence(
        db,
        session_id=session_id,
        asset_id=asset_id,
        page=page,
        page_size=page_size,
    )

    return PaginatedResponse(
        items=[TrainingEvidenceResponse.model_validate(e) for e in evidence_list],
        total=total,
        page=page,
        pageSize=page_size,
        hasMore=(page * page_size) < total,
    )


@router.get("/evidence/{evidence_id}", response_model=TrainingEvidenceResponse)
async def get_evidence(
    evidence_id: int,
    db: AsyncSession = Depends(get_db),
) -> TrainingEvidenceResponse:
    """Get specific evidence record.

    Args:
        evidence_id: Evidence record ID
        db: Database session

    Returns:
        Evidence record

    Raises:
        HTTPException: If evidence not found
    """
    service = EvidenceService()
    evidence = await service.get_evidence(db, evidence_id)

    if not evidence:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evidence {evidence_id} not found",
        )

    return TrainingEvidenceResponse.model_validate(evidence)


@router.get("/assets/{asset_id}/training-history", response_model=list[TrainingEvidenceResponse])
async def get_asset_training_history(
    asset_id: int,
    db: AsyncSession = Depends(get_db),
) -> list[TrainingEvidenceResponse]:
    """Get complete training history for an asset.

    Shows all training evidence records for this asset,
    ordered by creation time (most recent first).

    Args:
        asset_id: Image asset ID
        db: Database session

    Returns:
        List of evidence records for the asset
    """
    service = EvidenceService()
    history = await service.get_asset_history(db, asset_id)

    return [TrainingEvidenceResponse.model_validate(e) for e in history]


@router.get("/training/sessions/{session_id}/evidence/stats", response_model=EvidenceStatsResponse)
async def get_session_evidence_stats(
    session_id: int,
    db: AsyncSession = Depends(get_db),
) -> EvidenceStatsResponse:
    """Get aggregate statistics for session's training evidence.

    Provides summary statistics including:
    - Total, successful, and failed evidence records
    - Processing time statistics (min, max, average)
    - Devices and model versions used

    Args:
        session_id: Training session ID
        db: Database session

    Returns:
        Evidence statistics for the session
    """
    service = EvidenceService()
    stats = await service.get_session_evidence_stats(db, session_id)

    return EvidenceStatsResponse.model_validate(stats)


@router.delete("/evidence/{evidence_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_evidence(
    evidence_id: int,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a specific evidence record.

    Args:
        evidence_id: Evidence record ID
        db: Database session

    Raises:
        HTTPException: If evidence not found
    """
    service = EvidenceService()
    deleted = await service.delete_evidence(db, evidence_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evidence {evidence_id} not found",
        )


@router.get("/evidence/{evidence_id}/detail", response_model=TrainingEvidenceDetailResponse)
async def get_evidence_detail(
    evidence_id: int,
    db: AsyncSession = Depends(get_db),
) -> TrainingEvidenceDetailResponse:
    """Get evidence with asset details.

    Returns evidence record with asset path included.

    Args:
        evidence_id: Evidence record ID
        db: Database session

    Returns:
        Evidence record with asset information

    Raises:
        HTTPException: If evidence not found
    """
    # Get evidence with asset relationship loaded
    query = (
        select(TrainingEvidence)
        .where(TrainingEvidence.id == evidence_id)
        .options(selectinload(TrainingEvidence.asset))
    )
    result = await db.execute(query)
    evidence = result.scalar_one_or_none()

    if not evidence:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evidence {evidence_id} not found",
        )

    # Build response with asset path
    return TrainingEvidenceDetailResponse(
        id=evidence.id,
        assetId=evidence.asset_id,
        assetPath=evidence.asset.path,
        sessionId=evidence.session_id,
        modelName=evidence.model_name,
        modelVersion=evidence.model_version,
        embeddingChecksum=evidence.embedding_checksum,
        device=evidence.device,
        processingTimeMs=evidence.processing_time_ms,
        errorMessage=evidence.error_message,
        metadataJson=evidence.metadata_json,
        createdAt=evidence.created_at,
    )
