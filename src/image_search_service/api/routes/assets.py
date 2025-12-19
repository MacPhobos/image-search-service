"""Asset management endpoints."""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from rq import Retry
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.api.schemas import (
    Asset,
    IngestRequest,
    IngestResponse,
    PaginatedResponse,
)
from image_search_service.core.logging import get_logger
from image_search_service.db.models import ImageAsset
from image_search_service.db.session import get_db
from image_search_service.queue.worker import get_queue

logger = get_logger(__name__)
router = APIRouter(prefix="/assets", tags=["assets"])


def scan_directory(root_path: Path, recursive: bool, extensions: list[str]) -> list[Path]:
    """Scan directory for image files.

    Args:
        root_path: Root directory to scan
        recursive: Whether to scan subdirectories
        extensions: List of file extensions to include

    Returns:
        List of discovered image file paths
    """
    pattern = "**/*" if recursive else "*"
    files: list[Path] = []
    for ext in extensions:
        files.extend(root_path.glob(f"{pattern}.{ext}"))
        files.extend(root_path.glob(f"{pattern}.{ext.upper()}"))
    return sorted(set(files))


@router.post("/ingest", response_model=IngestResponse)
async def ingest_assets(
    request: IngestRequest, db: AsyncSession = Depends(get_db)
) -> IngestResponse:
    """Scan filesystem and ingest images.

    Args:
        request: Ingest request with path and options
        db: Database session

    Returns:
        Ingest response with counts

    Raises:
        HTTPException: If path doesn't exist or is not a directory
    """
    root = Path(request.root_path)

    if not root.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Path does not exist: {request.root_path}",
        )

    if not root.is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Path is not a directory: {request.root_path}",
        )

    # Scan for images
    discovered_files = scan_directory(root, request.recursive, request.extensions)
    discovered = len(discovered_files)

    if request.dry_run:
        return IngestResponse(discovered=discovered, enqueued=0, skipped=0)

    # Upsert assets and track new ones
    enqueued = 0
    skipped = 0
    queue = get_queue()

    for file_path in discovered_files:
        path_str = str(file_path.resolve())

        # Check if already exists
        existing = await db.execute(select(ImageAsset).where(ImageAsset.path == path_str))
        existing_asset = existing.scalar_one_or_none()

        if existing_asset:
            if existing_asset.indexed_at is not None:
                skipped += 1
                continue
            # Re-queue unindexed asset
            asset_id = existing_asset.id
        else:
            # Create new asset
            asset = ImageAsset(path=path_str)
            db.add(asset)
            await db.flush()
            asset_id = asset.id

        # Enqueue for indexing with retry on failure
        queue.enqueue(
            "image_search_service.queue.jobs.index_asset",
            str(asset_id),
            retry=Retry(max=3),
        )
        enqueued += 1

    await db.commit()

    return IngestResponse(discovered=discovered, enqueued=enqueued, skipped=skipped)


@router.get("", response_model=PaginatedResponse[Asset])
async def list_assets(
    page: int = 1,
    page_size: int = 50,
    from_date: str | None = None,
    to_date: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> PaginatedResponse[Asset]:
    """List assets with pagination and optional date filters.

    Args:
        page: Page number (1-indexed)
        page_size: Number of items per page
        from_date: Optional start date filter (ISO format)
        to_date: Optional end date filter (ISO format)
        db: Database session

    Returns:
        Paginated response with assets
    """
    query = select(ImageAsset)
    count_query = select(func.count(ImageAsset.id))

    # Apply date filters
    if from_date:
        query = query.where(ImageAsset.created_at >= from_date)
        count_query = count_query.where(ImageAsset.created_at >= from_date)
    if to_date:
        query = query.where(ImageAsset.created_at <= to_date)
        count_query = count_query.where(ImageAsset.created_at <= to_date)

    # Get total count
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Apply pagination
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size).order_by(ImageAsset.created_at.desc())

    result = await db.execute(query)
    assets = result.scalars().all()

    return PaginatedResponse(
        items=[Asset.model_validate(a) for a in assets],
        total=total,
        page=page,
        pageSize=page_size,
        hasMore=(offset + len(assets)) < total,
    )


@router.get("/{asset_id}", response_model=Asset)
async def get_asset(asset_id: int, db: AsyncSession = Depends(get_db)) -> Asset:
    """Get a single asset by ID.

    Args:
        asset_id: Asset ID
        db: Database session

    Returns:
        Asset details

    Raises:
        HTTPException: If asset not found
    """
    result = await db.execute(select(ImageAsset).where(ImageAsset.id == asset_id))
    asset = result.scalar_one_or_none()

    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Asset {asset_id} not found"
        )

    return Asset.model_validate(asset)
