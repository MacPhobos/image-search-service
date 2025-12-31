"""Image serving endpoints for thumbnails and full images."""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.api.schemas import (
    BatchThumbnailRequest,
    BatchThumbnailResponse,
)
from image_search_service.core.config import get_settings
from image_search_service.core.logging import get_logger
from image_search_service.db.models import ImageAsset
from image_search_service.db.session import get_db
from image_search_service.services.thumbnail_service import ThumbnailService

logger = get_logger(__name__)
router = APIRouter(prefix="/images", tags=["images"])


def _get_thumbnail_service() -> ThumbnailService:
    """Get thumbnail service instance with configured settings."""
    settings = get_settings()
    return ThumbnailService(
        thumbnail_dir=settings.thumbnail_dir, thumbnail_size=settings.thumbnail_size
    )


def _validate_path_security(file_path: Path, allowed_dirs: list[Path]) -> None:
    """Validate that file path is within allowed directories.

    Prevents directory traversal attacks.

    Args:
        file_path: Path to validate
        allowed_dirs: List of allowed root directories

    Raises:
        HTTPException: If path is outside allowed directories
    """
    # Resolve to absolute path to prevent traversal
    try:
        abs_path = file_path.resolve()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file path: {e}",
        )

    # Check if path is within any allowed directory
    for allowed_dir in allowed_dirs:
        try:
            abs_path.relative_to(allowed_dir.resolve())
            return  # Path is safe
        except ValueError:
            continue  # Try next allowed dir

    # Path is not within any allowed directory
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Access to file path not allowed",
    )


@router.get("/{asset_id}/thumbnail")
async def get_thumbnail(asset_id: int, db: AsyncSession = Depends(get_db)) -> FileResponse:
    """Serve thumbnail for an asset.

    Generates thumbnail on-the-fly if it doesn't exist.

    Args:
        asset_id: Image asset ID
        db: Database session

    Returns:
        Thumbnail image file

    Raises:
        HTTPException: If asset not found or image file missing
    """
    # Get asset from database
    query = select(ImageAsset).where(ImageAsset.id == asset_id)
    result = await db.execute(query)
    asset = result.scalar_one_or_none()

    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Asset {asset_id} not found",
        )

    # Get thumbnail service
    thumb_service = _get_thumbnail_service()
    settings = get_settings()

    # Check if thumbnail exists in database
    if asset.thumbnail_path and Path(asset.thumbnail_path).exists():
        thumb_path = Path(asset.thumbnail_path)
    else:
        # Generate thumbnail on-the-fly
        try:
            logger.info(f"Generating on-demand thumbnail for asset {asset_id}")
            thumb_path_str, width, height = thumb_service.generate_thumbnail(asset.path, asset.id)
            thumb_path = Path(thumb_path_str)

            # Update asset with thumbnail info
            asset.thumbnail_path = thumb_path_str
            asset.width = width
            asset.height = height
            await db.commit()

        except FileNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Original image not found: {asset.path}",
            )
        except Exception as e:
            logger.error(f"Failed to generate thumbnail for asset {asset_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate thumbnail: {str(e)}",
            )

    # Validate path security
    allowed_dirs = [Path(settings.thumbnail_dir)]
    _validate_path_security(thumb_path, allowed_dirs)

    # Serve thumbnail
    return FileResponse(
        path=thumb_path,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "public, max-age=86400",  # Cache for 24 hours
            "ETag": f'"{asset_id}"',
        },
    )


@router.get("/{asset_id}/full")
async def get_full_image(asset_id: int, db: AsyncSession = Depends(get_db)) -> FileResponse:
    """Serve full-size original image.

    Args:
        asset_id: Image asset ID
        db: Database session

    Returns:
        Original image file

    Raises:
        HTTPException: If asset or file not found
    """
    # Get asset from database
    query = select(ImageAsset).where(ImageAsset.id == asset_id)
    result = await db.execute(query)
    asset = result.scalar_one_or_none()

    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Asset {asset_id} not found",
        )

    # Validate file exists
    file_path = Path(asset.path)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image file not found: {asset.path}",
        )

    # Validate path security
    settings = get_settings()
    allowed_dirs = []
    if settings.image_root_dir:
        allowed_dirs.append(Path(settings.image_root_dir))

    if not allowed_dirs:
        # If no image_root_dir configured, only allow absolute paths
        # This is a fallback - normally image_root_dir should be set
        logger.warning("IMAGE_ROOT_DIR not configured, serving from absolute path")
        allowed_dirs.append(file_path.parent)

    _validate_path_security(file_path, allowed_dirs)

    # Get MIME type
    mime_type = ThumbnailService.get_mime_type(asset.path)

    # Serve full image
    return FileResponse(
        path=file_path,
        media_type=mime_type,
        headers={
            "Cache-Control": "public, max-age=86400",  # Cache for 24 hours
            "ETag": f'"{asset_id}"',
        },
    )


@router.post("/thumbnails/batch", response_model=BatchThumbnailResponse)
async def get_thumbnails_batch(
    request: BatchThumbnailRequest,
    db: AsyncSession = Depends(get_db),
) -> BatchThumbnailResponse:
    """Fetch multiple thumbnails as base64 data URIs.

    Efficiently retrieves multiple thumbnails in a single request.
    Generates thumbnails on-demand if they don't exist.

    Args:
        request: Batch thumbnail request with asset IDs
        db: Database session

    Returns:
        Batch thumbnail response with data URIs
    """
    thumb_service = _get_thumbnail_service()

    # Fetch all assets in one query
    query = select(ImageAsset).where(ImageAsset.id.in_(request.asset_ids))
    result = await db.execute(query)
    assets = {asset.id: asset for asset in result.scalars().all()}

    thumbnails: dict[str, str | None] = {}
    not_found: list[int] = []

    for asset_id in request.asset_ids:
        asset = assets.get(asset_id)
        if not asset:
            not_found.append(asset_id)
            thumbnails[str(asset_id)] = None
            continue

        data_uri = thumb_service.get_thumbnail_as_base64(asset_id, asset.path)
        thumbnails[str(asset_id)] = data_uri

        if data_uri is None:
            not_found.append(asset_id)

    found_count = len(request.asset_ids) - len(not_found)

    logger.debug(
        f"Batch thumbnail request: {len(request.asset_ids)} requested, "
        f"{found_count} found, {len(not_found)} not found"
    )

    return BatchThumbnailResponse(
        thumbnails=thumbnails,
        found=found_count,
        notFound=not_found,
    )


@router.post("/{asset_id}/thumbnail/generate")
async def generate_thumbnail(
    asset_id: int, db: AsyncSession = Depends(get_db)
) -> dict[str, object]:
    """Force regenerate thumbnail for an asset.

    Useful for updating thumbnails after image changes.

    Args:
        asset_id: Image asset ID
        db: Database session

    Returns:
        Thumbnail information

    Raises:
        HTTPException: If asset not found or generation fails
    """
    # Get asset from database
    query = select(ImageAsset).where(ImageAsset.id == asset_id)
    result = await db.execute(query)
    asset = result.scalar_one_or_none()

    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Asset {asset_id} not found",
        )

    # Generate thumbnail
    thumb_service = _get_thumbnail_service()
    try:
        thumb_path, width, height = thumb_service.generate_thumbnail(asset.path, asset.id)

        # Update asset
        asset.thumbnail_path = thumb_path
        asset.width = width
        asset.height = height
        await db.commit()

        logger.info(f"Regenerated thumbnail for asset {asset_id}")

        return {
            "assetId": asset.id,
            "thumbnailPath": thumb_path,
            "width": width,
            "height": height,
            "status": "generated",
        }

    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Original image not found: {asset.path}",
        )
    except Exception as e:
        logger.error(f"Failed to regenerate thumbnail for asset {asset_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate thumbnail: {str(e)}",
        )
