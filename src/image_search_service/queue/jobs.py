"""Background job definitions."""

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from image_search_service.core.logging import get_logger
from image_search_service.db.models import FaceInstance, ImageAsset
from image_search_service.db.session import get_sync_engine
from image_search_service.services.embedding import get_embedding_service
from image_search_service.vector.qdrant import (
    ensure_collection,
    update_vector_payload,
    upsert_vector,
)

logger = get_logger(__name__)


def index_asset(asset_id: str) -> dict[str, str]:
    """Index a single asset: generate embedding and upsert to Qdrant.

    Args:
        asset_id: ID string of the asset to index

    Returns:
        Dict with status and details
    """
    logger.info(f"Starting index job for asset {asset_id}")

    # Get embedding service and ensure collection exists
    embedding_service = get_embedding_service()
    ensure_collection(embedding_service.embedding_dim)

    # Load asset from database (sync for RQ worker)
    engine = get_sync_engine()
    with Session(engine) as session:
        asset = session.execute(
            select(ImageAsset).where(ImageAsset.id == int(asset_id))
        ).scalar_one_or_none()

        if not asset:
            logger.error(f"Asset {asset_id} not found")
            return {"status": "error", "message": "Asset not found"}

        # Generate embedding
        try:
            vector = embedding_service.embed_image(asset.path)
        except Exception as e:
            logger.error(f"Failed to embed image {asset.path}: {e}")
            return {"status": "error", "message": str(e)}

        # Upsert to Qdrant
        # Build payload with string values
        payload: dict[str, str | int] = {"path": asset.path}
        if asset.created_at:
            payload["created_at"] = asset.created_at.isoformat()

        upsert_vector(
            asset_id=asset.id,
            vector=vector,
            payload=payload,
        )

        # Update indexed_at timestamp
        asset.indexed_at = datetime.now(UTC)
        session.commit()

        logger.info(f"Successfully indexed asset {asset_id}")
        return {"status": "success", "asset_id": asset_id}


def update_asset_person_ids_job(asset_id: int) -> dict[str, str | int]:
    """Update Qdrant payload with current person_ids for an asset.

    This job is triggered when face assignments change (assign, unassign, or person merge).
    It queries the database for all person_ids associated with faces on the asset,
    then updates the Qdrant vector payload to keep it in sync.

    Args:
        asset_id: The asset ID to update

    Returns:
        Dict with status and details
    """
    logger.info(f"Starting person_ids update job for asset {asset_id}")

    engine = get_sync_engine()
    with Session(engine) as session:
        # Verify asset still exists before doing any work
        asset = session.execute(
            select(ImageAsset).where(ImageAsset.id == asset_id)
        ).scalar_one_or_none()

        if not asset:
            logger.info(
                f"Asset {asset_id} not found in database, skipping person_ids update"
            )
            return {
                "status": "skipped",
                "asset_id": str(asset_id),
                "message": "Asset not found in database",
            }

        # Get all distinct person_ids for faces on this asset
        # Only include non-NULL person_ids
        result = session.execute(
            select(FaceInstance.person_id)
            .where(FaceInstance.asset_id == asset_id)
            .where(FaceInstance.person_id.isnot(None))
            .distinct()
        )
        person_ids = [str(row[0]) for row in result.fetchall() if row[0]]

        # Update Qdrant payload
        # If the asset doesn't have a vector yet, this will fail silently
        # (which is expected - vectors are created during indexing)
        try:
            update_vector_payload(asset_id, {"person_ids": person_ids})
            logger.info(
                f"Successfully updated person_ids for asset {asset_id}: {len(person_ids)} persons"
            )
            return {
                "status": "success",
                "asset_id": str(asset_id),
                "person_count": len(person_ids),
            }
        except Exception as e:
            # Log but don't fail - the asset may not have a vector yet
            logger.warning(
                f"Could not update person_ids for asset {asset_id} (may not be indexed yet): {e}"
            )
            return {
                "status": "skipped",
                "asset_id": str(asset_id),
                "message": "Asset may not be indexed yet",
            }


