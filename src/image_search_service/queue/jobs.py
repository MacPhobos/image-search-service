"""Background job definitions."""

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from image_search_service.core.logging import get_logger
from image_search_service.db.models import ImageAsset
from image_search_service.db.session import get_sync_engine
from image_search_service.services.embedding import get_embedding_service
from image_search_service.vector.qdrant import ensure_collection, upsert_vector

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
        # Build payload with only string values
        payload: dict[str, str] = {"path": asset.path}
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


def process_image(image_path: str) -> dict[str, str]:
    """Process image and extract embeddings (deprecated).

    Args:
        image_path: Path to image file

    Returns:
        Result dictionary with processing status

    Note:
        This is deprecated in favor of index_asset which handles full workflow.
    """
    logger.info("Processing image: %s", image_path)

    return {
        "status": "success",
        "image_path": image_path,
        "message": "Use index_asset instead",
    }
