"""Search endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.api.schemas import (
    Asset,
    ErrorResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from image_search_service.core.logging import get_logger
from image_search_service.db.models import ImageAsset
from image_search_service.db.session import get_db
from image_search_service.services.embedding import get_embedding_service
from image_search_service.vector.qdrant import get_qdrant_client, search_vectors

logger = get_logger(__name__)
router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse, responses={503: {"model": ErrorResponse}})
async def search_assets(
    request: SearchRequest, db: AsyncSession = Depends(get_db)
) -> SearchResponse:
    """Semantic search for assets using text query.

    Args:
        request: Search request with query and filters
        db: Database session

    Returns:
        Search response with matching assets and scores

    Raises:
        HTTPException: If Qdrant is unavailable or embedding fails
    """
    # Check Qdrant connectivity
    try:
        client = get_qdrant_client()
        client.get_collections()
    except Exception as e:
        logger.error(f"Qdrant unavailable: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "service_unavailable",
                "message": "Vector database unavailable",
            },
        )

    # Embed the query
    embedding_service = get_embedding_service()
    try:
        query_vector = embedding_service.embed_text(request.query)
    except Exception as e:
        logger.error(f"Failed to embed query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "embedding_error", "message": str(e)},
        )

    # Search Qdrant
    try:
        vector_results = search_vectors(
            query_vector=query_vector,
            limit=request.limit,
            offset=request.offset,
            filters=request.filters,
        )
    except Exception as e:
        logger.error(f"Qdrant search failed: {e}")
        # Return empty results if collection doesn't exist yet
        return SearchResponse(results=[], total=0, query=request.query)

    # Fetch full asset records
    results = []
    for hit in vector_results:
        asset_id = hit.get("asset_id")
        if not asset_id:
            continue

        result = await db.execute(select(ImageAsset).where(ImageAsset.id == int(asset_id)))
        asset = result.scalar_one_or_none()

        if asset:
            # Ensure score is a float
            score = hit["score"]
            score_value = float(score) if isinstance(score, (int, float, str)) else 0.0

            results.append(
                SearchResult(
                    asset=Asset.model_validate(asset),
                    score=score_value,
                    highlights=[],  # Could extract from payload in future
                )
            )

    return SearchResponse(results=results, total=len(results), query=request.query)
