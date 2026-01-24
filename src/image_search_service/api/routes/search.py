"""Search endpoints."""

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from qdrant_client import QdrantClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.api.schemas import (
    Asset,
    ErrorResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SimilarSearchRequest,
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
    request: SearchRequest,
    db: AsyncSession = Depends(get_db),
    qdrant: QdrantClient = Depends(get_qdrant_client),
) -> SearchResponse:
    """Semantic search for assets using text query.

    Args:
        request: Search request with query and filters
        db: Database session
        qdrant: Qdrant client for vector search

    Returns:
        Search response with matching assets and scores

    Raises:
        HTTPException: If Qdrant is unavailable or embedding fails
    """
    # Check Qdrant connectivity
    try:
        qdrant.get_collections()
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
        # Build filters dict from request
        search_filters: dict[str, str | int] = {}
        if request.filters:
            search_filters.update(request.filters)
        if request.category_id is not None:
            search_filters["category_id"] = request.category_id

        vector_results = search_vectors(
            query_vector=query_vector,
            limit=request.limit,
            offset=request.offset,
            filters=search_filters if search_filters else None,
            client=qdrant,
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


@router.post("/image", response_model=SearchResponse, responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
async def search_by_image(
    file: UploadFile = File(..., description="Image file to search with"),
    limit: int = Form(50, description="Maximum number of results"),
    offset: int = Form(0, description="Offset for pagination"),
    category_id: int | None = Form(None, alias="categoryId", description="Filter by category ID"),
    db: AsyncSession = Depends(get_db),
    qdrant: QdrantClient = Depends(get_qdrant_client),
) -> SearchResponse:
    """Search for similar images using an uploaded image.

    Args:
        file: Image file to search with
        limit: Maximum number of results (default: 50)
        offset: Offset for pagination (default: 0)
        category_id: Optional category filter
        db: Database session
        qdrant: Qdrant client for vector search

    Returns:
        Search response with matching assets and scores

    Raises:
        HTTPException: 400 if file is not a valid image, 503 if Qdrant unavailable
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_file_type",
                "message": f"File must be an image. Got: {file.content_type}",
            },
        )

    # Check Qdrant connectivity
    try:
        qdrant.get_collections()
    except Exception as e:
        logger.error(f"Qdrant unavailable: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "service_unavailable",
                "message": "Vector database unavailable",
            },
        )

    # Save uploaded file to temporary location
    temp_path = None
    try:
        # Create temp file with proper extension
        suffix = Path(file.filename or "image.jpg").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = Path(temp_file.name)
            # Read and write file content
            content = await file.read()
            temp_file.write(content)

        # Embed the image
        embedding_service = get_embedding_service()
        try:
            query_vector = embedding_service.embed_image(temp_path)
        except Exception as e:
            logger.error(f"Failed to embed image: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "embedding_error",
                    "message": "Failed to process image. Ensure it is a valid image file.",
                },
            )

        # Build filters
        search_filters: dict[str, str | int] = {}
        if category_id is not None:
            search_filters["category_id"] = category_id

        # Search Qdrant
        try:
            vector_results = search_vectors(
                query_vector=query_vector,
                limit=limit,
                offset=offset,
                filters=search_filters if search_filters else None,
                client=qdrant,
            )
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            # Return empty results if collection doesn't exist yet
            return SearchResponse(results=[], total=0, query=f"<image: {file.filename}>")

        # Fetch full asset records
        results = []
        for hit in vector_results:
            asset_id = hit.get("asset_id")
            if not asset_id:
                continue

            result = await db.execute(select(ImageAsset).where(ImageAsset.id == int(asset_id)))
            asset = result.scalar_one_or_none()

            if asset:
                score = hit["score"]
                score_value = float(score) if isinstance(score, (int, float, str)) else 0.0

                results.append(
                    SearchResult(
                        asset=Asset.model_validate(asset),
                        score=score_value,
                        highlights=[],
                    )
                )

        return SearchResponse(results=results, total=len(results), query=f"<image: {file.filename}>")

    finally:
        # Cleanup temp file
        if temp_path and temp_path.exists():
            temp_path.unlink()


@router.get("/similar/{asset_id}", response_model=SearchResponse, responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
async def search_similar(
    asset_id: int,
    limit: int = 50,
    offset: int = 0,
    exclude_self: bool = True,
    category_id: int | None = None,
    db: AsyncSession = Depends(get_db),
    qdrant: QdrantClient = Depends(get_qdrant_client),
) -> SearchResponse:
    """Find similar images to a given asset by ID.

    Args:
        asset_id: ID of the source asset
        limit: Maximum number of results (default: 50)
        offset: Offset for pagination (default: 0)
        exclude_self: Whether to exclude the source image from results (default: True)
        category_id: Optional category filter
        db: Database session
        qdrant: Qdrant client for vector search

    Returns:
        Search response with similar assets and scores

    Raises:
        HTTPException: 404 if asset not found, 503 if Qdrant unavailable
    """
    # Check if asset exists in database
    result = await db.execute(select(ImageAsset).where(ImageAsset.id == asset_id))
    source_asset = result.scalar_one_or_none()

    if not source_asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "not_found",
                "message": f"Asset {asset_id} not found",
            },
        )

    # Check Qdrant connectivity
    try:
        qdrant.get_collections()
    except Exception as e:
        logger.error(f"Qdrant unavailable: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "service_unavailable",
                "message": "Vector database unavailable",
            },
        )

    # Retrieve the existing embedding from Qdrant
    from image_search_service.core.config import get_settings

    settings = get_settings()
    try:
        point = qdrant.retrieve(
            collection_name=settings.qdrant_collection,
            ids=[asset_id],
            with_vectors=True,
        )

        if not point or len(point) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "vector_not_found",
                    "message": f"No embedding found for asset {asset_id}. Image may not be indexed yet.",
                },
            )

        query_vector = point[0].vector
        # Validate vector type - must be a simple list[float], not dict or nested list
        if query_vector is None or isinstance(query_vector, dict):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "invalid_vector",
                    "message": "Retrieved vector is invalid",
                },
            )
        # Handle nested list case (shouldn't happen for our collection, but type-check requires it)
        if isinstance(query_vector, list) and len(query_vector) > 0 and isinstance(query_vector[0], list):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "invalid_vector",
                    "message": "Retrieved vector has unexpected nested structure",
                },
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve vector for asset {asset_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "retrieval_error",
                "message": "Failed to retrieve embedding",
            },
        )

    # Build filters
    search_filters: dict[str, str | int] = {}
    if category_id is not None:
        search_filters["category_id"] = category_id

    # At this point, we've validated that query_vector is list[float]
    # Cast to help type checker (we've already validated above)
    query_vector_validated: list[float] = query_vector  # type: ignore[assignment]

    # Search for similar vectors
    try:
        vector_results = search_vectors(
            query_vector=query_vector_validated,
            limit=limit + (1 if exclude_self else 0),  # Get extra if excluding self
            offset=offset,
            filters=search_filters if search_filters else None,
            client=qdrant,
        )
    except Exception as e:
        logger.error(f"Qdrant search failed: {e}")
        return SearchResponse(results=[], total=0, query=f"<similar to asset {asset_id}>")

    # Fetch full asset records and filter out source if requested
    results = []
    for hit in vector_results:
        hit_asset_id = hit.get("asset_id")
        if not hit_asset_id:
            continue

        hit_asset_id_int = int(hit_asset_id)

        # Skip source asset if exclude_self is True
        if exclude_self and hit_asset_id_int == asset_id:
            continue

        result = await db.execute(select(ImageAsset).where(ImageAsset.id == hit_asset_id_int))
        asset = result.scalar_one_or_none()

        if asset:
            score = hit["score"]
            score_value = float(score) if isinstance(score, (int, float, str)) else 0.0

            results.append(
                SearchResult(
                    asset=Asset.model_validate(asset),
                    score=score_value,
                    highlights=[],
                )
            )

        # Stop if we have enough results
        if len(results) >= limit:
            break

    return SearchResponse(results=results, total=len(results), query=f"<similar to asset {asset_id}>")
