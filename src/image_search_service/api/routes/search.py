"""Search endpoints."""

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from qdrant_client import QdrantClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.api.schemas import (
    Asset,
    ComposeSearchResponse,
    ErrorResponse,
    HybridSearchResponse,
    HybridSearchResult,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from image_search_service.core.logging import get_logger
from image_search_service.db.models import ImageAsset
from image_search_service.db.session import get_db
from image_search_service.services.embedding_router import get_search_embedding_service
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

    # Get embedding service based on feature flags (CLIP or SigLIP)
    embedding_service, collection = get_search_embedding_service(user_id=None)

    # Embed the query
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
            collection_name=collection,  # Use selected collection (CLIP or SigLIP)
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


@router.post(
    "/image",
    response_model=SearchResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
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

        # Get embedding service based on feature flags (CLIP or SigLIP)
        embedding_service, collection = get_search_embedding_service(user_id=None)

        # Embed the image
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
                collection_name=collection,  # Use selected collection (CLIP or SigLIP)
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

        return SearchResponse(
            results=results, total=len(results), query=f"<image: {file.filename}>"
        )

    finally:
        # Cleanup temp file
        if temp_path and temp_path.exists():
            temp_path.unlink()


@router.get(
    "/similar/{asset_id}",
    response_model=SearchResponse,
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
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

    # Get embedding service and collection based on feature flags
    # Note: For similar search, we need to retrieve from the same collection
    # that was used for the original asset embedding

    _, collection = get_search_embedding_service(user_id=None)

    try:
        point = qdrant.retrieve(
            collection_name=collection,
            ids=[asset_id],
            with_vectors=True,
        )

        if not point or len(point) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "vector_not_found",
                    "message": (
                        f"No embedding found for asset {asset_id}. "
                        "Image may not be indexed yet."
                    ),
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
        if (
            isinstance(query_vector, list)
            and len(query_vector) > 0
            and isinstance(query_vector[0], list)
        ):
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
            collection_name=collection,  # Use selected collection (CLIP or SigLIP)
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

    return SearchResponse(
        results=results, total=len(results), query=f"<similar to asset {asset_id}>"
    )


@router.post(
    "/hybrid",
    response_model=HybridSearchResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def hybrid_search(
    file: UploadFile | None = File(None, description="Optional image file"),
    text_query: str | None = Form(None, alias="textQuery", description="Optional text query"),
    text_weight: float = Form(0.5, alias="textWeight", description="Weight for text (0.0-1.0)"),
    limit: int = Form(20, description="Maximum results"),
    offset: int = Form(0, description="Offset for pagination"),
    category_id: int | None = Form(None, alias="categoryId", description="Category filter"),
    db: AsyncSession = Depends(get_db),
    qdrant: QdrantClient = Depends(get_qdrant_client),
) -> HybridSearchResponse:
    """Hybrid search combining text and image queries using RRF fusion.

    Accepts both text and image inputs (at least one required). Uses Reciprocal Rank
    Fusion (RRF) to combine results from text and image search modalities.

    Args:
        file: Optional image file for image search
        text_query: Optional text query for semantic search
        text_weight: Weight for text vs image (0.5 = equal weight)
        limit: Maximum results to return
        offset: Pagination offset
        category_id: Optional category filter
        db: Database session
        qdrant: Qdrant client

    Returns:
        HybridSearchResponse with fused results showing scores from both modalities

    Raises:
        HTTPException: 400 if neither text nor image provided, 503 if Qdrant unavailable
    """
    # Validate at least one query type provided
    if not text_query and not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "missing_query",
                "message": "At least one of text_query or image file must be provided",
            },
        )

    # Validate file type if provided
    if file and (not file.content_type or not file.content_type.startswith("image/")):
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

    # Get embedding service
    embedding_service, collection = get_search_embedding_service(user_id=None)

    # Perform text search if query provided
    text_results: list[dict[str, str | int | float]] = []
    if text_query:
        try:
            text_vector = embedding_service.embed_text(text_query)
            text_filters: dict[str, str | int] = {}
            if category_id is not None:
                text_filters["category_id"] = category_id

            text_results = search_vectors(
                query_vector=text_vector,
                limit=limit * 2,  # Get more candidates for fusion
                offset=0,  # Always start at 0 for fusion
                filters=text_filters if text_filters else None,
                client=qdrant,
                collection_name=collection,
            )
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            # Continue with image-only search

    # Perform image search if file provided
    image_results: list[dict[str, str | int | float]] = []
    image_filename = None
    temp_path = None

    if file:
        try:
            # Save temp file
            suffix = Path(file.filename or "image.jpg").suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_path = Path(temp_file.name)
                content = await file.read()
                temp_file.write(content)
                image_filename = file.filename

            # Embed image
            image_vector = embedding_service.embed_image(temp_path)

            image_filters: dict[str, str | int] = {}
            if category_id is not None:
                image_filters["category_id"] = category_id

            image_results = search_vectors(
                query_vector=image_vector,
                limit=limit * 2,  # Get more candidates for fusion
                offset=0,
                filters=image_filters if image_filters else None,
                client=qdrant,
                collection_name=collection,
            )
        except Exception as e:
            logger.error(f"Image search failed: {e}")
            # Continue with text-only search
        finally:
            # Cleanup temp file
            if temp_path and temp_path.exists():
                temp_path.unlink()

    # If only one modality, return those results directly
    if not text_results and image_results:
        # Image-only
        results = await _convert_to_hybrid_results(image_results, db, image_only=True)
        return HybridSearchResponse(
            results=results[:limit],
            total=len(results[:limit]),
            textQuery=None,  # Use camelCase alias
            imageFilename=image_filename,
        )
    elif text_results and not image_results:
        # Text-only
        results = await _convert_to_hybrid_results(text_results, db, text_only=True)
        return HybridSearchResponse(
            results=results[:limit],
            total=len(results[:limit]),
            textQuery=text_query,  # Use camelCase alias
            imageFilename=None,
        )

    # Both modalities: use RRF fusion
    from image_search_service.services.fusion import RankedItem, weighted_reciprocal_rank_fusion

    # Convert to ranked items
    text_ranked = [
        RankedItem(
            item=hit["asset_id"],
            rank=idx + 1,
            score=hit["score"],
            source="text",
        )
        for idx, hit in enumerate(text_results)
    ]

    image_ranked = [
        RankedItem(
            item=hit["asset_id"],
            rank=idx + 1,
            score=hit["score"],
            source="image",
        )
        for idx, hit in enumerate(image_results)
    ]

    # Compute weights: text_weight for text, (1 - text_weight) for image
    source_weights = {
        "text": text_weight,
        "image": 1.0 - text_weight,
    }

    # Fuse results
    fused = weighted_reciprocal_rank_fusion(
        ranked_lists=[text_ranked, image_ranked],
        source_weights=source_weights,
        k=60,
    )

    # Convert to hybrid results with asset details
    results = []
    for fused_item in fused[offset : offset + limit]:
        asset_id = int(fused_item.item)

        result = await db.execute(select(ImageAsset).where(ImageAsset.id == asset_id))
        asset = result.scalar_one_or_none()

        if asset:
            results.append(
                HybridSearchResult(
                    asset=Asset.model_validate(asset),
                    textScore=fused_item.scores.get("text"),  # Use camelCase alias
                    imageScore=fused_item.scores.get("image"),
                    combinedScore=fused_item.rrf_score,
                    rank=fused_item.combined_rank,
                )
            )

    return HybridSearchResponse(
        results=results,
        total=len(results),
        textQuery=text_query,  # Use camelCase alias
        imageFilename=image_filename,
    )


@router.post(
    "/compose",
    response_model=ComposeSearchResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def compose_search(
    file: UploadFile = File(..., description="Reference image"),
    modifier_text: str = Form(..., alias="modifierText", description="Text modifier"),
    alpha: float = Form(0.3, description="Mixing weight (0.0=image, 1.0=text)"),
    limit: int = Form(20, description="Maximum results"),
    offset: int = Form(0, description="Offset for pagination"),
    category_id: int | None = Form(None, alias="categoryId", description="Category filter"),
    db: AsyncSession = Depends(get_db),
    qdrant: QdrantClient = Depends(get_qdrant_client),
) -> ComposeSearchResponse:
    """Composed image retrieval: search using image + text modification.

    Combines a reference image with a text modifier using vector arithmetic:
        query_vector = (1 - alpha) * image_vector + alpha * text_vector

    Example: Upload beach photo + "at sunset" → finds similar beach scenes at sunset

    Args:
        file: Reference image file
        modifier_text: Text describing desired modification (e.g., "at sunset", "in winter")
        alpha: Mixing weight (0.0 = pure image, 1.0 = pure text, 0.3 = subtle modification)
        limit: Maximum results
        offset: Pagination offset
        category_id: Optional category filter
        db: Database session
        qdrant: Qdrant client

    Returns:
        ComposeSearchResponse with results from composed query

    Raises:
        HTTPException: 400 if invalid file, 503 if Qdrant unavailable
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

    # Get embedding service
    embedding_service, collection = get_search_embedding_service(user_id=None)

    # Save temp file and embed image
    temp_path = None
    try:
        suffix = Path(file.filename or "image.jpg").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = Path(temp_file.name)
            content = await file.read()
            temp_file.write(content)

        # Embed both image and text
        try:
            image_vector = embedding_service.embed_image(temp_path)
            text_vector = embedding_service.embed_text(modifier_text)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "embedding_error",
                    "message": "Failed to process image or text. Ensure valid inputs.",
                },
            )

        # Compose vectors: (1 - alpha) * image + alpha * text
        import numpy as np

        image_arr = np.array(image_vector)
        text_arr = np.array(text_vector)
        composed_vector = ((1 - alpha) * image_arr + alpha * text_arr).tolist()

        # Search with composed vector
        search_filters: dict[str, str | int] = {}
        if category_id is not None:
            search_filters["category_id"] = category_id

        try:
            vector_results = search_vectors(
                query_vector=composed_vector,
                limit=limit,
                offset=offset,
                filters=search_filters if search_filters else None,
                client=qdrant,
                collection_name=collection,
            )
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return ComposeSearchResponse(
                results=[],
                total=0,
                referenceImage=file.filename or "unknown",  # Use camelCase alias
                modifierText=modifier_text,
                alpha=alpha,
            )

        # Fetch asset details
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

        return ComposeSearchResponse(
            results=results,
            total=len(results),
            referenceImage=file.filename or "unknown",  # Use camelCase alias
            modifierText=modifier_text,
            alpha=alpha,
        )

    finally:
        # Cleanup temp file
        if temp_path and temp_path.exists():
            temp_path.unlink()


async def _convert_to_hybrid_results(
    vector_results: list[dict[str, str | int | float]],
    db: AsyncSession,
    text_only: bool = False,
    image_only: bool = False,
) -> list[HybridSearchResult]:
    """Convert vector search results to HybridSearchResult with asset details.

    Args:
        vector_results: Raw results from Qdrant search
        db: Database session
        text_only: If True, only text_score is populated
        image_only: If True, only image_score is populated

    Returns:
        List of HybridSearchResult with asset details
    """
    results = []
    for idx, hit in enumerate(vector_results, start=1):
        asset_id = hit.get("asset_id")
        if not asset_id:
            continue

        result = await db.execute(select(ImageAsset).where(ImageAsset.id == int(asset_id)))
        asset = result.scalar_one_or_none()

        if asset:
            score = hit["score"]
            score_value = float(score) if isinstance(score, (int, float, str)) else 0.0

            results.append(
                HybridSearchResult(
                    asset=Asset.model_validate(asset),
                    textScore=score_value if text_only else None,  # Use camelCase alias
                    imageScore=score_value if image_only else None,
                    combinedScore=score_value,
                    rank=idx,
                )
            )

    return results
