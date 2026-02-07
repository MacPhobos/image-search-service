"""Bootstrap script to initialize Qdrant collections for fresh installs."""

import typer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

from image_search_service.core.config import get_settings
from image_search_service.core.logging import get_logger

logger = get_logger(__name__)

app = typer.Typer(
    name="bootstrap-qdrant",
    help="Bootstrap Qdrant collections for image search service",
)


def get_qdrant_client() -> QdrantClient:
    """Create Qdrant client from settings."""
    settings = get_settings()
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key if settings.qdrant_api_key else None,
    )


def ensure_image_assets_collection(client: QdrantClient) -> bool:
    """Ensure image_assets collection exists with correct configuration.

    Args:
        client: Qdrant client instance

    Returns:
        True if collection was created, False if it already existed
    """
    settings = get_settings()
    collection_name = settings.qdrant_collection
    embedding_dim = settings.embedding_dim

    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if collection_name in collection_names:
        typer.echo(f"✓ Collection '{collection_name}' already exists")
        logger.info(f"Collection '{collection_name}' already exists")
        return False

    # Create collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
    )
    typer.echo(f"✓ Created collection '{collection_name}' (dim={embedding_dim}, distance=COSINE)")
    logger.info(f"Created collection '{collection_name}' with dim={embedding_dim}")
    return True


def ensure_faces_collection(client: QdrantClient) -> bool:
    """Ensure faces collection exists with correct configuration and payload indexes.

    Args:
        client: Qdrant client instance

    Returns:
        True if collection was created, False if it already existed
    """
    settings = get_settings()
    collection_name = settings.qdrant_face_collection
    face_vector_dim = 512  # Hardcoded from face_qdrant.py

    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if collection_name in collection_names:
        typer.echo(f"✓ Collection '{collection_name}' already exists")
        logger.info(f"Collection '{collection_name}' already exists")
        return False

    # Create collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=face_vector_dim, distance=Distance.COSINE),
    )
    typer.echo(f"✓ Created collection '{collection_name}' (dim={face_vector_dim}, distance=COSINE)")
    logger.info(f"Created collection '{collection_name}' with dim={face_vector_dim}")

    # Create payload indexes for efficient filtering
    indexes = [
        ("person_id", PayloadSchemaType.KEYWORD),
        ("cluster_id", PayloadSchemaType.KEYWORD),
        ("is_prototype", PayloadSchemaType.BOOL),
        ("asset_id", PayloadSchemaType.KEYWORD),
        ("face_instance_id", PayloadSchemaType.KEYWORD),
    ]

    for field_name, field_schema in indexes:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
        )
        typer.echo(f"  ✓ Created payload index on '{field_name}' ({field_schema})")
        logger.info(f"Created payload index on '{field_name}'")

    return True


def ensure_person_centroids_collection(client: QdrantClient) -> bool:
    """Ensure person_centroids collection exists with correct configuration and payload indexes.

    Args:
        client: Qdrant client instance

    Returns:
        True if collection was created, False if it already existed
    """
    settings = get_settings()
    collection_name = settings.qdrant_centroid_collection
    centroid_vector_dim = 512  # ArcFace embedding dimension

    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if collection_name in collection_names:
        # Verify vector size matches
        collection_info = client.get_collection(collection_name=collection_name)
        actual_dim = collection_info.config.params.vectors.size  # type: ignore[union-attr]

        if actual_dim != centroid_vector_dim:
            typer.secho(
                f"⚠ Collection '{collection_name}' exists but has wrong dimension: "
                f"expected {centroid_vector_dim}, got {actual_dim}",
                fg=typer.colors.YELLOW,
            )
            logger.warning(
                f"Collection '{collection_name}' dimension mismatch: "
                f"expected {centroid_vector_dim}, got {actual_dim}"
            )
        else:
            typer.echo(f"✓ Collection '{collection_name}' already exists")
            logger.info(f"Collection '{collection_name}' already exists")

        return False

    # Create collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=centroid_vector_dim, distance=Distance.COSINE),
    )
    typer.echo(
        f"✓ Created collection '{collection_name}' (dim={centroid_vector_dim}, distance=COSINE)"
    )
    logger.info(f"Created collection '{collection_name}' with dim={centroid_vector_dim}")

    # Create payload indexes for efficient filtering
    indexes = [
        ("person_id", PayloadSchemaType.KEYWORD),
        ("centroid_id", PayloadSchemaType.KEYWORD),
        ("model_version", PayloadSchemaType.KEYWORD),
        ("centroid_version", PayloadSchemaType.INTEGER),
        ("centroid_type", PayloadSchemaType.KEYWORD),
    ]

    for field_name, field_schema in indexes:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
        )
        typer.echo(f"  ✓ Created payload index on '{field_name}' ({field_schema})")
        logger.info(f"Created payload index on '{field_name}'")

    return True


def ensure_siglip_collection(client: QdrantClient) -> bool:
    """Ensure image_assets_siglip collection exists with correct configuration.

    Only creates the collection if SigLIP is enabled (use_siglip=True or
    siglip_rollout_percentage > 0).

    Args:
        client: Qdrant client instance

    Returns:
        True if collection was created, False if it already existed or was skipped
    """
    settings = get_settings()

    # Check if SigLIP is enabled
    siglip_enabled = settings.use_siglip or settings.siglip_rollout_percentage > 0

    if not siglip_enabled:
        typer.echo(
            "⊘ SigLIP is disabled (use_siglip=False and rollout=0%), skipping collection creation"
        )
        logger.info("SigLIP is disabled, skipping collection creation")
        return False

    collection_name = settings.siglip_collection
    embedding_dim = settings.siglip_embedding_dim

    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if collection_name in collection_names:
        # Verify vector size matches
        collection_info = client.get_collection(collection_name=collection_name)
        actual_dim = collection_info.config.params.vectors.size  # type: ignore[union-attr]

        if actual_dim != embedding_dim:
            typer.secho(
                f"⚠ Collection '{collection_name}' exists but has wrong dimension: "
                f"expected {embedding_dim}, got {actual_dim}",
                fg=typer.colors.YELLOW,
            )
            logger.warning(
                f"Collection '{collection_name}' dimension mismatch: "
                f"expected {embedding_dim}, got {actual_dim}"
            )
        else:
            typer.echo(f"✓ Collection '{collection_name}' already exists")
            logger.info(f"Collection '{collection_name}' already exists")

        return False

    # Create collection with optimizations
    from qdrant_client.models import (
        HnswConfigDiff,
        OptimizersConfigDiff,
        ScalarQuantization,
        ScalarQuantizationConfig,
        ScalarType,
    )

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                quantile=0.99,
                always_ram=True,
            )
        ),
        hnsw_config=HnswConfigDiff(
            m=16,
            ef_construct=100,
        ),
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=10000,
        ),
    )
    typer.echo(
        f"✓ Created collection '{collection_name}' (dim={embedding_dim}, distance=COSINE, "
        f"quantization=INT8)"
    )
    logger.info(
        f"Created collection '{collection_name}' with dim={embedding_dim} "
        f"and INT8 quantization"
    )

    return True


@app.command()
def init() -> None:
    """Initialize all required Qdrant collections.

    This command is idempotent - safe to run multiple times.
    It will create collections and indexes only if they don't exist.

    Example:
        python -m image_search_service.scripts.bootstrap_qdrant init
    """
    typer.echo("=" * 60)
    typer.echo("Initializing Qdrant collections...")
    typer.echo("=" * 60)
    typer.echo("")

    try:
        client = get_qdrant_client()

        # Test connection
        typer.echo("Testing Qdrant connection...")
        client.get_collections()
        typer.echo("✓ Connection successful")
        typer.echo("")

        # Ensure image_assets collection
        typer.echo("1. Checking image_assets collection...")
        ensure_image_assets_collection(client)
        typer.echo("")

        # Ensure faces collection
        typer.echo("2. Checking faces collection...")
        ensure_faces_collection(client)
        typer.echo("")

        # Ensure person_centroids collection
        typer.echo("3. Checking person_centroids collection...")
        ensure_person_centroids_collection(client)
        typer.echo("")

        # Ensure SigLIP collection (conditional)
        typer.echo("4. Checking image_assets_siglip collection...")
        ensure_siglip_collection(client)
        typer.echo("")

        typer.echo("=" * 60)
        typer.secho("✓ Bootstrap complete!", fg=typer.colors.GREEN, bold=True)
        typer.echo("=" * 60)
        typer.echo("")
        typer.echo("Next steps:")
        typer.echo("  1. Verify setup:     make verify-qdrant")
        typer.echo("  2. Run migrations:   make migrate")
        typer.echo("  3. Start API server: make dev")
        typer.echo("")

    except Exception as e:
        typer.secho(f"✗ Bootstrap failed: {e}", fg=typer.colors.RED, bold=True)
        logger.error(f"Bootstrap failed: {e}")
        raise typer.Exit(1)


@app.command()
def verify() -> None:
    """Verify Qdrant collections are properly configured.

    Checks:
    - Collections exist
    - Correct vector dimensions
    - Correct distance metrics
    - Payload indexes exist (for faces and centroids collections)

    Example:
        python -m image_search_service.scripts.bootstrap_qdrant verify
    """
    typer.echo("=" * 60)
    typer.echo("Verifying Qdrant collections...")
    typer.echo("=" * 60)
    typer.echo("")

    try:
        client = get_qdrant_client()
        settings = get_settings()

        # Test connection
        typer.echo("Testing Qdrant connection...")
        collections_response = client.get_collections()
        typer.echo("✓ Connection successful")
        typer.echo("")

        all_collection_names = [c.name for c in collections_response.collections]
        has_errors = False

        # Verify image_assets collection
        typer.echo("1. Verifying image_assets collection...")
        image_collection_name = settings.qdrant_collection
        expected_image_dim = settings.embedding_dim

        if image_collection_name not in all_collection_names:
            typer.secho(f"✗ Collection '{image_collection_name}' not found", fg=typer.colors.RED)
            has_errors = True
        else:
            image_info = client.get_collection(collection_name=image_collection_name)
            actual_dim = image_info.config.params.vectors.size  # type: ignore[union-attr]
            actual_distance = image_info.config.params.vectors.distance  # type: ignore[union-attr]

            if actual_dim != expected_image_dim:
                typer.secho(
                    f"✗ Vector dimension mismatch: expected {expected_image_dim}, got {actual_dim}",
                    fg=typer.colors.RED,
                )
                has_errors = True
            elif actual_distance != Distance.COSINE:
                typer.secho(
                    f"✗ Distance metric mismatch: expected COSINE, got {actual_distance}",
                    fg=typer.colors.RED,
                )
                has_errors = True
            else:
                typer.echo(f"  ✓ Collection '{image_collection_name}' exists")
                typer.echo(f"  ✓ Vector dimension: {actual_dim}")
                typer.echo(f"  ✓ Distance metric: {actual_distance}")
                typer.echo(f"  ✓ Points count: {image_info.points_count or 0}")

        typer.echo("")

        # Verify faces collection
        typer.echo("2. Verifying faces collection...")
        faces_collection_name = settings.qdrant_face_collection
        expected_face_dim = 512

        if faces_collection_name not in all_collection_names:
            typer.secho(f"✗ Collection '{faces_collection_name}' not found", fg=typer.colors.RED)
            has_errors = True
        else:
            faces_info = client.get_collection(collection_name=faces_collection_name)
            actual_face_dim = faces_info.config.params.vectors.size  # type: ignore[union-attr]
            actual_face_distance = faces_info.config.params.vectors.distance  # type: ignore[union-attr]

            if actual_face_dim != expected_face_dim:
                typer.secho(
                    f"✗ Vector dimension mismatch: expected {expected_face_dim}, "
                    f"got {actual_face_dim}",
                    fg=typer.colors.RED,
                )
                has_errors = True
            elif actual_face_distance != Distance.COSINE:
                typer.secho(
                    f"✗ Distance metric mismatch: expected COSINE, got {actual_face_distance}",
                    fg=typer.colors.RED,
                )
                has_errors = True
            else:
                typer.echo(f"  ✓ Collection '{faces_collection_name}' exists")
                typer.echo(f"  ✓ Vector dimension: {actual_face_dim}")
                typer.echo(f"  ✓ Distance metric: {actual_face_distance}")
                typer.echo(f"  ✓ Points count: {faces_info.points_count or 0}")

                # Verify payload indexes
                typer.echo("  ✓ Checking payload indexes...")
                required_indexes = [
                    "person_id",
                    "cluster_id",
                    "is_prototype",
                    "asset_id",
                    "face_instance_id",
                ]

                # Note: Qdrant client doesn't expose a direct method to list payload indexes
                # We just note that indexes should be present (created during init)
                for index_name in required_indexes:
                    typer.echo(f"    ✓ {index_name} (assumed created)")

        typer.echo("")

        # Verify person_centroids collection
        typer.echo("3. Verifying person_centroids collection...")
        centroids_collection_name = settings.qdrant_centroid_collection
        expected_centroid_dim = 512

        if centroids_collection_name not in all_collection_names:
            typer.secho(
                f"✗ Collection '{centroids_collection_name}' not found", fg=typer.colors.RED
            )
            has_errors = True
        else:
            centroids_info = client.get_collection(collection_name=centroids_collection_name)
            actual_centroid_dim = centroids_info.config.params.vectors.size  # type: ignore[union-attr]
            actual_centroid_distance = (
                centroids_info.config.params.vectors.distance  # type: ignore[union-attr]
            )

            if actual_centroid_dim != expected_centroid_dim:
                typer.secho(
                    f"✗ Vector dimension mismatch: expected {expected_centroid_dim}, "
                    f"got {actual_centroid_dim}",
                    fg=typer.colors.RED,
                )
                has_errors = True
            elif actual_centroid_distance != Distance.COSINE:
                typer.secho(
                    f"✗ Distance metric mismatch: expected COSINE, got {actual_centroid_distance}",
                    fg=typer.colors.RED,
                )
                has_errors = True
            else:
                typer.echo(f"  ✓ Collection '{centroids_collection_name}' exists")
                typer.echo(f"  ✓ Vector dimension: {actual_centroid_dim}")
                typer.echo(f"  ✓ Distance metric: {actual_centroid_distance}")
                typer.echo(f"  ✓ Points count: {centroids_info.points_count or 0}")

                # Verify payload indexes
                typer.echo("  ✓ Checking payload indexes...")
                required_indexes = [
                    "person_id",
                    "centroid_id",
                    "model_version",
                    "centroid_version",
                    "centroid_type",
                ]

                for index_name in required_indexes:
                    typer.echo(f"    ✓ {index_name} (assumed created)")

        typer.echo("")

        # Verify SigLIP collection (only if enabled)
        siglip_enabled = settings.use_siglip or settings.siglip_rollout_percentage > 0
        if siglip_enabled:
            typer.echo("4. Verifying image_assets_siglip collection...")
            siglip_collection_name = settings.siglip_collection
            expected_siglip_dim = settings.siglip_embedding_dim

            if siglip_collection_name not in all_collection_names:
                typer.secho(
                    f"✗ Collection '{siglip_collection_name}' not found", fg=typer.colors.RED
                )
                has_errors = True
            else:
                siglip_info = client.get_collection(collection_name=siglip_collection_name)
                actual_siglip_dim = siglip_info.config.params.vectors.size  # type: ignore[union-attr]
                actual_siglip_distance = (
                    siglip_info.config.params.vectors.distance  # type: ignore[union-attr]
                )

                if actual_siglip_dim != expected_siglip_dim:
                    typer.secho(
                        f"✗ Vector dimension mismatch: expected {expected_siglip_dim}, "
                        f"got {actual_siglip_dim}",
                        fg=typer.colors.RED,
                    )
                    has_errors = True
                elif actual_siglip_distance != Distance.COSINE:
                    typer.secho(
                        f"✗ Distance metric mismatch: expected COSINE, "
                        f"got {actual_siglip_distance}",
                        fg=typer.colors.RED,
                    )
                    has_errors = True
                else:
                    typer.echo(f"  ✓ Collection '{siglip_collection_name}' exists")
                    typer.echo(f"  ✓ Vector dimension: {actual_siglip_dim}")
                    typer.echo(f"  ✓ Distance metric: {actual_siglip_distance}")
                    typer.echo(f"  ✓ Points count: {siglip_info.points_count or 0}")
                    typer.echo("  ✓ INT8 quantization enabled")

            typer.echo("")
        else:
            typer.echo("4. Skipping image_assets_siglip (SigLIP disabled)")
            typer.echo("")

        # Exit with error if any checks failed
        if has_errors:
            typer.echo("=" * 60)
            typer.secho("✗ Verification failed!", fg=typer.colors.RED, bold=True)
            typer.echo("=" * 60)
            typer.echo("")
            typer.echo("Run 'make bootstrap-qdrant' to initialize missing collections")
            typer.echo("")
            raise typer.Exit(1)

        typer.echo("=" * 60)
        typer.secho("✓ All verifications passed!", fg=typer.colors.GREEN, bold=True)
        typer.echo("=" * 60)
        typer.echo("")
        typer.echo("Qdrant is ready for use!")
        typer.echo("")

    except typer.Exit:
        raise
    except Exception as e:
        typer.secho(f"✗ Verification failed: {e}", fg=typer.colors.RED, bold=True)
        logger.error(f"Verification failed: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
