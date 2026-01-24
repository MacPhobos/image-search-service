#!/usr/bin/env python3
"""Create parallel Qdrant collection for SigLIP embeddings.

This script creates a new Qdrant collection for SigLIP (768-dimensional) embeddings
alongside the existing CLIP collection (512-dimensional). The new collection is
optimized from the start with:

- INT8 scalar quantization (75% memory reduction, <1% accuracy loss)
- Optimized HNSW settings (m=16, ef_construct=100)
- Cosine distance metric

Usage:
    # Preview what will happen (dry run)
    uv run python scripts/create_siglip_collection.py --dry-run

    # Create the collection
    uv run python scripts/create_siglip_collection.py

    # Check if collection exists
    uv run python scripts/create_siglip_collection.py --check

Requirements:
    - Qdrant server must be running
    - Environment variables: QDRANT_URL, SIGLIP_COLLECTION, SIGLIP_EMBEDDING_DIM
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
)

from image_search_service.core.config import get_settings


def get_client() -> QdrantClient:
    """Get Qdrant client from environment."""
    settings = get_settings()
    api_key = settings.qdrant_api_key if settings.qdrant_api_key else None
    return QdrantClient(url=settings.qdrant_url, api_key=api_key)


def check_collection_exists(client: QdrantClient, collection_name: str) -> bool:
    """Check if collection exists."""
    try:
        collections = client.get_collections().collections
        return any(c.name == collection_name for c in collections)
    except Exception as e:
        print(f"Error checking collections: {e}")
        return False


def get_collection_info(client: QdrantClient, collection_name: str) -> dict:
    """Get collection information."""
    try:
        info = client.get_collection(collection_name)
        return {
            "name": collection_name,
            "status": info.status.value,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "vector_size": (
                info.config.params.vectors.size
                if info.config.params.vectors
                else None
            ),
            "distance": (
                info.config.params.vectors.distance.value
                if info.config.params.vectors
                else None
            ),
            "quantization_enabled": info.config.quantization_config is not None,
            "hnsw_m": (
                info.config.hnsw_config.m if info.config.hnsw_config else None
            ),
            "hnsw_ef_construct": (
                info.config.hnsw_config.ef_construct
                if info.config.hnsw_config
                else None
            ),
        }
    except Exception as e:
        return {"name": collection_name, "error": str(e)}


def create_siglip_collection(dry_run: bool = False) -> bool:
    """Create SigLIP collection with optimized settings.

    Args:
        dry_run: If True, only print what would be done without making changes

    Returns:
        True if successful, False otherwise
    """
    settings = get_settings()
    client = get_client()

    collection_name = settings.siglip_collection
    embedding_dim = settings.siglip_embedding_dim

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Creating SigLIP collection...")
    print(f"  Collection name: {collection_name}")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Qdrant URL: {settings.qdrant_url}")

    # Check if collection already exists
    if check_collection_exists(client, collection_name):
        print(f"\n✓ Collection '{collection_name}' already exists")
        info = get_collection_info(client, collection_name)
        if "error" not in info:
            print("\nCollection details:")
            print(f"  Status: {info['status']}")
            print(f"  Points: {info['points_count']}")
            print(f"  Vector size: {info['vector_size']}")
            print(f"  Distance: {info['distance']}")
            print(f"  Quantization enabled: {info['quantization_enabled']}")
            print(f"  HNSW m: {info['hnsw_m']}")
            print(f"  HNSW ef_construct: {info['hnsw_ef_construct']}")
        return True

    if dry_run:
        print(
            "\n[DRY RUN] Would create collection with:"
        )
        print(f"  - Vector size: {embedding_dim}")
        print("  - Distance: COSINE")
        print("  - Quantization: INT8 scalar (enabled from start)")
        print("  - HNSW settings: m=16, ef_construct=100")
        return True

    # Create collection with optimized settings
    try:
        print(f"\nCreating collection '{collection_name}'...")

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embedding_dim,  # 768 for SigLIP
                distance=Distance.COSINE,
            ),
            # Enable quantization from start (75% memory savings)
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=0.99,  # Use 99th percentile for calibration
                    always_ram=True,  # Keep quantized vectors in RAM for speed
                ),
            ),
            # Optimized HNSW settings for better recall
            hnsw_config=HnswConfigDiff(
                m=16,  # Number of neighbors per node (higher = better recall)
                ef_construct=100,  # Search depth during construction (higher = better quality)
            ),
        )

        print(f"✓ Collection '{collection_name}' created successfully")

        # Verify creation
        info = get_collection_info(client, collection_name)
        if "error" in info:
            print(f"Warning: Could not verify collection: {info['error']}")
            return False

        print("\nCollection details:")
        print(f"  Status: {info['status']}")
        print(f"  Vector size: {info['vector_size']}")
        print(f"  Distance: {info['distance']}")
        print(f"  Quantization enabled: {info['quantization_enabled']}")
        print(f"  HNSW m: {info['hnsw_m']}")
        print(f"  HNSW ef_construct: {info['hnsw_ef_construct']}")

        return True

    except Exception as e:
        print(f"✗ Failed to create collection: {e}")
        return False


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create SigLIP Qdrant collection with optimized settings"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what will be done without making changes",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if collection exists and show details",
    )

    args = parser.parse_args()

    if args.check:
        settings = get_settings()
        client = get_client()
        collection_name = settings.siglip_collection

        print(f"\nChecking collection '{collection_name}'...")
        exists = check_collection_exists(client, collection_name)

        if exists:
            print("✓ Collection exists")
            info = get_collection_info(client, collection_name)
            if "error" not in info:
                print("\nCollection details:")
                print(f"  Status: {info['status']}")
                print(f"  Points: {info['points_count']}")
                print(f"  Vector size: {info['vector_size']}")
                print(f"  Distance: {info['distance']}")
                print(f"  Quantization enabled: {info['quantization_enabled']}")
                print(f"  HNSW m: {info['hnsw_m']}")
                print(f"  HNSW ef_construct: {info['hnsw_ef_construct']}")
        else:
            print("✗ Collection does not exist")

        return 0 if exists else 1

    # Create collection
    success = create_siglip_collection(dry_run=args.dry_run)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
