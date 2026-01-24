#!/usr/bin/env python3
"""Enable scalar quantization on existing Qdrant collections.

This script enables INT8 scalar quantization on the image_assets collection
to reduce memory usage by ~75% with negligible accuracy loss (<1%).

Usage:
    # Preview what will happen (dry run)
    python scripts/enable_quantization.py --dry-run

    # Apply quantization
    python scripts/enable_quantization.py

    # Check status only
    python scripts/enable_quantization.py --status

Requirements:
    - Qdrant server must be running
    - QDRANT_URL environment variable (default: http://localhost:6333)
"""

import argparse
import sys
import time

from qdrant_client import QdrantClient
from qdrant_client.models import (
    OptimizersConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
)


def get_client() -> QdrantClient:
    """Get Qdrant client from environment."""
    import os

    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY", None)

    return QdrantClient(url=url, api_key=api_key)


def get_collection_info(client: QdrantClient, collection_name: str) -> dict:
    """Get collection statistics."""
    try:
        info = client.get_collection(collection_name)
        return {
            "name": collection_name,
            "status": info.status.value,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "vector_size": info.config.params.vectors.size if info.config.params.vectors else None,
            "quantization_enabled": info.config.quantization_config is not None,
            "quantization_config": info.config.quantization_config,
        }
    except Exception as e:
        return {"name": collection_name, "error": str(e)}


def estimate_memory_savings(vectors_count: int, vector_dim: int) -> dict:
    """Estimate memory savings from quantization."""
    # Float32: 4 bytes per dimension
    # INT8: 1 byte per dimension
    # HNSW overhead multiplier: ~1.5x

    float32_bytes = vectors_count * vector_dim * 4 * 1.5
    int8_bytes = vectors_count * vector_dim * 1 * 1.5

    return {
        "current_memory_mb": float32_bytes / (1024 * 1024),
        "quantized_memory_mb": int8_bytes / (1024 * 1024),
        "savings_mb": (float32_bytes - int8_bytes) / (1024 * 1024),
        "savings_percent": ((float32_bytes - int8_bytes) / float32_bytes) * 100,
    }


def enable_quantization(
    client: QdrantClient,
    collection_name: str,
    dry_run: bool = False,
) -> bool:
    """Enable scalar quantization on a collection."""
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Enabling quantization on '{collection_name}'...")

    # Get current state
    info = get_collection_info(client, collection_name)

    if "error" in info:
        print(f"  Error: {info['error']}")
        return False

    if info["quantization_enabled"]:
        print(f"  Quantization already enabled: {info['quantization_config']}")
        return True

    # Show stats
    print(f"  Current status: {info['status']}")
    print(f"  Vectors: {info['vectors_count']}")
    print(f"  Vector dimension: {info['vector_size']}")

    # Estimate savings
    if info["vectors_count"] and info["vector_size"]:
        savings = estimate_memory_savings(info["vectors_count"], info["vector_size"])
        print(f"\n  Memory estimation:")
        print(f"    Current: {savings['current_memory_mb']:.2f} MB")
        print(f"    After quantization: {savings['quantized_memory_mb']:.2f} MB")
        print(f"    Savings: {savings['savings_mb']:.2f} MB ({savings['savings_percent']:.1f}%)")

    if dry_run:
        print("\n  [DRY RUN] Would apply scalar quantization (INT8)")
        return True

    # Apply quantization
    print("\n  Applying scalar quantization (INT8)...")

    try:
        client.update_collection(
            collection_name=collection_name,
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=0.99,  # Use 99th percentile for better range coverage
                    always_ram=True,  # Keep quantized vectors in RAM for speed
                ),
            ),
        )
        print("  Quantization config applied.")

        # Trigger reindexing
        print("  Triggering reindexing...")
        client.update_collection(
            collection_name=collection_name,
            optimizer_config=OptimizersConfigDiff(
                indexing_threshold=10000,
            ),
        )

        # Wait for indexing to complete
        print("  Waiting for indexing to complete...")
        max_wait = 300  # 5 minutes max
        start_time = time.time()

        while time.time() - start_time < max_wait:
            info = client.get_collection(collection_name)
            status = info.status.value

            if status == "green":
                print(f"\n  Done! Collection '{collection_name}' is now quantized.")
                return True

            indexed = info.indexed_vectors_count or 0
            total = info.vectors_count or 0
            pct = (indexed / total * 100) if total > 0 else 0
            print(f"    Status: {status}, indexed: {indexed}/{total} ({pct:.1f}%)")
            time.sleep(5)

        print(f"\n  Warning: Indexing not complete after {max_wait}s, but config is applied.")
        return True

    except Exception as e:
        print(f"\n  Error applying quantization: {e}")
        return False


def show_status(client: QdrantClient, collection_names: list[str]) -> None:
    """Show status of collections."""
    print("\nCollection Status:")
    print("-" * 60)

    for name in collection_names:
        info = get_collection_info(client, name)

        if "error" in info:
            print(f"\n{name}: ERROR - {info['error']}")
            continue

        print(f"\n{name}:")
        print(f"  Status: {info['status']}")
        print(f"  Vectors: {info['vectors_count']}")
        print(f"  Vector dimension: {info['vector_size']}")
        print(f"  Quantization: {'Enabled' if info['quantization_enabled'] else 'Disabled'}")

        if info["quantization_enabled"]:
            print(f"  Config: {info['quantization_config']}")

        if info["vectors_count"] and info["vector_size"]:
            savings = estimate_memory_savings(info["vectors_count"], info["vector_size"])
            if info["quantization_enabled"]:
                print(f"  Memory (quantized): ~{savings['quantized_memory_mb']:.2f} MB")
            else:
                print(f"  Memory (current): ~{savings['current_memory_mb']:.2f} MB")
                print(
                    f"  Potential savings: ~{savings['savings_mb']:.2f} MB ({savings['savings_percent']:.1f}%)"
                )


def main():
    parser = argparse.ArgumentParser(description="Enable scalar quantization on Qdrant collections")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without making changes")
    parser.add_argument("--status", action="store_true", help="Show collection status only")
    parser.add_argument(
        "--collections",
        nargs="+",
        default=["image_assets"],
        help="Collections to process (default: image_assets)",
    )

    args = parser.parse_args()

    print("Qdrant Quantization Tool")
    print("=" * 60)

    client = get_client()

    # Test connection
    try:
        client.get_collections()
        print("Connected to Qdrant")
    except Exception as e:
        print(f"Failed to connect to Qdrant: {e}")
        sys.exit(1)

    if args.status:
        show_status(client, args.collections)
        return

    # Process collections
    success = True
    for collection_name in args.collections:
        if not enable_quantization(client, collection_name, dry_run=args.dry_run):
            success = False

    if not success:
        sys.exit(1)

    print("\n" + "=" * 60)
    if args.dry_run:
        print("Dry run complete. No changes made.")
    else:
        print("Quantization complete!")


if __name__ == "__main__":
    main()
