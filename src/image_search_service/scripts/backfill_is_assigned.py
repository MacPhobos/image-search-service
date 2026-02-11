#!/usr/bin/env python3
"""One-time backfill script to set is_assigned on all face points in Qdrant.

This script backfills the is_assigned boolean sentinel field for all existing
face points in the Qdrant faces collection. This enables fast server-side
filtering for unlabeled faces without using IsEmptyCondition.

Usage:
    python -m image_search_service.scripts.backfill_is_assigned [--batch-size 100]
"""

import typer

from image_search_service.core.logging import get_logger
from image_search_service.vector.face_qdrant import get_face_qdrant_client

logger = get_logger(__name__)

app = typer.Typer(
    name="backfill-is-assigned",
    help="Backfill is_assigned field for all face points in Qdrant",
)


@app.command()
def backfill(
    batch_size: int = typer.Option(
        100,
        "--batch-size",
        "-b",
        help="Number of points to process per batch",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview changes without applying them",
    ),
) -> None:
    """Backfill is_assigned field for all face points.

    This sets:
    - is_assigned=True for faces with person_id
    - is_assigned=False for faces without person_id

    The field enables fast server-side filtering for unlabeled faces.

    Example:
        python -m image_search_service.scripts.backfill_is_assigned
        python -m image_search_service.scripts.backfill_is_assigned --batch-size 200
        python -m image_search_service.scripts.backfill_is_assigned --dry-run
    """
    typer.echo("=" * 60)
    typer.echo("Backfilling is_assigned field for face points...")
    typer.echo("=" * 60)
    typer.echo("")

    if dry_run:
        typer.secho("DRY RUN MODE - No changes will be applied", fg=typer.colors.YELLOW, bold=True)
        typer.echo("")

    try:
        # Get Qdrant client
        qdrant_client = get_face_qdrant_client()

        # Get collection info
        collection_info = qdrant_client.get_collection_info()
        if not collection_info:
            typer.secho("✗ Could not get collection info", fg=typer.colors.RED)
            raise typer.Exit(1)

        total_points = collection_info.get("points_count", 0)
        typer.echo(f"Total face points in collection: {total_points}")
        typer.echo(f"Batch size: {batch_size}")
        typer.echo("")

        if total_points == 0:
            typer.secho("✓ No face points to process", fg=typer.colors.GREEN)
            return

        if dry_run:
            typer.echo("Would process all face points and set:")
            typer.echo("  - is_assigned=True for faces with person_id")
            typer.echo("  - is_assigned=False for faces without person_id")
            typer.echo("")
            typer.secho("✓ Dry run complete (no changes made)", fg=typer.colors.GREEN)
            return

        # Run backfill
        typer.echo("Processing face points...")
        updated_count = qdrant_client.backfill_is_assigned(batch_size=batch_size)

        typer.echo("")
        typer.echo("=" * 60)
        typer.secho("✓ Backfill complete!", fg=typer.colors.GREEN, bold=True)
        typer.echo("=" * 60)
        typer.echo("")
        typer.echo(f"Points updated: {updated_count}")
        typer.echo("")
        typer.echo("Next steps:")
        typer.echo("  - Verify with: make faces-stats")
        typer.echo("  - Test unlabeled face queries are now faster")
        typer.echo("")

    except Exception as e:
        typer.secho(f"✗ Backfill failed: {e}", fg=typer.colors.RED, bold=True)
        logger.error(f"Backfill failed: {e}")
        raise typer.Exit(1)


@app.command()
def verify() -> None:
    """Verify is_assigned field coverage in the collection.

    Checks:
    - How many points have is_assigned field
    - How many are assigned vs unassigned
    - Identifies any points missing the field

    Example:
        python -m image_search_service.scripts.backfill_is_assigned verify
    """
    typer.echo("=" * 60)
    typer.echo("Verifying is_assigned field coverage...")
    typer.echo("=" * 60)
    typer.echo("")

    try:
        # Get Qdrant client
        qdrant_client = get_face_qdrant_client()

        # Get collection info
        collection_info = qdrant_client.get_collection_info()
        if not collection_info:
            typer.secho("✗ Could not get collection info", fg=typer.colors.RED)
            raise typer.Exit(1)

        total_points = collection_info.get("points_count", 0)
        typer.echo(f"Total face points: {total_points}")
        typer.echo("")

        if total_points == 0:
            typer.secho("✓ No face points to verify", fg=typer.colors.GREEN)
            return

        # Scroll through collection to check is_assigned coverage
        typer.echo("Scanning collection for is_assigned field...")
        offset = None
        has_is_assigned = 0
        missing_is_assigned = 0
        assigned_true = 0
        assigned_false = 0

        while True:
            from image_search_service.vector.face_qdrant import _get_face_collection_name

            records, next_offset = qdrant_client.client.scroll(
                collection_name=_get_face_collection_name(),
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            if not records:
                break

            for record in records:
                if record.payload and "is_assigned" in record.payload:
                    has_is_assigned += 1
                    if record.payload["is_assigned"]:
                        assigned_true += 1
                    else:
                        assigned_false += 1
                else:
                    missing_is_assigned += 1

            if next_offset is None:
                break

            offset = next_offset

        typer.echo("")
        typer.echo("=" * 60)
        typer.echo("Verification Results")
        typer.echo("=" * 60)
        typer.echo(f"Points with is_assigned:    {has_is_assigned}")
        typer.echo(f"  - is_assigned=True:       {assigned_true}")
        typer.echo(f"  - is_assigned=False:      {assigned_false}")
        typer.echo(f"Points missing is_assigned: {missing_is_assigned}")
        typer.echo("")

        if missing_is_assigned > 0:
            coverage_pct = (has_is_assigned / total_points) * 100
            typer.secho(
                f"⚠ Coverage: {coverage_pct:.1f}% ({has_is_assigned}/{total_points})",
                fg=typer.colors.YELLOW,
                bold=True,
            )
            typer.echo("")
            typer.echo("Run backfill command to fix:")
            typer.echo("  python -m image_search_service.scripts.backfill_is_assigned backfill")
        else:
            typer.secho("✓ All points have is_assigned field!", fg=typer.colors.GREEN, bold=True)

        typer.echo("")

    except Exception as e:
        typer.secho(f"✗ Verification failed: {e}", fg=typer.colors.RED, bold=True)
        logger.error(f"Verification failed: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
