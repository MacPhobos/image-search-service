"""CLI commands for face detection and recognition pipeline."""

import logging
from datetime import datetime

import typer

logger = logging.getLogger(__name__)

faces_app = typer.Typer(name="faces", help="Face detection and recognition commands")


@faces_app.command("backfill")
def backfill_faces(
    limit: int = typer.Option(1000, help="Number of assets to process"),
    offset: int = typer.Option(0, help="Starting offset"),
    min_confidence: float = typer.Option(0.5, help="Detection confidence threshold"),
    queue: bool = typer.Option(False, help="Run as background job instead of directly"),
) -> None:
    """Backfill face detection for existing assets without faces.

    Example:
        faces backfill --limit 500 --min-confidence 0.6
    """
    if queue:
        from redis import Redis
        from rq import Queue

        from image_search_service.core.config import get_settings
        from image_search_service.queue.face_jobs import backfill_faces_job

        settings = get_settings()
        redis = Redis.from_url(settings.redis_url)
        q = Queue("default", connection=redis)

        job = q.enqueue(
            backfill_faces_job,
            limit=limit,
            offset=offset,
            min_confidence=min_confidence,
        )
        typer.echo(f"Queued backfill job: {job.id}")
    else:
        from sqlalchemy import select

        from image_search_service.db.models import FaceInstance, ImageAsset
        from image_search_service.db.sync_operations import get_sync_session
        from image_search_service.faces.service import get_face_service

        db_session = get_sync_session()
        try:
            # Get assets without faces
            subquery = select(FaceInstance.asset_id).distinct()
            query = (
                select(ImageAsset)
                .where(~ImageAsset.id.in_(subquery))
                .offset(offset)
                .limit(limit)
            )
            assets = db_session.execute(query).scalars().all()

            typer.echo(f"Processing {len(assets)} assets...")

            service = get_face_service(db_session)
            result = service.process_assets_batch(
                asset_ids=[a.id for a in assets],
                min_confidence=min_confidence,
            )

            typer.echo(f"Processed: {result['processed']} assets")
            typer.echo(f"Faces detected: {result['total_faces']}")
            typer.echo(f"Errors: {result['errors']}")
        finally:
            db_session.close()


@faces_app.command("cluster")
def cluster_faces(
    quality_threshold: float = typer.Option(0.5, help="Minimum quality score"),
    max_faces: int = typer.Option(50000, help="Maximum faces to cluster"),
    min_cluster_size: int = typer.Option(5, help="HDBSCAN min_cluster_size"),
    min_samples: int = typer.Option(3, help="HDBSCAN min_samples"),
    time_bucket: str | None = typer.Option(None, help="Filter by YYYY-MM"),
    queue: bool = typer.Option(False, help="Run as background job"),
) -> None:
    """Cluster unlabeled faces using HDBSCAN.

    Example:
        faces cluster --quality-threshold 0.6 --min-cluster-size 3
    """
    if queue:
        from redis import Redis
        from rq import Queue

        from image_search_service.core.config import get_settings
        from image_search_service.queue.face_jobs import cluster_faces_job

        settings = get_settings()
        redis = Redis.from_url(settings.redis_url)
        q = Queue("default", connection=redis)

        job = q.enqueue(
            cluster_faces_job,
            quality_threshold=quality_threshold,
            max_faces=max_faces,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            time_bucket=time_bucket,
        )
        typer.echo(f"Queued clustering job: {job.id}")
    else:
        from image_search_service.db.sync_operations import get_sync_session
        from image_search_service.faces.clusterer import get_face_clusterer

        typer.echo(f"Clustering up to {max_faces} faces...")

        db_session = get_sync_session()
        try:
            clusterer = get_face_clusterer(
                db_session=db_session,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
            )
            result = clusterer.cluster_unlabeled_faces(
                quality_threshold=quality_threshold,
                max_faces=max_faces,
                time_bucket=time_bucket,
            )

            typer.echo(f"Total faces processed: {result['total_faces']}")
            typer.echo(f"Clusters found: {result['clusters_found']}")
            typer.echo(f"Noise (unclustered): {result['noise_count']}")
        finally:
            db_session.close()


@faces_app.command("assign")
def assign_faces(
    since: str | None = typer.Option(
        None, help="Only faces created after YYYY-MM-DD"
    ),
    max_faces: int = typer.Option(1000, help="Maximum faces to process"),
    threshold: float = typer.Option(0.6, help="Similarity threshold for assignment"),
    queue: bool = typer.Option(False, help="Run as background job"),
) -> None:
    """Assign new faces to known persons via prototype matching.

    Example:
        faces assign --since 2025-12-01 --threshold 0.65
    """
    since_dt = None
    if since:
        since_dt = datetime.fromisoformat(since)

    if queue:
        from redis import Redis
        from rq import Queue

        from image_search_service.core.config import get_settings
        from image_search_service.queue.face_jobs import assign_faces_job

        settings = get_settings()
        redis = Redis.from_url(settings.redis_url)
        q = Queue("default", connection=redis)

        # Convert datetime to ISO string for serialization
        since_iso = since_dt.isoformat() if since_dt else None

        job = q.enqueue(
            assign_faces_job,
            since=since_iso,
            max_faces=max_faces,
            similarity_threshold=threshold,
        )
        typer.echo(f"Queued assignment job: {job.id}")
    else:
        from image_search_service.db.sync_operations import get_sync_session
        from image_search_service.faces.assigner import get_face_assigner

        typer.echo(f"Assigning up to {max_faces} faces...")

        db_session = get_sync_session()
        try:
            assigner = get_face_assigner(
                db_session=db_session, similarity_threshold=threshold
            )
            result = assigner.assign_new_faces(since=since_dt, max_faces=max_faces)

            typer.echo(f"Processed: {result['processed']} faces")
            typer.echo(f"Assigned: {result['assigned']}")
            typer.echo(f"Unassigned: {result['unassigned']}")
            typer.echo(f"Status: {result['status']}")
        finally:
            db_session.close()


@faces_app.command("centroids")
def compute_centroids(
    queue: bool = typer.Option(False, help="Run as background job"),
) -> None:
    """Compute/update person centroid embeddings.

    Example:
        faces centroids
    """
    if queue:
        from redis import Redis
        from rq import Queue

        from image_search_service.core.config import get_settings
        from image_search_service.queue.face_jobs import compute_centroids_job

        settings = get_settings()
        redis = Redis.from_url(settings.redis_url)
        q = Queue("default", connection=redis)

        job = q.enqueue(compute_centroids_job)
        typer.echo(f"Queued centroid computation job: {job.id}")
    else:
        from image_search_service.db.sync_operations import get_sync_session
        from image_search_service.faces.assigner import get_face_assigner

        typer.echo("Computing centroids...")

        db_session = get_sync_session()
        try:
            assigner = get_face_assigner(db_session=db_session)
            result = assigner.compute_person_centroids()

            typer.echo(f"Persons processed: {result['persons_processed']}")
            typer.echo(f"Centroids computed: {result['centroids_computed']}")
        finally:
            db_session.close()


@faces_app.command("ensure-collection")
def ensure_qdrant_collection() -> None:
    """Ensure the Qdrant faces collection exists with proper indexes.

    Example:
        faces ensure-collection
    """
    from image_search_service.vector.face_qdrant import get_face_qdrant_client

    typer.echo("Ensuring faces collection...")

    client = get_face_qdrant_client()
    client.ensure_collection()

    info = client.get_collection_info()
    if info:
        typer.echo(f"Collection: {info.get('name', 'faces')}")
        typer.echo(f"Vectors: {info.get('vectors_count', 0)}")
        typer.echo(f"Points: {info.get('points_count', 0)}")
    else:
        typer.echo("Collection created successfully")


@faces_app.command("cluster-dual")
def cluster_faces_dual(
    person_threshold: float = typer.Option(0.7, help="Person match threshold (0-1)"),
    unknown_method: str = typer.Option("hdbscan", help="Unknown clustering method"),
    unknown_min_size: int = typer.Option(3, help="Min cluster size for unknown"),
    unknown_eps: float = typer.Option(0.5, help="Distance threshold for DBSCAN/Agglomerative"),
    max_faces: int | None = typer.Option(None, help="Max faces to process"),
    queue: bool = typer.Option(False, help="Run as background job"),
) -> None:
    """Run dual-mode clustering (supervised + unsupervised).

    Example:
        faces cluster-dual --person-threshold 0.75 --unknown-method hdbscan
    """
    if queue:
        # Note: cluster_dual_job needs to be implemented in queue/face_jobs.py
        typer.echo("Queue support not yet implemented for cluster-dual")
        typer.echo("Run without --queue flag to execute directly")
        raise typer.Exit(1)
    else:
        from image_search_service.db.sync_operations import get_sync_session
        from image_search_service.faces.dual_clusterer import get_dual_mode_clusterer

        typer.echo("Running dual-mode clustering...")

        db_session = get_sync_session()
        try:
            clusterer = get_dual_mode_clusterer(
                db_session=db_session,
                person_match_threshold=person_threshold,
                unknown_min_cluster_size=unknown_min_size,
                unknown_method=unknown_method,
                unknown_eps=unknown_eps,
            )

            result = clusterer.cluster_all_faces(max_faces=max_faces)

            typer.echo(f"Assigned to people: {result['assigned_to_people']}")
            typer.echo(f"Unknown clusters: {result['unknown_clusters']}")
            typer.echo(f"Total processed: {result['total_processed']}")
        finally:
            db_session.close()


@faces_app.command("stats")
def show_stats() -> None:
    """Show face detection and recognition statistics.

    Example:
        faces stats
    """
    from sqlalchemy import func, select

    from image_search_service.db.models import (
        FaceInstance,
        Person,
        PersonPrototype,
        PersonStatus,
    )
    from image_search_service.db.sync_operations import get_sync_session
    from image_search_service.vector.face_qdrant import get_face_qdrant_client

    db_session = get_sync_session()
    try:
        # Face counts
        total_faces = (
            db_session.execute(select(func.count(FaceInstance.id))).scalar() or 0
        )
        assigned_faces = (
            db_session.execute(
                select(func.count(FaceInstance.id)).where(
                    FaceInstance.person_id.isnot(None)
                )
            ).scalar()
            or 0
        )
        clustered_faces = (
            db_session.execute(
                select(func.count(FaceInstance.id)).where(
                    FaceInstance.cluster_id.isnot(None)
                )
            ).scalar()
            or 0
        )

        # Person counts
        total_persons = (
            db_session.execute(select(func.count(Person.id))).scalar() or 0
        )
        active_persons = (
            db_session.execute(
                select(func.count(Person.id)).where(Person.status == PersonStatus.ACTIVE)
            ).scalar()
            or 0
        )

        # Prototype counts
        total_prototypes = (
            db_session.execute(select(func.count(PersonPrototype.id))).scalar() or 0
        )

        # Cluster counts
        cluster_count = (
            db_session.execute(
                select(func.count(func.distinct(FaceInstance.cluster_id))).where(
                    FaceInstance.cluster_id.isnot(None)
                )
            ).scalar()
            or 0
        )
    finally:
        db_session.close()

    # Qdrant stats
    client = get_face_qdrant_client()
    qdrant_info = client.get_collection_info()

    typer.echo("=== Face Detection Statistics ===")
    typer.echo(f"Total faces: {total_faces}")
    typer.echo(f"Assigned to persons: {assigned_faces}")
    typer.echo(f"In clusters: {clustered_faces}")
    typer.echo(f"Unassigned: {total_faces - assigned_faces - clustered_faces}")
    typer.echo("")
    typer.echo("=== Person Statistics ===")
    typer.echo(f"Total persons: {total_persons}")
    typer.echo(f"Active persons: {active_persons}")
    typer.echo(f"Total prototypes: {total_prototypes}")
    typer.echo(f"Total clusters: {cluster_count}")
    typer.echo("")
    typer.echo("=== Qdrant Statistics ===")
    if qdrant_info:
        typer.echo(f"Vectors count: {qdrant_info.get('vectors_count', 'N/A')}")
        typer.echo(f"Points count: {qdrant_info.get('points_count', 'N/A')}")
    else:
        typer.echo("Collection not found or empty")


if __name__ == "__main__":
    faces_app()
