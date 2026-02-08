"""CLI commands for face detection and recognition pipeline."""

import logging
from datetime import datetime

import typer
from tqdm import tqdm

logger = logging.getLogger(__name__)

faces_app = typer.Typer(name="faces", help="Face detection and recognition commands")


def _print_bottleneck_analysis(
    io_time: float,
    gpu_time: float,
    batch_size: int,
) -> None:
    """Print bottleneck analysis based on I/O vs GPU time distribution.

    Args:
        io_time: Total time spent on I/O operations (seconds)
        gpu_time: Total time spent on GPU operations (seconds)
        batch_size: Current batch size setting
    """
    if io_time <= 0 and gpu_time <= 0:
        return

    total_time = io_time + gpu_time
    io_pct = (io_time / total_time * 100) if total_time > 0 else 0
    gpu_pct = (gpu_time / total_time * 100) if total_time > 0 else 0

    typer.echo("\n" + "=" * 50)
    typer.echo("BOTTLENECK ANALYSIS:")
    typer.echo("=" * 50)
    typer.echo(f"I/O Time:  {io_time:.1f}s ({io_pct:.1f}%)")
    typer.echo(f"GPU Time:  {gpu_time:.1f}s ({gpu_pct:.1f}%)")
    typer.echo(f"Total:     {total_time:.1f}s")
    typer.echo("")

    # Determine bottleneck
    io_bound_threshold = 70.0  # If I/O > 70%, it's I/O bound
    gpu_bound_threshold = 70.0  # If GPU > 70%, it's GPU bound

    if io_pct > io_bound_threshold:
        typer.echo("⚠️  I/O BOUND - Consider:")
        typer.echo("  • Increase NFS buffer size (current: likely 2KB → try 1MB)")
        typer.echo(f"  • Increase batch size (current: {batch_size} → try {batch_size * 2})")
        typer.echo("  • Add image prefetching with threading")
        typer.echo("  • Check NFS mount options (rsize, wsize)")
    elif gpu_pct > gpu_bound_threshold:
        typer.echo("✓ GPU BOUND - Processing is GPU-limited (optimal for this workload)")
        if batch_size < 8:
            typer.echo(f"  • Consider increasing batch size (current: {batch_size} → try 8)")
            typer.echo("    to better overlap I/O with GPU processing")
    else:
        typer.echo("⚖️  BALANCED - I/O and GPU are roughly equal")
        typer.echo(f"  • Current batch size ({batch_size}) seems appropriate")
        typer.echo("  • Monitor performance over larger workloads")


@faces_app.command("backfill")
def backfill_faces(
    limit: int = typer.Option(1000, help="Number of assets to process"),
    offset: int = typer.Option(0, help="Starting offset"),
    min_confidence: float = typer.Option(0.5, help="Detection confidence threshold"),
    batch_size: int = typer.Option(8, help="Number of images to pre-load for GPU pipeline"),
    queue: bool = typer.Option(False, help="Run as background job instead of directly"),
) -> None:
    """Backfill face detection for existing assets without faces.

    Example:
        faces backfill --limit 500 --min-confidence 0.6 --batch-size 16
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
            batch_size=batch_size,
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

            # Track cumulative timing for progress reports
            cumulative_io_time = 0.0
            cumulative_gpu_time = 0.0
            batch_counter = 0
            report_interval = 10  # Report every 10 batches

            # Setup progress bar with timing callback
            def progress_with_timing(step: int) -> None:
                """Progress callback that also prints timing every 10 batches."""
                nonlocal batch_counter, cumulative_io_time, cumulative_gpu_time
                pbar.update(step)
                batch_counter += step

                # Report every report_interval * batch_size assets
                if batch_counter > 0 and batch_counter % (report_interval * batch_size) == 0:
                    if cumulative_io_time > 0 or cumulative_gpu_time > 0:
                        total_time = cumulative_io_time + cumulative_gpu_time
                        io_pct = (cumulative_io_time / total_time * 100) if total_time > 0 else 0
                        gpu_pct = (cumulative_gpu_time / total_time * 100) if total_time > 0 else 0
                        typer.echo(
                            f"\nBatch {batch_counter // batch_size}: "
                            f"I/O {io_pct:.1f}% ({cumulative_io_time:.1f}s) | "
                            f"GPU {gpu_pct:.1f}% ({cumulative_gpu_time:.1f}s)"
                        )

            with tqdm(total=len(assets), desc="Face detection", unit="img") as pbar:
                service = get_face_service(db_session)
                result = service.process_assets_batch(
                    asset_ids=[a.id for a in assets],
                    min_confidence=min_confidence,
                    prefetch_batch_size=batch_size,
                    progress_callback=progress_with_timing,
                )

                # Update cumulative times
                cumulative_io_time = result.get('io_time', 0.0)
                cumulative_gpu_time = result.get('gpu_time', 0.0)

            typer.echo(f"\nProcessed: {result['processed']} assets")
            typer.echo(f"Faces detected: {result['total_faces']}")
            typer.echo(f"Errors: {result['errors']}")
            if result.get('throughput'):
                typer.echo(f"Throughput: {result['throughput']:.2f} images/second")

            # Print bottleneck analysis
            _print_bottleneck_analysis(
                io_time=cumulative_io_time,
                gpu_time=cumulative_gpu_time,
                batch_size=batch_size,
            )
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
        from image_search_service.vector.face_qdrant import get_face_qdrant_client

        db_session = get_sync_session()
        try:
            qdrant_client = get_face_qdrant_client()
            clusterer = get_face_clusterer(
                db_session=db_session,
                qdrant_client=qdrant_client,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
            )

            with tqdm(desc="Clustering faces", unit="face") as pbar:
                pbar.set_postfix_str(f"quality>={quality_threshold}, max={max_faces}")
                result = clusterer.cluster_unlabeled_faces(
                    quality_threshold=quality_threshold,
                    max_faces=max_faces,
                    time_bucket=time_bucket,
                )
                pbar.update(result['total_faces'])

            typer.echo(f"\nTotal faces processed: {result['total_faces']}")
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

        db_session = get_sync_session()
        try:
            assigner = get_face_assigner(
                db_session=db_session, similarity_threshold=threshold
            )

            with tqdm(total=max_faces, desc="Assigning faces", unit="face") as pbar:
                pbar.set_postfix_str(f"threshold={threshold:.2f}")
                result = assigner.assign_new_faces(since=since_dt, max_faces=max_faces)
                pbar.update(result['processed'])

            typer.echo(f"\nProcessed: {result['processed']} faces")
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

        db_session = get_sync_session()
        try:
            assigner = get_face_assigner(db_session=db_session)

            with tqdm(desc="Computing centroids", unit="person") as pbar:
                result = assigner.compute_person_centroids()
                pbar.update(result['persons_processed'])

            typer.echo(f"\nPersons processed: {result['persons_processed']}")
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

        db_session = get_sync_session()
        try:
            clusterer = get_dual_mode_clusterer(
                db_session=db_session,
                person_match_threshold=person_threshold,
                unknown_min_cluster_size=unknown_min_size,
                unknown_method=unknown_method,
                unknown_eps=unknown_eps,
            )

            with tqdm(desc="Dual-mode clustering", unit="face") as pbar:
                pbar.set_postfix_str(f"method={unknown_method}, threshold={person_threshold:.2f}")
                result = clusterer.cluster_all_faces(max_faces=max_faces)
                pbar.update(result['total_processed'])

            typer.echo(f"\nAssigned to people: {result['assigned_to_people']}")
            typer.echo(f"Unknown clusters: {result['unknown_clusters']}")
            typer.echo(f"Total processed: {result['total_processed']}")
        finally:
            db_session.close()


@faces_app.command("train-matching")
def train_person_matching(
    epochs: int = typer.Option(20, help="Training epochs"),
    margin: float = typer.Option(0.2, help="Triplet loss margin"),
    batch_size: int = typer.Option(32, help="Batch size"),
    learning_rate: float = typer.Option(0.0001, help="Learning rate"),
    min_faces: int = typer.Option(5, help="Min faces per person to include"),
    checkpoint: str | None = typer.Option(None, help="Checkpoint save path"),
    queue: bool = typer.Option(False, help="Run as background job"),
) -> None:
    """Train face matching model using triplet loss on labeled faces.

    Example:
        faces train-matching --epochs 50 --margin 0.3
    """
    if queue:
        # Note: train_matching_job needs to be implemented in queue/face_jobs.py
        typer.echo("Queue support not yet implemented for train-matching")
        typer.echo("Run without --queue flag to execute directly")
        raise typer.Exit(1)
    else:
        from image_search_service.db.sync_operations import get_sync_session
        from image_search_service.faces.trainer import get_face_trainer

        typer.echo("Training face matching model with triplet loss...")
        typer.echo(f"Epochs: {epochs}, Margin: {margin}, Batch size: {batch_size}")
        typer.echo(f"Learning rate: {learning_rate}, Min faces per person: {min_faces}")

        db_session = get_sync_session()
        try:
            trainer = get_face_trainer(
                db_session=db_session,
                margin=margin,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )

            with tqdm(total=epochs, desc="Training epochs", unit="epoch") as pbar:
                pbar.set_postfix_str(f"margin={margin}, lr={learning_rate}")
                result = trainer.fine_tune_for_person_clustering(
                    min_faces_per_person=min_faces,
                    checkpoint_path=checkpoint,
                )
                pbar.update(result['epochs'])

            typer.echo("\n=== Training Results ===")
            typer.echo(f"Epochs completed: {result['epochs']}")
            typer.echo(f"Final loss: {result['final_loss']:.4f}")
            typer.echo(f"Persons trained: {result['persons_trained']}")
            typer.echo(f"Triplets used: {result['triplets_used']}")

            if checkpoint:
                typer.echo(f"Checkpoint saved to: {checkpoint}")
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


@faces_app.command("expire-suggestions")
def expire_suggestions(
    days: int = typer.Option(30, help="Expire suggestions older than this many days"),
    queue: bool = typer.Option(False, help="Run as background job"),
) -> None:
    """Expire old pending face suggestions."""
    if queue:
        from redis import Redis
        from rq import Queue

        from image_search_service.core.config import get_settings
        from image_search_service.queue.face_jobs import expire_old_suggestions_job

        settings = get_settings()
        redis_conn = Redis.from_url(settings.redis_url)
        q = Queue("default", connection=redis_conn)
        job = q.enqueue(expire_old_suggestions_job, days_threshold=days)
        typer.echo(f"Queued expiration job: {job.id}")
    else:
        from image_search_service.queue.face_jobs import expire_old_suggestions_job

        result = expire_old_suggestions_job(days_threshold=days)
        typer.echo(f"Expired {result.get('expired_count', 0)} suggestions")


@faces_app.command("backfill-prototypes")
def backfill_prototypes(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be created without creating"
    ),
    max_per_person: int = typer.Option(10, help="Maximum faces to process per person"),
) -> None:
    """Create prototypes from all existing face→person assignments.

    This command backfills prototypes for persons who have labeled faces but
    no prototypes. Useful for fixing existing data after prototype system deployment.

    Example:
        faces backfill-prototypes --dry-run
        faces backfill-prototypes --max-per-person 5
    """
    import asyncio

    from sqlalchemy import select

    from image_search_service.core.config import get_settings
    from image_search_service.db.models import FaceInstance, Person, PersonStatus
    from image_search_service.db.session import get_async_session_context
    from image_search_service.services.prototype_service import (
        create_or_update_prototypes,
        get_prototype_count,
    )
    from image_search_service.vector.face_qdrant import get_face_qdrant_client

    async def _backfill() -> dict:
        """Async backfill implementation."""
        settings = get_settings()
        qdrant = get_face_qdrant_client()

        async with get_async_session_context() as db:
            # Get all active persons
            person_query = select(Person).where(Person.status == PersonStatus.ACTIVE)
            person_result = await db.execute(person_query)
            persons = list(person_result.scalars().all())

            typer.echo(f"Found {len(persons)} active persons")
            typer.echo("")

            prototypes_created = 0
            persons_processed = 0

            with tqdm(total=len(persons), desc="Processing persons", unit="person") as pbar:
                for person in persons:
                    # Check if person already has prototypes
                    existing_count = await get_prototype_count(db, person.id)

                    if existing_count > 0:
                        pbar.set_postfix_str(f"{person.name}: {existing_count} existing")
                        pbar.update(1)
                        continue

                    # Get all labeled faces for this person, sorted by quality descending
                    faces_query = (
                        select(FaceInstance)
                        .where(FaceInstance.person_id == person.id)
                        .order_by(FaceInstance.quality_score.desc())
                        .limit(max_per_person)
                    )
                    faces_result = await db.execute(faces_query)
                    faces = list(faces_result.scalars().all())

                    if not faces:
                        pbar.set_postfix_str(f"{person.name}: no faces")
                        pbar.update(1)
                        continue

                    # Create prototypes from top quality faces
                    person_prototypes = 0
                    for face in faces:
                        if dry_run:
                            # Just show what would be created
                            quality = face.quality_score or 0.0
                            typer.echo(
                                f"  Would create prototype for {person.name}: "
                                f"face {face.id} (quality={quality:.2f})"
                            )
                            person_prototypes += 1
                        else:
                            # Actually create prototype
                            try:
                                proto = await create_or_update_prototypes(
                                    db=db,
                                    qdrant=qdrant,
                                    person_id=person.id,
                                    newly_labeled_face_id=face.id,
                                    max_exemplars=settings.face_prototype_max_exemplars,
                                    min_quality_threshold=settings.face_prototype_min_quality,
                                )
                                if proto:
                                    person_prototypes += 1
                            except Exception as e:
                                logger.warning(
                                    f"Failed to create prototype for face {face.id}: {e}"
                                )

                    if not dry_run:
                        await db.commit()

                    if person_prototypes > 0:
                        prototypes_created += person_prototypes
                        persons_processed += 1

                    pbar.set_postfix_str(f"{person.name}: {person_prototypes} prototypes")
                    pbar.update(1)

        return {
            "persons_processed": persons_processed,
            "prototypes_created": prototypes_created,
        }

    # Run async function
    result = asyncio.run(_backfill())

    typer.echo("")
    typer.echo("=" * 50)
    if dry_run:
        typer.echo("DRY RUN RESULTS:")
    else:
        typer.echo("BACKFILL RESULTS:")
    typer.echo("=" * 50)
    typer.echo(f"Persons processed: {result['persons_processed']}")
    typer.echo(f"Prototypes created: {result['prototypes_created']}")
    typer.echo("")


@faces_app.command("find-orphans")
def find_orphan_faces(
    limit: int = typer.Option(100, help="Maximum faces to check"),
    fix: bool = typer.Option(False, "--fix", help="Re-detect assets with orphaned faces"),
) -> None:
    """Find faces that exist in PostgreSQL but not in Qdrant.

    These "orphan" faces were created when DB commit succeeded but Qdrant upsert failed.
    Use --fix to automatically re-run face detection on affected assets.

    Example:
        faces find-orphans --limit 1000
        faces find-orphans --fix
    """
    from collections import defaultdict

    from sqlalchemy import select

    from image_search_service.db.models import FaceInstance, ImageAsset
    from image_search_service.db.sync_operations import get_sync_session
    from image_search_service.faces.service import get_face_service
    from image_search_service.vector.face_qdrant import get_face_qdrant_client

    db_session = get_sync_session()
    try:
        # Get face instances from PostgreSQL
        query = select(FaceInstance).limit(limit)
        faces = db_session.execute(query).scalars().all()

        typer.echo(f"Checking {len(faces)} faces for orphaned embeddings...")
        typer.echo("")

        # Initialize Qdrant client
        qdrant_client = get_face_qdrant_client()

        # Check each face for Qdrant point existence
        orphaned_faces: list[FaceInstance] = []
        asset_orphan_counts: dict[int, int] = defaultdict(int)

        with tqdm(total=len(faces), desc="Checking faces", unit="face") as pbar:
            for face in faces:
                exists = qdrant_client.point_exists(face.qdrant_point_id)
                if not exists:
                    orphaned_faces.append(face)
                    asset_orphan_counts[face.asset_id] += 1
                pbar.update(1)

        # Report results
        typer.echo("")
        typer.echo("=" * 50)
        typer.echo("ORPHAN DETECTION RESULTS:")
        typer.echo("=" * 50)
        typer.echo(f"Total faces checked: {len(faces)}")
        typer.echo(f"Orphaned faces found: {len(orphaned_faces)}")
        typer.echo(f"Affected assets: {len(asset_orphan_counts)}")
        typer.echo("")

        if orphaned_faces:
            # Show sample of orphaned faces
            typer.echo("Sample orphaned faces:")
            for face in orphaned_faces[:10]:
                typer.echo(
                    f"  - Face {face.id} (asset_id={face.asset_id}, "
                    f"qdrant_point_id={face.qdrant_point_id})"
                )
            if len(orphaned_faces) > 10:
                typer.echo(f"  ... and {len(orphaned_faces) - 10} more")
            typer.echo("")

            # Show affected assets
            typer.echo("Affected assets (asset_id: orphan_count):")
            sorted_assets = sorted(asset_orphan_counts.items(), key=lambda x: x[1], reverse=True)
            for asset_id, count in sorted_assets[:10]:
                typer.echo(f"  - Asset {asset_id}: {count} orphaned face(s)")
            if len(sorted_assets) > 10:
                typer.echo(f"  ... and {len(sorted_assets) - 10} more assets")
            typer.echo("")

            # Fix if requested
            if fix:
                typer.echo("=" * 50)
                typer.echo("RE-DETECTING FACES FOR AFFECTED ASSETS:")
                typer.echo("=" * 50)

                # Get unique asset IDs
                unique_asset_ids = list(asset_orphan_counts.keys())
                typer.echo(f"Re-detecting faces for {len(unique_asset_ids)} assets...")
                typer.echo("")

                # Get face service
                service = get_face_service(db_session)

                # Process each asset
                processed = 0
                errors = 0
                total_faces = 0

                with tqdm(total=len(unique_asset_ids), desc="Re-detecting", unit="asset") as pbar:
                    for asset_id in unique_asset_ids:
                        try:
                            # Get asset
                            asset = db_session.get(ImageAsset, asset_id)
                            if not asset:
                                typer.echo(f"Warning: Asset {asset_id} not found")
                                errors += 1
                                pbar.update(1)
                                continue

                            # Process asset
                            face_instances = service.process_asset(
                                asset=asset,
                                min_confidence=0.5,
                            )

                            processed += 1
                            total_faces += len(face_instances)

                        except Exception as e:
                            typer.echo(f"Error processing asset {asset_id}: {e}")
                            errors += 1

                        pbar.update(1)

                typer.echo("")
                typer.echo("=" * 50)
                typer.echo("RE-DETECTION RESULTS:")
                typer.echo("=" * 50)
                typer.echo(f"Assets processed: {processed}")
                typer.echo(f"Errors: {errors}")
                typer.echo(f"Total faces re-detected: {total_faces}")
                typer.echo("")
                typer.echo("NOTE: Re-run 'faces find-orphans' to verify fix")

            else:
                typer.echo("TIP: Use --fix to re-detect faces for affected assets")
        else:
            typer.echo("✓ No orphaned faces found!")

    finally:
        db_session.close()


if __name__ == "__main__":
    faces_app()
