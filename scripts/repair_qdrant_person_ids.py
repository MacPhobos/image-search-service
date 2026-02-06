#!/usr/bin/env python3
"""Repair script: Sync person_ids from PostgreSQL to Qdrant for all assigned faces.

Run this once after deploying the C2 fix to repair historical desync.

Usage:
    cd image-search-service
    python -m scripts.repair_qdrant_person_ids
"""
import logging
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy import select

from image_search_service.db.models import FaceInstance
from image_search_service.db.session import get_sync_session
from image_search_service.vector.face_qdrant import get_face_qdrant_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def repair_qdrant_person_ids() -> None:
    """Sync all person_id values from PostgreSQL to Qdrant."""
    db_session = get_sync_session()
    qdrant = get_face_qdrant_client()

    try:
        # Get all faces with person_id assigned
        query = select(FaceInstance).where(FaceInstance.person_id.isnot(None))
        result = db_session.execute(query)
        faces = result.scalars().all()

        logger.info(f"Found {len(faces)} faces with person_id assigned in DB")

        # Group by person_id for batch Qdrant updates
        person_faces: dict = defaultdict(list)
        for face in faces:
            if face.qdrant_point_id:
                person_faces[face.person_id].append(face.qdrant_point_id)

        # Batch update Qdrant
        total_synced = 0
        total_errors = 0
        for person_id, point_ids in person_faces.items():
            try:
                qdrant.update_person_ids(point_ids, person_id)
                total_synced += len(point_ids)
                logger.debug(f"Synced {len(point_ids)} faces for person {person_id}")
            except Exception as e:
                total_errors += len(point_ids)
                logger.error(f"Failed to sync faces for person {person_id}: {e}")

        logger.info(
            f"Repair complete: synced {total_synced} faces "
            f"across {len(person_faces)} persons "
            f"({total_errors} errors)"
        )

    finally:
        db_session.close()


if __name__ == "__main__":
    repair_qdrant_person_ids()
