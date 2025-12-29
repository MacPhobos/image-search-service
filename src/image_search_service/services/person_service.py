"""Person service for unified people management."""

import logging
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.api.face_schemas import (
    PersonType,
    UnifiedPeopleListResponse,
    UnifiedPersonResponse,
)
from image_search_service.db.models import FaceInstance, Person, PersonStatus

logger = logging.getLogger(__name__)


class PersonService:
    """Service for managing unified person view (identified + unidentified)."""

    def __init__(self, db: AsyncSession):
        """Initialize person service.

        Args:
            db: Async database session
        """
        self.db = db

    async def get_all_people(
        self,
        include_identified: bool = True,
        include_unidentified: bool = True,
        include_noise: bool = False,
        sort_by: str = "face_count",
        sort_order: str = "desc",
    ) -> UnifiedPeopleListResponse:
        """Get all people (identified + unidentified clusters) in unified format.

        This combines:
        - Identified people from Person table
        - Unidentified clusters from FaceInstance where person_id is NULL
        - Optionally noise faces (ungrouped/outliers)

        Args:
            include_identified: Include persons with names
            include_unidentified: Include face clusters without names
            include_noise: Include noise/outlier faces
            sort_by: Field to sort by (face_count, name, created_at)
            sort_order: Sort order (asc, desc)

        Returns:
            Unified list of people with counts and metadata
        """
        people: list[UnifiedPersonResponse] = []

        # 1. Get identified people (from Person table)
        if include_identified:
            identified = await self._get_identified_people()
            people.extend(identified)

        # 2. Get unidentified clusters (from FaceInstance where person_id is NULL)
        if include_unidentified:
            unidentified = await self._get_unidentified_clusters()
            people.extend(unidentified)

        # 3. Optionally get noise faces
        if include_noise:
            noise = await self._get_noise_faces()
            if noise:
                people.append(noise)

        # Sort by requested field
        reverse = sort_order == "desc"
        if sort_by == "face_count":
            people.sort(key=lambda p: p.face_count, reverse=reverse)
        elif sort_by == "name":
            people.sort(key=lambda p: p.name.lower(), reverse=reverse)
        # created_at not available in unified view, default to face_count

        # Calculate counts
        identified_count = len([p for p in people if p.type == PersonType.IDENTIFIED])
        unidentified_count = len([p for p in people if p.type == PersonType.UNIDENTIFIED])
        noise_count = len([p for p in people if p.type == PersonType.NOISE])

        return UnifiedPeopleListResponse(
            people=people,
            total=len(people),
            identified_count=identified_count,
            unidentified_count=unidentified_count,
            noise_count=noise_count,
        )

    async def _get_identified_people(self) -> list[UnifiedPersonResponse]:
        """Get list of identified persons with face counts.

        Returns:
            List of identified persons as UnifiedPersonResponse
        """
        # Get active persons with face counts
        query = (
            select(
                Person.id,
                Person.name,
                func.count(FaceInstance.id).label("face_count"),
            )
            .outerjoin(FaceInstance, FaceInstance.person_id == Person.id)
            .where(Person.status == PersonStatus.ACTIVE)
            .group_by(Person.id, Person.name)
        )

        result = await self.db.execute(query)
        rows = result.all()

        people = []
        for row in rows:
            # Get thumbnail from highest quality face
            thumbnail_url = await self._get_person_thumbnail(row.id)

            people.append(
                UnifiedPersonResponse(
                    id=str(row.id),
                    name=row.name,
                    type=PersonType.IDENTIFIED,
                    face_count=row.face_count,
                    thumbnail_url=thumbnail_url,
                    confidence=None,  # Not applicable for identified persons
                )
            )

        return people

    async def _get_unidentified_clusters(self) -> list[UnifiedPersonResponse]:
        """Get list of unidentified face clusters.

        Returns:
            List of unidentified clusters as UnifiedPersonResponse
        """
        # Get clusters from faces with no person assignment
        # Exclude noise cluster ('-1')
        query = (
            select(
                FaceInstance.cluster_id,
                func.count(FaceInstance.id).label("face_count"),
                func.avg(FaceInstance.quality_score).label("avg_quality"),
            )
            .where(
                FaceInstance.person_id.is_(None),
                FaceInstance.cluster_id.isnot(None),
                FaceInstance.cluster_id != "-1",
            )
            .group_by(FaceInstance.cluster_id)
        )

        result = await self.db.execute(query)
        rows = result.all()

        people = []
        for index, row in enumerate(rows):
            # Get thumbnail from highest quality face in cluster
            thumbnail_url = await self._get_cluster_thumbnail(row.cluster_id)

            # Generate display name
            display_name = self._generate_display_name(row.cluster_id, index)

            people.append(
                UnifiedPersonResponse(
                    id=row.cluster_id,
                    name=display_name,
                    type=PersonType.UNIDENTIFIED,
                    face_count=row.face_count,
                    thumbnail_url=thumbnail_url,
                    confidence=row.avg_quality,
                )
            )

        return people

    async def _get_noise_faces(self) -> UnifiedPersonResponse | None:
        """Get aggregated noise faces (outliers not in any cluster).

        Returns:
            Single UnifiedPersonResponse for all noise faces, or None if no noise
        """
        # Count faces with cluster_id = '-1' or NULL (and no person_id)
        query = select(func.count(FaceInstance.id)).where(
            FaceInstance.person_id.is_(None),
            (FaceInstance.cluster_id == "-1") | (FaceInstance.cluster_id.is_(None)),
        )

        result = await self.db.execute(query)
        noise_count = result.scalar() or 0

        if noise_count == 0:
            return None

        # Get thumbnail from any noise face
        thumbnail_query = (
            select(FaceInstance.asset_id)
            .where(
                FaceInstance.person_id.is_(None),
                (FaceInstance.cluster_id == "-1") | (FaceInstance.cluster_id.is_(None)),
            )
            .limit(1)
        )
        thumbnail_result = await self.db.execute(thumbnail_query)
        asset_id = thumbnail_result.scalar()

        thumbnail_url = f"/api/v1/images/{asset_id}/thumbnail" if asset_id else None

        return UnifiedPersonResponse(
            id="-1",
            name="Unknown Faces",
            type=PersonType.NOISE,
            face_count=noise_count,
            thumbnail_url=thumbnail_url,
            confidence=None,
        )

    async def _get_person_thumbnail(self, person_id: UUID) -> str | None:
        """Get thumbnail URL for a person (highest quality face).

        Args:
            person_id: Person UUID

        Returns:
            Thumbnail URL or None
        """
        query = (
            select(FaceInstance.asset_id)
            .where(FaceInstance.person_id == person_id)
            .order_by(FaceInstance.quality_score.desc().nullslast())
            .limit(1)
        )

        result = await self.db.execute(query)
        asset_id = result.scalar()

        return f"/api/v1/images/{asset_id}/thumbnail" if asset_id else None

    async def _get_cluster_thumbnail(self, cluster_id: str) -> str | None:
        """Get thumbnail URL for a cluster (highest quality face).

        Args:
            cluster_id: Cluster ID string

        Returns:
            Thumbnail URL or None
        """
        query = (
            select(FaceInstance.asset_id)
            .where(
                FaceInstance.cluster_id == cluster_id,
                FaceInstance.person_id.is_(None),
            )
            .order_by(FaceInstance.quality_score.desc().nullslast())
            .limit(1)
        )

        result = await self.db.execute(query)
        asset_id = result.scalar()

        return f"/api/v1/images/{asset_id}/thumbnail" if asset_id else None

    def _generate_display_name(self, cluster_id: str, index: int) -> str:
        """Generate user-friendly display name for unidentified cluster.

        Args:
            cluster_id: Cluster ID string
            index: Sequential index for naming

        Returns:
            Display name like "Unidentified Person 1"
        """
        if cluster_id in ["-1", "noise", None]:
            return "Unknown Faces"

        # Generate sequential name based on index
        return f"Unidentified Person {index + 1}"
