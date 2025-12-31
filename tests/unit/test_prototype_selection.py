"""Tests for temporal prototype selection logic."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from image_search_service.db.models import (
    AgeEraBucket,
    FaceInstance,
    PersonPrototype,
    PrototypeRole,
)
from image_search_service.services.prototype_service import (
    prune_temporal_prototypes,
    select_temporal_prototypes,
)


@pytest.fixture
def mock_person_id() -> UUID:
    """Fixture for a test person UUID."""
    return uuid4()


@pytest.fixture
def mock_qdrant():
    """Fixture for mock Qdrant client."""
    mock = MagicMock()
    mock.update_payload = MagicMock()
    return mock


@pytest.fixture
def mock_face_instance():
    """Factory fixture for creating mock face instances."""

    def _create_face(
        face_id: UUID | None = None,
        quality_score: float = 0.8,
        age_estimate: int | None = 25,
        person_id: UUID | None = None,
    ) -> FaceInstance:
        """Create a mock face instance with temporal metadata."""
        face = MagicMock(spec=FaceInstance)
        face.id = face_id or uuid4()
        face.qdrant_point_id = uuid4()
        face.quality_score = quality_score
        face.person_id = person_id
        face.asset = MagicMock()
        face.asset.file_modified_at = None

        # Mock landmarks with age metadata
        if age_estimate is not None:
            face.landmarks = {
                "age_estimate": age_estimate,
                "age_confidence": 0.9,
                "pose": "frontal",
                "bbox": {"width": 150, "height": 150},
            }
        else:
            face.landmarks = None

        return face

    return _create_face


class TestSelectTemporalPrototypes:
    """Tests for temporal prototype selection."""

    @pytest.mark.asyncio
    async def test_selects_one_per_era(self, mock_person_id, mock_qdrant, mock_face_instance):
        """Selects one prototype per age era when available."""
        # Create mock DB session
        mock_db = AsyncMock()

        # Create faces for different eras
        faces = [
            mock_face_instance(quality_score=0.9, age_estimate=2),  # infant
            mock_face_instance(quality_score=0.85, age_estimate=10),  # child
            mock_face_instance(quality_score=0.8, age_estimate=17),  # teen
        ]

        # Mock get_faces_with_temporal_metadata
        with patch(
            "image_search_service.services.prototype_service.get_faces_with_temporal_metadata"
        ) as mock_get_faces:
            mock_get_faces.return_value = faces

            # Mock get_prototypes_for_person (no existing prototypes)
            with patch(
                "image_search_service.services.prototype_service.get_prototypes_for_person"
            ) as mock_get_protos:
                mock_get_protos.return_value = []

                # Mock DB operations
                mock_db.execute = AsyncMock()
                mock_db.execute.return_value.scalar_one_or_none.return_value = None
                mock_db.add = MagicMock()
                mock_db.flush = AsyncMock()

                # Call function
                result = await select_temporal_prototypes(
                    db=mock_db, qdrant=mock_qdrant, person_id=mock_person_id, preserve_pins=True
                )

                # Verify we created prototypes for all eras represented
                assert len(result) >= 3  # At least one per era

    @pytest.mark.asyncio
    async def test_preserves_pinned(self, mock_person_id, mock_qdrant, mock_face_instance):
        """Pinned prototypes are preserved."""
        mock_db = AsyncMock()

        # Create a face
        face = mock_face_instance(quality_score=0.9, age_estimate=25)

        # Create a pinned prototype
        pinned_proto = MagicMock(spec=PersonPrototype)
        pinned_proto.id = uuid4()
        pinned_proto.is_pinned = True
        pinned_proto.age_era_bucket = AgeEraBucket.YOUNG_ADULT.value
        pinned_proto.face_instance_id = uuid4()

        with patch(
            "image_search_service.services.prototype_service.get_faces_with_temporal_metadata"
        ) as mock_get_faces:
            mock_get_faces.return_value = [face]

            with patch(
                "image_search_service.services.prototype_service.get_prototypes_for_person"
            ) as mock_get_protos:
                mock_get_protos.return_value = [pinned_proto]

                mock_db.execute = AsyncMock()
                mock_db.execute.return_value.scalar_one_or_none.return_value = None
                mock_db.add = MagicMock()
                mock_db.flush = AsyncMock()

                result = await select_temporal_prototypes(
                    db=mock_db, qdrant=mock_qdrant, person_id=mock_person_id, preserve_pins=True
                )

                # Should not create prototype for era that already has pinned one
                # Result should not include duplicate for YOUNG_ADULT era
                era_buckets = [p.age_era_bucket for p in result if hasattr(p, "age_era_bucket")]
                assert (
                    AgeEraBucket.YOUNG_ADULT.value not in era_buckets
                )  # Pinned era should be skipped

    @pytest.mark.asyncio
    async def test_uses_highest_quality_per_era(
        self, mock_person_id, mock_qdrant, mock_face_instance
    ):
        """Selects highest quality face within each era."""
        mock_db = AsyncMock()

        # Create multiple faces in same era with different qualities
        faces = [
            mock_face_instance(quality_score=0.6, age_estimate=25),
            mock_face_instance(quality_score=0.9, age_estimate=28),  # Highest quality
            mock_face_instance(quality_score=0.7, age_estimate=30),
        ]

        with patch(
            "image_search_service.services.prototype_service.get_faces_with_temporal_metadata"
        ) as mock_get_faces:
            mock_get_faces.return_value = faces

            with patch(
                "image_search_service.services.prototype_service.get_prototypes_for_person"
            ) as mock_get_protos:
                mock_get_protos.return_value = []

                mock_db.execute = AsyncMock()
                mock_db.execute.return_value.scalar_one_or_none.return_value = None
                mock_db.add = MagicMock()
                mock_db.flush = AsyncMock()

                result = await select_temporal_prototypes(
                    db=mock_db, qdrant=mock_qdrant, person_id=mock_person_id, preserve_pins=True
                )

                # Verify at least one prototype was created
                assert len(result) > 0

                # The selected face should be the highest quality one (0.9)
                # We can verify this by checking the order of db.add calls
                added_protos = [call.args[0] for call in mock_db.add.call_args_list]
                if added_protos:
                    # Check that the first added prototype for YOUNG_ADULT era uses highest quality face
                    young_adult_protos = [
                        p
                        for p in added_protos
                        if hasattr(p, "age_era_bucket")
                        and p.age_era_bucket == AgeEraBucket.YOUNG_ADULT.value
                    ]
                    if young_adult_protos:
                        # The highest quality face (0.9) should be selected
                        assert young_adult_protos[0].face_instance_id == faces[1].id

    @pytest.mark.asyncio
    async def test_fills_exemplars_after_temporal(
        self, mock_person_id, mock_qdrant, mock_face_instance
    ):
        """Remaining slots filled with EXEMPLAR role."""
        mock_db = AsyncMock()

        # Create many faces (more than temporal slots)
        faces = [
            mock_face_instance(quality_score=0.95, age_estimate=2),  # infant
            mock_face_instance(quality_score=0.9, age_estimate=50),  # adult - high quality
            mock_face_instance(quality_score=0.85, age_estimate=55),  # adult - lower quality
        ]

        with patch(
            "image_search_service.services.prototype_service.get_faces_with_temporal_metadata"
        ) as mock_get_faces:
            mock_get_faces.return_value = faces

            with patch(
                "image_search_service.services.prototype_service.get_prototypes_for_person"
            ) as mock_get_protos:
                mock_get_protos.return_value = []

                mock_db.execute = AsyncMock()
                mock_db.execute.return_value.scalar_one_or_none.return_value = None
                mock_db.add = MagicMock()
                mock_db.flush = AsyncMock()

                # Mock settings with max_total that allows for exemplars
                with patch("image_search_service.services.prototype_service.get_settings") as mock_settings:
                    mock_settings.return_value.face_prototype_max_total = 10
                    mock_settings.return_value.face_prototype_min_quality = 0.5

                    result = await select_temporal_prototypes(
                        db=mock_db,
                        qdrant=mock_qdrant,
                        person_id=mock_person_id,
                        preserve_pins=True,
                    )

                    # Should have created multiple prototypes
                    assert len(result) >= 2


class TestPruneTemporalPrototypes:
    """Tests for temporal prototype pruning."""

    @pytest.mark.asyncio
    async def test_never_prunes_pinned(self, mock_person_id, mock_qdrant):
        """Pinned prototypes never pruned."""
        mock_db = AsyncMock()

        # Create pinned and unpinned prototypes
        pinned_proto = MagicMock(spec=PersonPrototype)
        pinned_proto.id = uuid4()
        pinned_proto.is_pinned = True
        pinned_proto.role = PrototypeRole.PRIMARY
        pinned_proto.qdrant_point_id = uuid4()

        unpinned_proto = MagicMock(spec=PersonPrototype)
        unpinned_proto.id = uuid4()
        unpinned_proto.is_pinned = False
        unpinned_proto.role = PrototypeRole.EXEMPLAR
        unpinned_proto.qdrant_point_id = uuid4()
        unpinned_proto.face_instance_id = uuid4()

        all_protos = [pinned_proto, unpinned_proto]

        with patch(
            "image_search_service.services.prototype_service.get_prototypes_for_person"
        ) as mock_get_protos:
            mock_get_protos.return_value = all_protos

            mock_db.execute = AsyncMock()
            mock_db.execute.return_value.scalars.return_value.all.return_value = []
            mock_db.delete = AsyncMock()
            mock_db.flush = AsyncMock()

            # Prune to max_total=1 (should delete unpinned, keep pinned)
            deleted_ids = await prune_temporal_prototypes(
                db=mock_db,
                qdrant=mock_qdrant,
                person_id=mock_person_id,
                max_total=1,
                preserve_pins=True,
            )

            # Should have deleted unpinned only
            assert len(deleted_ids) == 1
            assert unpinned_proto.id in deleted_ids
            assert pinned_proto.id not in deleted_ids

    @pytest.mark.asyncio
    async def test_prunes_lowest_priority_first(self, mock_person_id, mock_qdrant):
        """Prunes FALLBACK before EXEMPLAR before TEMPORAL."""
        mock_db = AsyncMock()

        # Create prototypes with different roles
        temporal_proto = MagicMock(spec=PersonPrototype)
        temporal_proto.id = uuid4()
        temporal_proto.is_pinned = False
        temporal_proto.role = PrototypeRole.TEMPORAL
        temporal_proto.qdrant_point_id = uuid4()
        temporal_proto.face_instance_id = uuid4()

        exemplar_proto = MagicMock(spec=PersonPrototype)
        exemplar_proto.id = uuid4()
        exemplar_proto.is_pinned = False
        exemplar_proto.role = PrototypeRole.EXEMPLAR
        exemplar_proto.qdrant_point_id = uuid4()
        exemplar_proto.face_instance_id = uuid4()

        fallback_proto = MagicMock(spec=PersonPrototype)
        fallback_proto.id = uuid4()
        fallback_proto.is_pinned = False
        fallback_proto.role = PrototypeRole.FALLBACK
        fallback_proto.qdrant_point_id = uuid4()
        fallback_proto.face_instance_id = uuid4()

        all_protos = [temporal_proto, exemplar_proto, fallback_proto]

        # Mock face instances with equal quality
        face1 = MagicMock(spec=FaceInstance)
        face1.id = temporal_proto.face_instance_id
        face1.quality_score = 0.8

        face2 = MagicMock(spec=FaceInstance)
        face2.id = exemplar_proto.face_instance_id
        face2.quality_score = 0.8

        face3 = MagicMock(spec=FaceInstance)
        face3.id = fallback_proto.face_instance_id
        face3.quality_score = 0.8

        with patch(
            "image_search_service.services.prototype_service.get_prototypes_for_person"
        ) as mock_get_protos:
            mock_get_protos.return_value = all_protos

            mock_db.execute = AsyncMock()
            mock_db.execute.return_value.scalars.return_value.all.return_value = [face1, face2, face3]
            mock_db.delete = AsyncMock()
            mock_db.flush = AsyncMock()

            # Prune to max_total=1 (should keep TEMPORAL, delete others)
            deleted_ids = await prune_temporal_prototypes(
                db=mock_db,
                qdrant=mock_qdrant,
                person_id=mock_person_id,
                max_total=1,
                preserve_pins=True,
            )

            # Should have deleted FALLBACK first, then EXEMPLAR
            assert len(deleted_ids) == 2
            assert fallback_proto.id in deleted_ids
            assert exemplar_proto.id in deleted_ids
            assert temporal_proto.id not in deleted_ids

    @pytest.mark.asyncio
    async def test_keeps_one_per_era(self, mock_person_id, mock_qdrant):
        """Keeps at least one prototype per covered era."""
        mock_db = AsyncMock()

        # Create temporal prototypes for different eras
        infant_proto = MagicMock(spec=PersonPrototype)
        infant_proto.id = uuid4()
        infant_proto.is_pinned = False
        infant_proto.role = PrototypeRole.TEMPORAL
        infant_proto.age_era_bucket = AgeEraBucket.INFANT.value
        infant_proto.qdrant_point_id = uuid4()
        infant_proto.face_instance_id = uuid4()

        child_proto = MagicMock(spec=PersonPrototype)
        child_proto.id = uuid4()
        child_proto.is_pinned = False
        child_proto.role = PrototypeRole.TEMPORAL
        child_proto.age_era_bucket = AgeEraBucket.CHILD.value
        child_proto.qdrant_point_id = uuid4()
        child_proto.face_instance_id = uuid4()

        all_protos = [infant_proto, child_proto]

        # Mock face instances
        face1 = MagicMock(spec=FaceInstance)
        face1.id = infant_proto.face_instance_id
        face1.quality_score = 0.7

        face2 = MagicMock(spec=FaceInstance)
        face2.id = child_proto.face_instance_id
        face2.quality_score = 0.8

        with patch(
            "image_search_service.services.prototype_service.get_prototypes_for_person"
        ) as mock_get_protos:
            mock_get_protos.return_value = all_protos

            mock_db.execute = AsyncMock()
            mock_db.execute.return_value.scalars.return_value.all.return_value = [face1, face2]
            mock_db.delete = AsyncMock()
            mock_db.flush = AsyncMock()

            # Prune to max_total=2 (should keep both - one per era)
            deleted_ids = await prune_temporal_prototypes(
                db=mock_db,
                qdrant=mock_qdrant,
                person_id=mock_person_id,
                max_total=2,
                preserve_pins=True,
            )

            # Should not delete any (both are temporal for different eras)
            assert len(deleted_ids) == 0
