"""Tests for face assignment module."""

import uuid
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestFaceAssigner:
    """Tests for FaceAssigner."""

    @pytest.mark.asyncio
    async def test_assign_no_prototypes(self, db_session):
        """Test assignment when no prototypes exist."""
        from image_search_service.faces.assigner import FaceAssigner

        with patch("image_search_service.faces.assigner.get_face_qdrant_client"):
            mock_session = MagicMock()
            mock_session.execute = MagicMock(
                return_value=MagicMock(scalars=lambda: MagicMock(all=lambda: []))
            )

            assigner = FaceAssigner(mock_session)
            result = assigner.assign_new_faces()

        assert result["status"] == "no_prototypes"
        assert result["assigned"] == 0

    @pytest.mark.asyncio
    async def test_assign_no_new_faces(self, db_session, mock_person):
        """Test assignment when no unassigned faces exist."""
        from image_search_service.faces.assigner import FaceAssigner
        from image_search_service.db.models import PersonPrototype, PrototypeRole

        # Create a prototype
        prototype = PersonPrototype(
            person_id=mock_person.id,
            qdrant_point_id=uuid.uuid4(),
            role=PrototypeRole.EXEMPLAR.value,
        )
        db_session.add(prototype)
        await db_session.commit()

        with patch("image_search_service.faces.assigner.get_face_qdrant_client"):
            mock_session = MagicMock()

            # Mock prototypes query to return one prototype
            mock_prototypes_result = MagicMock()
            mock_prototypes_result.scalars.return_value.all.return_value = [prototype]

            # Mock faces query to return no faces
            mock_faces_result = MagicMock()
            mock_faces_result.scalars.return_value.all.return_value = []

            mock_session.execute.side_effect = [
                mock_prototypes_result,
                mock_faces_result,
            ]

            assigner = FaceAssigner(mock_session)
            result = assigner.assign_new_faces()

        assert result["status"] == "no_new_faces"
        assert result["assigned"] == 0

    @pytest.mark.asyncio
    async def test_assign_face_to_person(self, db_session, mock_person, mock_face_instance):
        """Test successful face assignment to person."""
        from image_search_service.faces.assigner import FaceAssigner
        from image_search_service.db.models import PersonPrototype, PrototypeRole

        # Create a prototype
        prototype = PersonPrototype(
            person_id=mock_person.id,
            qdrant_point_id=uuid.uuid4(),
            role=PrototypeRole.EXEMPLAR.value,
        )

        with patch("image_search_service.faces.assigner.get_face_qdrant_client") as mock_get:
            # Mock Qdrant client
            mock_qdrant = MagicMock()
            mock_match = MagicMock()
            mock_match.id = str(prototype.qdrant_point_id)
            mock_match.score = 0.85
            mock_match.payload = {"person_id": str(mock_person.id)}
            mock_qdrant.search_against_prototypes.return_value = [mock_match]
            mock_qdrant.update_person_ids = MagicMock()
            mock_get.return_value = mock_qdrant

            mock_session = MagicMock()

            # Mock prototypes query
            mock_prototypes_result = MagicMock()
            mock_prototypes_result.scalars.return_value.all.return_value = [prototype]

            # Mock faces query
            mock_faces_result = MagicMock()
            mock_faces_result.scalars.return_value.all.return_value = [mock_face_instance]

            mock_session.execute.side_effect = [
                mock_prototypes_result,
                mock_faces_result,
            ]
            mock_session.commit = MagicMock()

            assigner = FaceAssigner(mock_session)

            # Mock _get_face_embedding
            with patch.object(assigner, "_get_face_embedding") as mock_get_emb:
                mock_get_emb.return_value = [0.1] * 512
                result = assigner.assign_new_faces()

        assert result["status"] == "completed"
        assert result["assigned"] == 1

    @pytest.mark.asyncio
    async def test_assign_filters_by_since(self, db_session, mock_person, mock_face_instance):
        """Test assignment filters faces by creation date."""
        from image_search_service.faces.assigner import FaceAssigner
        from image_search_service.db.models import PersonPrototype, PrototypeRole

        # Create a prototype
        prototype = PersonPrototype(
            person_id=mock_person.id,
            qdrant_point_id=uuid.uuid4(),
            role=PrototypeRole.EXEMPLAR.value,
        )

        since_date = datetime.utcnow() - timedelta(days=1)

        with patch("image_search_service.faces.assigner.get_face_qdrant_client"):
            mock_session = MagicMock()

            # Mock prototypes query
            mock_prototypes_result = MagicMock()
            mock_prototypes_result.scalars.return_value.all.return_value = [prototype]

            # Mock faces query (should be filtered by since)
            mock_faces_result = MagicMock()
            mock_faces_result.scalars.return_value.all.return_value = []

            mock_session.execute.side_effect = [
                mock_prototypes_result,
                mock_faces_result,
            ]

            assigner = FaceAssigner(mock_session)
            result = assigner.assign_new_faces(since=since_date)

        assert result["status"] == "no_new_faces"

    def test_get_face_embedding_not_found(self):
        """Test getting embedding for non-existent face."""
        from image_search_service.faces.assigner import FaceAssigner

        with patch("image_search_service.faces.assigner.get_face_qdrant_client") as mock_get:
            mock_qdrant = MagicMock()
            mock_qdrant.client.retrieve.return_value = []
            mock_get.return_value = mock_qdrant

            mock_session = MagicMock()
            assigner = FaceAssigner(mock_session)

            result = assigner._get_face_embedding(uuid.uuid4())

        assert result is None

    def test_get_face_embedding_success(self):
        """Test successfully getting face embedding."""
        from image_search_service.faces.assigner import FaceAssigner

        mock_embedding = [0.1] * 512

        with patch("image_search_service.faces.assigner.get_face_qdrant_client") as mock_get:
            mock_qdrant = MagicMock()
            mock_point = MagicMock()
            mock_point.vector = mock_embedding
            mock_qdrant.client.retrieve.return_value = [mock_point]
            mock_get.return_value = mock_qdrant

            mock_session = MagicMock()
            assigner = FaceAssigner(mock_session)

            result = assigner._get_face_embedding(uuid.uuid4())

        assert result == mock_embedding

    @pytest.mark.asyncio
    async def test_compute_person_centroids_no_faces(self, db_session, mock_person):
        """Test centroid computation when person has no faces."""
        from image_search_service.faces.assigner import FaceAssigner

        with patch("image_search_service.faces.assigner.get_face_qdrant_client"):
            mock_session = MagicMock()

            # Mock persons query
            mock_persons_result = MagicMock()
            mock_persons_result.scalars.return_value.all.return_value = [mock_person]

            # Mock faces query (no faces)
            mock_faces_result = MagicMock()
            mock_faces_result.scalars.return_value.all.return_value = []

            mock_session.execute.side_effect = [
                mock_persons_result,
                mock_faces_result,
            ]
            mock_session.commit = MagicMock()

            assigner = FaceAssigner(mock_session)
            result = assigner.compute_person_centroids()

        assert result["centroids_computed"] == 0

    @pytest.mark.asyncio
    async def test_compute_person_centroids_creates_new(
        self, db_session, mock_person, mock_face_instance
    ):
        """Test centroid computation creates new centroid prototype."""
        from image_search_service.faces.assigner import FaceAssigner

        # Create two face instances for the person
        face1 = mock_face_instance
        face2 = MagicMock()
        face2.id = uuid.uuid4()
        face2.person_id = mock_person.id
        face2.qdrant_point_id = uuid.uuid4()
        face2.asset_id = face1.asset_id

        mock_embeddings = [
            [0.1] * 512,
            [0.2] * 512,
        ]

        with patch("image_search_service.faces.assigner.get_face_qdrant_client") as mock_get:
            mock_qdrant = MagicMock()
            mock_qdrant.upsert_face = MagicMock()
            mock_get.return_value = mock_qdrant

            mock_session = MagicMock()

            # Mock persons query
            mock_persons_result = MagicMock()
            mock_persons_result.scalars.return_value.all.return_value = [mock_person]

            # Mock faces query
            mock_faces_result = MagicMock()
            mock_faces_result.scalars.return_value.all.return_value = [face1, face2]

            # Mock existing centroid query (none exists)
            mock_centroid_result = MagicMock()
            mock_centroid_result.scalar_one_or_none.return_value = None

            mock_session.execute.side_effect = [
                mock_persons_result,
                mock_faces_result,
                mock_centroid_result,
            ]
            mock_session.add = MagicMock()
            mock_session.commit = MagicMock()

            assigner = FaceAssigner(mock_session)

            # Mock _get_face_embedding to return embeddings
            with patch.object(assigner, "_get_face_embedding") as mock_get_emb:
                mock_get_emb.side_effect = mock_embeddings
                result = assigner.compute_person_centroids()

        assert result["centroids_computed"] == 1
        # Verify upsert_face was called with centroid
        mock_qdrant.upsert_face.assert_called_once()

    def test_assigner_initialization(self):
        """Test FaceAssigner initialization with custom parameters."""
        from image_search_service.faces.assigner import FaceAssigner

        mock_session = MagicMock()
        assigner = FaceAssigner(
            mock_session,
            similarity_threshold=0.75,
            max_matches_per_face=5
        )

        assert assigner.similarity_threshold == 0.75
        assert assigner.max_matches_per_face == 5

    def test_get_face_assigner_factory(self):
        """Test factory function for FaceAssigner."""
        from image_search_service.faces.assigner import get_face_assigner

        mock_session = MagicMock()
        assigner = get_face_assigner(mock_session, similarity_threshold=0.7)

        assert assigner.similarity_threshold == 0.7
