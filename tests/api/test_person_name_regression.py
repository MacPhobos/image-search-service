"""Regression tests for personName population bug.

Bug fixed: API endpoints were returning personName=null even when personId was set.

Root cause: Queries were not JOINing with the Person table to fetch person names.

Fixed endpoints:
- GET /api/v1/faces/clusters/{cluster_id}
- GET /api/v1/faces/assets/{asset_id}

These tests ensure that personName is correctly populated when a face is assigned to a person.
"""

import uuid

import pytest

# mock_image_asset, mock_person from root conftest.py


class TestGetClusterPersonName:
    """Regression tests for GET /api/v1/faces/clusters/{cluster_id} personName."""

    @pytest.mark.asyncio
    async def test_get_cluster_returns_person_name_when_assigned(
        self, test_client, db_session, mock_image_asset, mock_person
    ):
        """Test that personName is populated when face is assigned to a person.

        Regression test for bug where personId was set but personName was null.
        The query must JOIN with Person table to fetch person names.
        """
        from image_search_service.db.models import FaceInstance

        cluster_id = "test_cluster_with_person"

        # Create a face assigned to a person
        face = FaceInstance(
            id=uuid.uuid4(),
            asset_id=mock_image_asset.id,
            bbox_x=100,
            bbox_y=100,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.95,
            quality_score=0.85,
            qdrant_point_id=uuid.uuid4(),
            cluster_id=cluster_id,
            person_id=mock_person.id,  # Assigned to person
        )
        db_session.add(face)
        await db_session.commit()

        # Call the endpoint
        response = await test_client.get(f"/api/v1/faces/clusters/{cluster_id}")

        assert response.status_code == 200
        data = response.json()

        # Verify cluster-level person info
        assert data["clusterId"] == cluster_id
        assert data["personId"] == str(mock_person.id)
        assert data["personName"] == "Test Person", (
            f"Expected personName='Test Person' but got {data['personName']!r}. "
            "The query should join with the persons table to populate person names."
        )

        # Verify face-level person info
        assert len(data["faces"]) == 1
        face_response = data["faces"][0]
        assert face_response["personId"] == str(mock_person.id)
        assert face_response["personName"] == "Test Person", (
            f"Expected face personName='Test Person' but got {face_response['personName']!r}. "
            "Each face should also have personName populated."
        )

    @pytest.mark.asyncio
    async def test_get_cluster_returns_null_person_when_unassigned(
        self, test_client, db_session, mock_image_asset
    ):
        """Test that personName is null when no person is assigned."""
        from image_search_service.db.models import FaceInstance

        cluster_id = "test_cluster_no_person"

        # Create a face NOT assigned to any person
        face = FaceInstance(
            id=uuid.uuid4(),
            asset_id=mock_image_asset.id,
            bbox_x=100,
            bbox_y=100,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.95,
            quality_score=0.85,
            qdrant_point_id=uuid.uuid4(),
            cluster_id=cluster_id,
            person_id=None,  # No person assigned
        )
        db_session.add(face)
        await db_session.commit()

        # Call the endpoint
        response = await test_client.get(f"/api/v1/faces/clusters/{cluster_id}")

        assert response.status_code == 200
        data = response.json()

        # Verify cluster-level person info is null
        assert data["clusterId"] == cluster_id
        assert data["personId"] is None
        assert data["personName"] is None

        # Verify face-level person info is null
        assert len(data["faces"]) == 1
        face_response = data["faces"][0]
        assert face_response["personId"] is None
        assert face_response["personName"] is None

    @pytest.mark.asyncio
    async def test_get_cluster_with_multiple_faces_same_person(
        self, test_client, db_session, mock_image_asset, mock_person
    ):
        """Test cluster with multiple faces all assigned to the same person."""
        from image_search_service.db.models import FaceInstance

        cluster_id = "test_cluster_multiple_same_person"

        # Create 3 faces all assigned to the same person
        for i in range(3):
            face = FaceInstance(
                id=uuid.uuid4(),
                asset_id=mock_image_asset.id,
                bbox_x=100 + i * 100,
                bbox_y=100,
                bbox_w=80,
                bbox_h=80,
                detection_confidence=0.95,
                quality_score=0.85,
                qdrant_point_id=uuid.uuid4(),
                cluster_id=cluster_id,
                person_id=mock_person.id,
            )
            db_session.add(face)
        await db_session.commit()

        # Call the endpoint
        response = await test_client.get(f"/api/v1/faces/clusters/{cluster_id}")

        assert response.status_code == 200
        data = response.json()

        # All faces should have the same person
        assert data["personId"] == str(mock_person.id)
        assert data["personName"] == "Test Person"
        assert len(data["faces"]) == 3

        for face_response in data["faces"]:
            assert face_response["personId"] == str(mock_person.id)
            assert face_response["personName"] == "Test Person"


class TestGetFacesForAssetPersonName:
    """Regression tests for GET /api/v1/faces/assets/{asset_id} personName."""

    @pytest.mark.asyncio
    async def test_get_faces_for_asset_returns_person_name_when_assigned(
        self, test_client, db_session, mock_image_asset, mock_person
    ):
        """Test that personName is populated when face is assigned to a person.

        Regression test for bug where personId was set but personName was null.
        """
        from image_search_service.db.models import FaceInstance

        # Create a face assigned to a person
        face = FaceInstance(
            id=uuid.uuid4(),
            asset_id=mock_image_asset.id,
            bbox_x=100,
            bbox_y=100,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.95,
            quality_score=0.85,
            qdrant_point_id=uuid.uuid4(),
            person_id=mock_person.id,  # Assigned to person
        )
        db_session.add(face)
        await db_session.commit()

        # Call the endpoint
        response = await test_client.get(f"/api/v1/faces/assets/{mock_image_asset.id}")

        assert response.status_code == 200
        data = response.json()

        assert data["total"] == 1
        assert len(data["items"]) == 1

        face_response = data["items"][0]
        assert face_response["id"] == str(face.id)
        assert face_response["personId"] == str(mock_person.id)
        assert face_response["personName"] == "Test Person", (
            f"Expected personName='Test Person' but got {face_response['personName']!r}. "
            "The query should join with the persons table to populate person names."
        )

    @pytest.mark.asyncio
    async def test_get_faces_for_asset_returns_null_person_when_unassigned(
        self, test_client, db_session, mock_image_asset
    ):
        """Test that personName is null when no person is assigned."""
        from image_search_service.db.models import FaceInstance

        # Create a face NOT assigned to any person
        face = FaceInstance(
            id=uuid.uuid4(),
            asset_id=mock_image_asset.id,
            bbox_x=100,
            bbox_y=100,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.95,
            quality_score=0.85,
            qdrant_point_id=uuid.uuid4(),
            person_id=None,  # No person assigned
        )
        db_session.add(face)
        await db_session.commit()

        # Call the endpoint
        response = await test_client.get(f"/api/v1/faces/assets/{mock_image_asset.id}")

        assert response.status_code == 200
        data = response.json()

        assert data["total"] == 1
        assert len(data["items"]) == 1

        face_response = data["items"][0]
        assert face_response["id"] == str(face.id)
        assert face_response["personId"] is None
        assert face_response["personName"] is None

    @pytest.mark.asyncio
    async def test_get_faces_for_asset_with_multiple_faces_different_persons(
        self, test_client, db_session, mock_image_asset
    ):
        """Test asset with multiple faces assigned to different persons."""
        from image_search_service.db.models import FaceInstance, Person, PersonStatus

        # Create two different persons
        person1 = Person(
            id=uuid.uuid4(),
            name="Alice",
            status=PersonStatus.ACTIVE.value,
        )
        person2 = Person(
            id=uuid.uuid4(),
            name="Bob",
            status=PersonStatus.ACTIVE.value,
        )
        db_session.add(person1)
        db_session.add(person2)
        await db_session.flush()

        # Create face assigned to person1
        face1 = FaceInstance(
            id=uuid.uuid4(),
            asset_id=mock_image_asset.id,
            bbox_x=100,
            bbox_y=100,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.95,
            quality_score=0.85,
            qdrant_point_id=uuid.uuid4(),
            person_id=person1.id,
        )

        # Create face assigned to person2
        face2 = FaceInstance(
            id=uuid.uuid4(),
            asset_id=mock_image_asset.id,
            bbox_x=300,
            bbox_y=100,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.92,
            quality_score=0.78,
            qdrant_point_id=uuid.uuid4(),
            person_id=person2.id,
        )

        # Create face with no person
        face3 = FaceInstance(
            id=uuid.uuid4(),
            asset_id=mock_image_asset.id,
            bbox_x=500,
            bbox_y=100,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.88,
            quality_score=0.72,
            qdrant_point_id=uuid.uuid4(),
            person_id=None,
        )

        db_session.add(face1)
        db_session.add(face2)
        db_session.add(face3)
        await db_session.commit()

        # Call the endpoint
        response = await test_client.get(f"/api/v1/faces/assets/{mock_image_asset.id}")

        assert response.status_code == 200
        data = response.json()

        assert data["total"] == 3
        assert len(data["items"]) == 3

        # Find each face in the response
        faces_by_id = {item["id"]: item for item in data["items"]}

        # Verify face1 has Alice's name
        face1_response = faces_by_id[str(face1.id)]
        assert face1_response["personId"] == str(person1.id)
        assert face1_response["personName"] == "Alice"

        # Verify face2 has Bob's name
        face2_response = faces_by_id[str(face2.id)]
        assert face2_response["personId"] == str(person2.id)
        assert face2_response["personName"] == "Bob"

        # Verify face3 has no person
        face3_response = faces_by_id[str(face3.id)]
        assert face3_response["personId"] is None
        assert face3_response["personName"] is None
