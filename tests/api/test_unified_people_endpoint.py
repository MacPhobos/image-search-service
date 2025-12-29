"""Integration tests for GET /api/v1/faces/people endpoint."""

import uuid

import pytest
from sqlalchemy import select

from image_search_service.db.models import FaceInstance, ImageAsset, Person, PersonStatus


class TestUnifiedPeopleEndpoint:
    """Integration tests for unified people listing endpoint."""

    @pytest.fixture
    async def seed_test_data(self, db_session):
        """Seed test data for people endpoint tests.

        Creates:
        - 2 identified persons with faces
        - 3 unidentified clusters with faces
        - 5 noise faces (cluster_id = '-1')
        """
        # Create test ImageAssets
        assets = []
        for i in range(15):
            asset = ImageAsset(
                path=f"/test/images/photo{i}.jpg",
                training_status="completed",
                width=640,
                height=480,
                file_size=102400,
                mime_type="image/jpeg",
            )
            db_session.add(asset)
            assets.append(asset)

        await db_session.flush()

        # Create identified persons
        person1 = Person(name="John Doe", status=PersonStatus.ACTIVE)
        person2 = Person(name="Jane Smith", status=PersonStatus.ACTIVE)
        db_session.add(person1)
        db_session.add(person2)
        await db_session.flush()

        # Create faces for identified persons
        # John Doe: 5 faces
        for i in range(5):
            face = FaceInstance(
                asset_id=assets[i].id,
                bbox_x=100,
                bbox_y=100,
                bbox_w=80,
                bbox_h=80,
                detection_confidence=0.95,
                quality_score=0.8,
                qdrant_point_id=uuid.uuid4(),
                person_id=person1.id,
            )
            db_session.add(face)

        # Jane Smith: 3 faces
        for i in range(5, 8):
            face = FaceInstance(
                asset_id=assets[i].id,
                bbox_x=100,
                bbox_y=100,
                bbox_w=80,
                bbox_h=80,
                detection_confidence=0.92,
                quality_score=0.75,
                qdrant_point_id=uuid.uuid4(),
                person_id=person2.id,
            )
            db_session.add(face)

        # Create unidentified clusters
        # Cluster 1: 4 faces
        for i in range(8, 12):
            face = FaceInstance(
                asset_id=assets[i].id,
                bbox_x=100,
                bbox_y=100,
                bbox_w=80,
                bbox_h=80,
                detection_confidence=0.90,
                quality_score=0.70,
                qdrant_point_id=uuid.uuid4(),
                cluster_id="cluster_abc",
                person_id=None,
            )
            db_session.add(face)

        # Cluster 2: 2 faces
        for i in range(12, 14):
            face = FaceInstance(
                asset_id=assets[i].id,
                bbox_x=100,
                bbox_y=100,
                bbox_w=80,
                bbox_h=80,
                detection_confidence=0.88,
                quality_score=0.65,
                qdrant_point_id=uuid.uuid4(),
                cluster_id="cluster_xyz",
                person_id=None,
            )
            db_session.add(face)

        # Noise faces: 1 face with cluster_id = '-1'
        face = FaceInstance(
            asset_id=assets[14].id,
            bbox_x=100,
            bbox_y=100,
            bbox_w=80,
            bbox_h=80,
            detection_confidence=0.85,
            quality_score=0.60,
            qdrant_point_id=uuid.uuid4(),
            cluster_id="-1",
            person_id=None,
        )
        db_session.add(face)

        await db_session.commit()

        return {
            "person1": person1,
            "person2": person2,
            "assets": assets,
        }

    @pytest.mark.asyncio
    async def test_list_people_returns_both_types(self, test_client, seed_test_data):
        """Should return both identified and unidentified people."""
        response = await test_client.get("/api/v1/faces/people")
        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "people" in data
        assert "total" in data
        assert "identifiedCount" in data  # camelCase
        assert "unidentifiedCount" in data
        assert "noiseCount" in data

        # Verify we have both types
        types = {p["type"] for p in data["people"]}
        assert "identified" in types
        assert "unidentified" in types

        # Verify counts (2 identified + 2 unidentified, no noise by default)
        assert data["identifiedCount"] == 2
        assert data["unidentifiedCount"] == 2
        assert data["noiseCount"] == 0
        assert data["total"] == 4

    @pytest.mark.asyncio
    async def test_filter_identified_only(self, test_client, seed_test_data):
        """Should filter to only identified people."""
        response = await test_client.get(
            "/api/v1/faces/people",
            params={"include_identified": True, "include_unidentified": False},
        )
        assert response.status_code == 200
        data = response.json()

        # All returned people should be identified
        for person in data["people"]:
            assert person["type"] == "identified"

        # Verify counts
        assert data["identifiedCount"] == 2
        assert data["unidentifiedCount"] == 0
        assert data["total"] == 2

        # Verify names match our test data
        names = {p["name"] for p in data["people"]}
        assert names == {"John Doe", "Jane Smith"}

    @pytest.mark.asyncio
    async def test_filter_unidentified_only(self, test_client, seed_test_data):
        """Should filter to only unidentified clusters."""
        response = await test_client.get(
            "/api/v1/faces/people",
            params={"include_identified": False, "include_unidentified": True},
        )
        assert response.status_code == 200
        data = response.json()

        # All returned people should be unidentified
        for person in data["people"]:
            assert person["type"] == "unidentified"

        # Verify counts
        assert data["identifiedCount"] == 0
        assert data["unidentifiedCount"] == 2
        assert data["total"] == 2

        # Verify cluster IDs
        cluster_ids = {p["id"] for p in data["people"]}
        assert cluster_ids == {"cluster_abc", "cluster_xyz"}

    @pytest.mark.asyncio
    async def test_include_noise(self, test_client, seed_test_data):
        """Should include noise faces when requested."""
        response = await test_client.get(
            "/api/v1/faces/people", params={"include_noise": True}
        )
        assert response.status_code == 200
        data = response.json()

        # Check if noise entry exists
        noise_entries = [p for p in data["people"] if p["type"] == "noise"]
        assert len(noise_entries) == 1

        noise_entry = noise_entries[0]
        assert noise_entry["id"] == "-1"
        assert noise_entry["name"] == "Unknown Faces"
        assert noise_entry["faceCount"] == 1  # We created 1 noise face

        # Verify counts
        assert data["noiseCount"] == 1

    @pytest.mark.asyncio
    async def test_exclude_noise_by_default(self, test_client, seed_test_data):
        """Should exclude noise faces by default."""
        response = await test_client.get("/api/v1/faces/people")
        assert response.status_code == 200
        data = response.json()

        # No noise entries should be present
        noise_entries = [p for p in data["people"] if p["type"] == "noise"]
        assert len(noise_entries) == 0
        assert data["noiseCount"] == 0

    @pytest.mark.asyncio
    async def test_sort_by_face_count_desc(self, test_client, seed_test_data):
        """Should sort by face count descending by default."""
        response = await test_client.get("/api/v1/faces/people")
        assert response.status_code == 200
        data = response.json()

        # Extract face counts
        face_counts = [p["faceCount"] for p in data["people"]]

        # Should be sorted descending (5, 4, 3, 2)
        assert face_counts == sorted(face_counts, reverse=True)

        # Verify first person has most faces
        assert data["people"][0]["faceCount"] == 5  # John Doe
        assert data["people"][0]["name"] == "John Doe"

    @pytest.mark.asyncio
    async def test_sort_by_face_count_asc(self, test_client, seed_test_data):
        """Should sort by face count ascending when requested."""
        response = await test_client.get(
            "/api/v1/faces/people", params={"sort_by": "face_count", "sort_order": "asc"}
        )
        assert response.status_code == 200
        data = response.json()

        # Extract face counts
        face_counts = [p["faceCount"] for p in data["people"]]

        # Should be sorted ascending (2, 3, 4, 5)
        assert face_counts == sorted(face_counts)

        # Verify first person has least faces
        assert data["people"][0]["faceCount"] == 2  # cluster_xyz

    @pytest.mark.asyncio
    async def test_sort_by_name_asc(self, test_client, seed_test_data):
        """Should sort by name ascending when requested."""
        response = await test_client.get(
            "/api/v1/faces/people", params={"sort_by": "name", "sort_order": "asc"}
        )
        assert response.status_code == 200
        data = response.json()

        # Extract names
        names = [p["name"] for p in data["people"]]

        # Should be sorted alphabetically (case-insensitive)
        assert names == sorted(names, key=str.lower)

        # First should be "Jane Smith" (alphabetically before "John Doe")
        assert data["people"][0]["name"] == "Jane Smith"

    @pytest.mark.asyncio
    async def test_sort_by_name_desc(self, test_client, seed_test_data):
        """Should sort by name descending when requested."""
        response = await test_client.get(
            "/api/v1/faces/people", params={"sort_by": "name", "sort_order": "desc"}
        )
        assert response.status_code == 200
        data = response.json()

        # Extract names
        names = [p["name"] for p in data["people"]]

        # Should be sorted reverse alphabetically
        assert names == sorted(names, key=str.lower, reverse=True)

        # First should be "Unidentified Person 1" or "Unidentified Person 2"
        assert data["people"][0]["name"].startswith("Unidentified Person")

    @pytest.mark.asyncio
    async def test_unidentified_person_naming(self, test_client, seed_test_data):
        """Should generate sequential names for unidentified people."""
        response = await test_client.get(
            "/api/v1/faces/people",
            params={"include_identified": False, "include_unidentified": True},
        )
        assert response.status_code == 200
        data = response.json()

        # All should be unidentified
        for person in data["people"]:
            assert person["type"] == "unidentified"
            assert person["name"].startswith("Unidentified Person ")

        # Extract numbers from names
        names = [p["name"] for p in data["people"]]
        # Should have sequential numbers (order depends on sorting)
        assert "Unidentified Person 1" in names
        assert "Unidentified Person 2" in names

    @pytest.mark.asyncio
    async def test_response_has_camel_case_keys(self, test_client, seed_test_data):
        """Response should use camelCase keys."""
        response = await test_client.get("/api/v1/faces/people")
        assert response.status_code == 200
        data = response.json()

        # Check top-level keys (camelCase)
        assert "identifiedCount" in data
        assert "unidentifiedCount" in data
        assert "noiseCount" in data

        # Check that snake_case versions don't exist
        assert "identified_count" not in data
        assert "unidentified_count" not in data
        assert "noise_count" not in data

        # Check person object keys
        if data["people"]:
            person = data["people"][0]
            assert "faceCount" in person  # not face_count
            assert "thumbnailUrl" in person or person.get("thumbnailUrl") is None

    @pytest.mark.asyncio
    async def test_empty_database(self, test_client):
        """Should handle empty database gracefully."""
        response = await test_client.get("/api/v1/faces/people")
        assert response.status_code == 200
        data = response.json()

        assert data["people"] == []
        assert data["total"] == 0
        assert data["identifiedCount"] == 0
        assert data["unidentifiedCount"] == 0
        assert data["noiseCount"] == 0

    @pytest.mark.asyncio
    async def test_thumbnail_urls_format(self, test_client, seed_test_data):
        """Should generate correct thumbnail URLs."""
        response = await test_client.get("/api/v1/faces/people")
        assert response.status_code == 200
        data = response.json()

        # Check that thumbnail URLs are properly formatted
        for person in data["people"]:
            if person.get("thumbnailUrl"):
                assert person["thumbnailUrl"].startswith("/api/v1/images/")
                assert person["thumbnailUrl"].endswith("/thumbnail")

    @pytest.mark.asyncio
    async def test_identified_person_has_uuid_id(self, test_client, seed_test_data):
        """Identified persons should have UUID as id."""
        response = await test_client.get(
            "/api/v1/faces/people",
            params={"include_identified": True, "include_unidentified": False},
        )
        assert response.status_code == 200
        data = response.json()

        for person in data["people"]:
            # ID should be a valid UUID string
            try:
                uuid.UUID(person["id"])
            except ValueError:
                pytest.fail(f"Person ID {person['id']} is not a valid UUID")

    @pytest.mark.asyncio
    async def test_unidentified_cluster_has_cluster_id(self, test_client, seed_test_data):
        """Unidentified clusters should have cluster_id as id."""
        response = await test_client.get(
            "/api/v1/faces/people",
            params={"include_identified": False, "include_unidentified": True},
        )
        assert response.status_code == 200
        data = response.json()

        for person in data["people"]:
            # ID should be cluster_id (string, not UUID)
            assert isinstance(person["id"], str)
            # Should match our test cluster IDs
            assert person["id"] in {"cluster_abc", "cluster_xyz"}

    @pytest.mark.asyncio
    async def test_face_count_accuracy(self, test_client, seed_test_data):
        """Face counts should match actual face instances."""
        response = await test_client.get("/api/v1/faces/people")
        assert response.status_code == 200
        data = response.json()

        # Verify specific face counts from our test data
        for person in data["people"]:
            if person["name"] == "John Doe":
                assert person["faceCount"] == 5
            elif person["name"] == "Jane Smith":
                assert person["faceCount"] == 3
            elif person["id"] == "cluster_abc":
                assert person["faceCount"] == 4
            elif person["id"] == "cluster_xyz":
                assert person["faceCount"] == 2

    @pytest.mark.asyncio
    async def test_confidence_field_for_unidentified(self, test_client, seed_test_data):
        """Unidentified clusters should have confidence (avg quality)."""
        response = await test_client.get(
            "/api/v1/faces/people",
            params={"include_identified": False, "include_unidentified": True},
        )
        assert response.status_code == 200
        data = response.json()

        for person in data["people"]:
            # Confidence should be present and be a number
            assert "confidence" in person
            if person["confidence"] is not None:
                assert isinstance(person["confidence"], (int, float))
                assert 0.0 <= person["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_confidence_null_for_identified(self, test_client, seed_test_data):
        """Identified persons should have null confidence."""
        response = await test_client.get(
            "/api/v1/faces/people",
            params={"include_identified": True, "include_unidentified": False},
        )
        assert response.status_code == 200
        data = response.json()

        for person in data["people"]:
            # Confidence should be null for identified persons
            assert person["confidence"] is None

    @pytest.mark.asyncio
    async def test_invalid_sort_by_parameter(self, test_client, seed_test_data):
        """Should reject invalid sort_by parameter."""
        response = await test_client.get(
            "/api/v1/faces/people", params={"sort_by": "invalid_field"}
        )
        # FastAPI should return 422 for invalid enum value
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_sort_order_parameter(self, test_client, seed_test_data):
        """Should reject invalid sort_order parameter."""
        response = await test_client.get(
            "/api/v1/faces/people", params={"sort_order": "invalid"}
        )
        # FastAPI should return 422 for invalid enum value
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_only_active_persons_included(self, test_client, db_session):
        """Should only include active persons, not merged or hidden."""
        # Create persons with different statuses
        active_person = Person(name="Active Person", status=PersonStatus.ACTIVE)
        merged_person = Person(name="Merged Person", status=PersonStatus.MERGED)
        hidden_person = Person(name="Hidden Person", status=PersonStatus.HIDDEN)

        db_session.add(active_person)
        db_session.add(merged_person)
        db_session.add(hidden_person)
        await db_session.commit()

        response = await test_client.get(
            "/api/v1/faces/people",
            params={"include_identified": True, "include_unidentified": False},
        )
        assert response.status_code == 200
        data = response.json()

        # Only active person should be returned
        names = {p["name"] for p in data["people"]}
        assert "Active Person" in names
        assert "Merged Person" not in names
        assert "Hidden Person" not in names

    @pytest.mark.asyncio
    async def test_all_filters_false_returns_empty(self, test_client, seed_test_data):
        """Should return empty when all filters are disabled."""
        response = await test_client.get(
            "/api/v1/faces/people",
            params={
                "include_identified": False,
                "include_unidentified": False,
                "include_noise": False,
            },
        )
        assert response.status_code == 200
        data = response.json()

        assert data["people"] == []
        assert data["total"] == 0
        assert data["identifiedCount"] == 0
        assert data["unidentifiedCount"] == 0
        assert data["noiseCount"] == 0
