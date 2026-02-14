"""Tests for unknown person schema validation."""

from uuid import uuid4

import pytest

from image_search_service.api.face_schemas import LabelClusterRequest
from image_search_service.api.unknown_person_schemas import AcceptUnknownPersonRequest


class TestAcceptUnknownPersonRequestValidation:
    """Tests for AcceptUnknownPersonRequest mutual exclusion validator."""

    def test_name_only_valid(self):
        """Original flow: provide name to create/find person."""
        req = AcceptUnknownPersonRequest(name="Alice")
        assert req.name == "Alice"
        assert req.person_id is None

    def test_person_id_only_valid(self):
        """New flow: provide person_id to assign to existing person."""
        pid = uuid4()
        req = AcceptUnknownPersonRequest(person_id=pid)
        assert req.person_id == pid
        assert req.name is None

    def test_both_name_and_person_id_raises(self):
        """Cannot provide both name and person_id."""
        with pytest.raises(ValueError, match="Provide either name or personId, not both"):
            AcceptUnknownPersonRequest(name="Alice", person_id=uuid4())

    def test_neither_name_nor_person_id_raises(self):
        """Must provide at least one."""
        with pytest.raises(ValueError, match="Either name or personId must be provided"):
            AcceptUnknownPersonRequest()

    def test_name_with_exclusions_valid(self):
        """Name flow with face exclusions still works."""
        req = AcceptUnknownPersonRequest(
            name="Bob",
            face_ids_to_exclude=[uuid4(), uuid4()],
            trigger_reclustering=False,
        )
        assert req.name == "Bob"
        assert len(req.face_ids_to_exclude) == 2
        assert req.trigger_reclustering is False

    def test_person_id_with_exclusions_valid(self):
        """Person ID flow with face exclusions works."""
        pid = uuid4()
        req = AcceptUnknownPersonRequest(
            person_id=pid,
            face_ids_to_exclude=[uuid4()],
        )
        assert req.person_id == pid

    def test_empty_name_raises(self):
        """Empty string name should fail min_length validation."""
        with pytest.raises(ValueError):
            AcceptUnknownPersonRequest(name="")

    def test_camel_case_alias(self):
        """Verify JSON alias works for personId."""
        pid = uuid4()
        req = AcceptUnknownPersonRequest.model_validate({"personId": str(pid)})
        assert req.person_id == pid


class TestLabelClusterRequestValidation:
    """Tests for LabelClusterRequest mutual exclusion validator."""

    def test_name_only_valid(self):
        """Provide name to create/find person."""
        req = LabelClusterRequest(name="Alice")
        assert req.name == "Alice"
        assert req.person_id is None

    def test_person_id_only_valid(self):
        """Provide person_id to assign to existing person."""
        pid = uuid4()
        req = LabelClusterRequest(person_id=pid)
        assert req.person_id == pid
        assert req.name is None

    def test_both_raises(self):
        """Cannot provide both name and person_id."""
        with pytest.raises(ValueError, match="Provide either name or personId, not both"):
            LabelClusterRequest(name="Alice", person_id=uuid4())

    def test_neither_raises(self):
        """Must provide at least one."""
        with pytest.raises(ValueError, match="Either name or personId must be provided"):
            LabelClusterRequest()
