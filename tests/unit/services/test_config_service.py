"""Unit tests for ConfigService (async) and SyncConfigService (sync).

Tests exercise the actual get/set operations, type coercion methods,
default fallback behavior, and validation logic that are NOT covered
by the structural tests in test_config_keys_sync.py and
test_config_unknown_person.py.
"""

import pytest
from sqlalchemy import select

from image_search_service.db.models import ConfigDataType, SystemConfig
from image_search_service.services.config_service import ConfigService, SyncConfigService

# ============ Helpers ============


def _make_config(
    key: str,
    value: str,
    data_type: str = ConfigDataType.FLOAT.value,
    category: str = "test",
    min_value: str | None = None,
    max_value: str | None = None,
    allowed_values: list[str] | None = None,
    description: str | None = None,
) -> SystemConfig:
    """Create a SystemConfig record for testing."""
    return SystemConfig(
        key=key,
        value=value,
        data_type=data_type,
        category=category,
        min_value=min_value,
        max_value=max_value,
        allowed_values=allowed_values,
        description=description,
    )


# ============ Async ConfigService Tests ============


class TestConfigServiceGet:
    """Tests for ConfigService.get_* methods reading from DB and defaults."""

    @pytest.mark.asyncio
    async def test_get_float_when_key_exists_in_db_then_returns_db_value(self, db_session) -> None:
        """get_float returns the stored database value when key exists."""
        config = _make_config("face_auto_assign_threshold", "0.92")
        db_session.add(config)
        await db_session.commit()

        service = ConfigService(db_session)
        result = await service.get_float("face_auto_assign_threshold")

        assert result == 0.92
        assert isinstance(result, float)

    @pytest.mark.asyncio
    async def test_get_float_when_key_missing_from_db_then_returns_default(
        self, db_session
    ) -> None:
        """get_float falls back to DEFAULTS when key is not in database."""
        service = ConfigService(db_session)
        result = await service.get_float("face_auto_assign_threshold")

        assert result == 0.85
        assert isinstance(result, float)

    @pytest.mark.asyncio
    async def test_get_float_when_key_unknown_then_raises_value_error(self, db_session) -> None:
        """get_float raises ValueError for keys not in DB or DEFAULTS."""
        service = ConfigService(db_session)

        with pytest.raises(ValueError, match="Unknown configuration key"):
            await service.get_float("totally_nonexistent_key")

    @pytest.mark.asyncio
    async def test_get_int_when_key_exists_in_db_then_returns_db_value(self, db_session) -> None:
        """get_int returns the stored database value cast to int."""
        config = _make_config(
            "face_suggestion_max_results",
            "75",
            data_type=ConfigDataType.INT.value,
        )
        db_session.add(config)
        await db_session.commit()

        service = ConfigService(db_session)
        result = await service.get_int("face_suggestion_max_results")

        assert result == 75
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_get_int_when_key_missing_from_db_then_returns_default(self, db_session) -> None:
        """get_int falls back to DEFAULTS when key is not in database."""
        service = ConfigService(db_session)
        result = await service.get_int("face_suggestion_max_results")

        assert result == 50
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_get_int_when_key_unknown_then_raises_value_error(self, db_session) -> None:
        """get_int raises ValueError for keys not in DB or DEFAULTS."""
        service = ConfigService(db_session)

        with pytest.raises(ValueError, match="Unknown configuration key"):
            await service.get_int("missing_int_key")

    @pytest.mark.asyncio
    async def test_get_string_when_key_exists_in_db_then_returns_db_value(self, db_session) -> None:
        """get_string returns the stored database value as string."""
        config = _make_config(
            "post_training_suggestions_mode",
            "top_n",
            data_type=ConfigDataType.STRING.value,
        )
        db_session.add(config)
        await db_session.commit()

        service = ConfigService(db_session)
        result = await service.get_string("post_training_suggestions_mode")

        assert result == "top_n"
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_get_string_when_key_missing_from_db_then_returns_default(
        self, db_session
    ) -> None:
        """get_string falls back to DEFAULTS when key is not in database."""
        service = ConfigService(db_session)
        result = await service.get_string("post_training_suggestions_mode")

        assert result == "all"
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_get_string_when_key_unknown_then_raises_value_error(self, db_session) -> None:
        """get_string raises ValueError for keys not in DB or DEFAULTS."""
        service = ConfigService(db_session)

        with pytest.raises(ValueError, match="Unknown configuration key"):
            await service.get_string("missing_string_key")

    @pytest.mark.asyncio
    async def test_get_float_db_value_overrides_default(self, db_session) -> None:
        """Database value takes precedence over DEFAULTS for the same key."""
        config = _make_config("face_suggestion_threshold", "0.55")
        db_session.add(config)
        await db_session.commit()

        service = ConfigService(db_session)
        result = await service.get_float("face_suggestion_threshold")

        # Default is 0.70, but DB value should win
        assert result == 0.55


class TestConfigServiceGetBool:
    """Tests for ConfigService.get_bool() coercion logic."""

    @pytest.mark.asyncio
    async def test_get_bool_when_value_is_true_string(self, db_session) -> None:
        """get_bool returns True for 'true' (case-insensitive)."""
        config = _make_config(
            "post_training_use_centroids",
            "true",
            data_type=ConfigDataType.BOOLEAN.value,
        )
        db_session.add(config)
        await db_session.commit()

        service = ConfigService(db_session)
        result = await service.get_bool("post_training_use_centroids")

        assert result is True

    @pytest.mark.asyncio
    async def test_get_bool_when_value_is_true_mixed_case(self, db_session) -> None:
        """get_bool returns True for 'True' (mixed case)."""
        config = _make_config(
            "post_training_use_centroids",
            "True",
            data_type=ConfigDataType.BOOLEAN.value,
        )
        db_session.add(config)
        await db_session.commit()

        service = ConfigService(db_session)
        result = await service.get_bool("post_training_use_centroids")

        assert result is True

    @pytest.mark.asyncio
    async def test_get_bool_when_value_is_1(self, db_session) -> None:
        """get_bool returns True for '1'."""
        config = _make_config(
            "post_training_use_centroids",
            "1",
            data_type=ConfigDataType.BOOLEAN.value,
        )
        db_session.add(config)
        await db_session.commit()

        service = ConfigService(db_session)
        result = await service.get_bool("post_training_use_centroids")

        assert result is True

    @pytest.mark.asyncio
    async def test_get_bool_when_value_is_yes(self, db_session) -> None:
        """get_bool returns True for 'yes'."""
        config = _make_config(
            "post_training_use_centroids",
            "yes",
            data_type=ConfigDataType.BOOLEAN.value,
        )
        db_session.add(config)
        await db_session.commit()

        service = ConfigService(db_session)
        result = await service.get_bool("post_training_use_centroids")

        assert result is True

    @pytest.mark.asyncio
    async def test_get_bool_when_value_is_false_string(self, db_session) -> None:
        """get_bool returns False for 'false'."""
        config = _make_config(
            "post_training_use_centroids",
            "false",
            data_type=ConfigDataType.BOOLEAN.value,
        )
        db_session.add(config)
        await db_session.commit()

        service = ConfigService(db_session)
        result = await service.get_bool("post_training_use_centroids")

        assert result is False

    @pytest.mark.asyncio
    async def test_get_bool_when_value_is_0(self, db_session) -> None:
        """get_bool returns False for '0'."""
        config = _make_config(
            "post_training_use_centroids",
            "0",
            data_type=ConfigDataType.BOOLEAN.value,
        )
        db_session.add(config)
        await db_session.commit()

        service = ConfigService(db_session)
        result = await service.get_bool("post_training_use_centroids")

        assert result is False

    @pytest.mark.asyncio
    async def test_get_bool_when_value_is_no(self, db_session) -> None:
        """get_bool returns False for 'no'."""
        config = _make_config(
            "post_training_use_centroids",
            "no",
            data_type=ConfigDataType.BOOLEAN.value,
        )
        db_session.add(config)
        await db_session.commit()

        service = ConfigService(db_session)
        result = await service.get_bool("post_training_use_centroids")

        assert result is False

    @pytest.mark.asyncio
    async def test_get_bool_when_value_is_unexpected_then_returns_false(self, db_session) -> None:
        """get_bool returns False for unexpected string values."""
        config = _make_config(
            "post_training_use_centroids",
            "maybe",
            data_type=ConfigDataType.BOOLEAN.value,
        )
        db_session.add(config)
        await db_session.commit()

        service = ConfigService(db_session)
        result = await service.get_bool("post_training_use_centroids")

        assert result is False

    @pytest.mark.asyncio
    async def test_get_bool_when_default_is_true_bool(self, db_session) -> None:
        """get_bool falls back to DEFAULTS and coerces True to 'True' then True."""
        # post_training_use_centroids default is True (bool)
        service = ConfigService(db_session)
        result = await service.get_bool("post_training_use_centroids")

        assert result is True


class TestConfigServiceGetIntCoercionErrors:
    """Tests for type coercion error handling in get_int and get_float."""

    @pytest.mark.asyncio
    async def test_get_int_when_value_is_non_numeric_then_raises(self, db_session) -> None:
        """get_int raises ValueError when DB value is not convertible to int."""
        config = _make_config(
            "face_suggestion_max_results",
            "not_a_number",
            data_type=ConfigDataType.INT.value,
        )
        db_session.add(config)
        await db_session.commit()

        service = ConfigService(db_session)

        with pytest.raises(ValueError):
            await service.get_int("face_suggestion_max_results")

    @pytest.mark.asyncio
    async def test_get_float_when_value_is_non_numeric_then_raises(self, db_session) -> None:
        """get_float raises ValueError when DB value is not convertible to float."""
        config = _make_config(
            "face_auto_assign_threshold",
            "not_a_float",
            data_type=ConfigDataType.FLOAT.value,
        )
        db_session.add(config)
        await db_session.commit()

        service = ConfigService(db_session)

        with pytest.raises(ValueError):
            await service.get_float("face_auto_assign_threshold")


class TestConfigServiceSetValue:
    """Tests for ConfigService.set_value() create/update behavior."""

    @pytest.mark.asyncio
    async def test_set_value_when_key_exists_then_updates(self, db_session) -> None:
        """set_value updates an existing config entry."""
        config = _make_config(
            "face_auto_assign_threshold",
            "0.85",
            min_value="0.5",
            max_value="1.0",
        )
        db_session.add(config)
        await db_session.commit()

        service = ConfigService(db_session)
        updated = await service.set_value("face_auto_assign_threshold", 0.90)

        assert updated.value == "0.9"
        assert updated.key == "face_auto_assign_threshold"

    @pytest.mark.asyncio
    async def test_set_value_when_key_not_in_db_then_raises_value_error(self, db_session) -> None:
        """set_value raises ValueError when key does not exist in database."""
        service = ConfigService(db_session)

        with pytest.raises(ValueError, match="Unknown configuration key"):
            await service.set_value("nonexistent_key", 42)

    @pytest.mark.asyncio
    async def test_set_value_persists_to_database(self, db_session) -> None:
        """set_value writes the new value to the database."""
        config = _make_config(
            "face_suggestion_threshold",
            "0.70",
            min_value="0.3",
            max_value="0.95",
        )
        db_session.add(config)
        await db_session.commit()

        service = ConfigService(db_session)
        await service.set_value("face_suggestion_threshold", 0.80)

        # Verify by re-reading from database
        query = select(SystemConfig).where(SystemConfig.key == "face_suggestion_threshold")
        result = await db_session.execute(query)
        row = result.scalar_one_or_none()

        assert row is not None
        assert row.value == "0.8"


class TestConfigServiceValidation:
    """Tests for ConfigService._validate_value() constraint checking."""

    @pytest.mark.asyncio
    async def test_set_value_when_float_below_min_then_raises(self, db_session) -> None:
        """set_value raises ValueError when float is below min_value constraint."""
        config = _make_config(
            "face_auto_assign_threshold",
            "0.85",
            min_value="0.5",
            max_value="1.0",
        )
        db_session.add(config)
        await db_session.commit()

        service = ConfigService(db_session)

        with pytest.raises(ValueError, match="must be >= 0.5"):
            await service.set_value("face_auto_assign_threshold", 0.3)

    @pytest.mark.asyncio
    async def test_set_value_when_float_above_max_then_raises(self, db_session) -> None:
        """set_value raises ValueError when float exceeds max_value constraint."""
        config = _make_config(
            "face_auto_assign_threshold",
            "0.85",
            min_value="0.5",
            max_value="1.0",
        )
        db_session.add(config)
        await db_session.commit()

        service = ConfigService(db_session)

        with pytest.raises(ValueError, match="must be <= 1.0"):
            await service.set_value("face_auto_assign_threshold", 1.5)

    @pytest.mark.asyncio
    async def test_set_value_when_int_type_gets_string_then_raises(self, db_session) -> None:
        """set_value raises ValueError for non-numeric value on int config."""
        config = _make_config(
            "face_suggestion_max_results",
            "50",
            data_type=ConfigDataType.INT.value,
            min_value="1",
            max_value="200",
        )
        db_session.add(config)
        await db_session.commit()

        service = ConfigService(db_session)

        with pytest.raises(ValueError, match="must be an integer"):
            await service.set_value("face_suggestion_max_results", "not_a_number")

    @pytest.mark.asyncio
    async def test_set_value_when_boolean_type_gets_non_bool_then_raises(self, db_session) -> None:
        """set_value raises ValueError for non-boolean value on boolean config."""
        config = _make_config(
            "post_training_use_centroids",
            "true",
            data_type=ConfigDataType.BOOLEAN.value,
        )
        db_session.add(config)
        await db_session.commit()

        service = ConfigService(db_session)

        with pytest.raises(ValueError, match="must be a boolean"):
            await service.set_value("post_training_use_centroids", "true")

    @pytest.mark.asyncio
    async def test_set_value_when_allowed_values_violated_then_raises(self, db_session) -> None:
        """set_value raises ValueError when value is not in allowed_values list."""
        config = _make_config(
            "post_training_suggestions_mode",
            "all",
            data_type=ConfigDataType.STRING.value,
            allowed_values=["all", "top_n", "none"],
        )
        db_session.add(config)
        await db_session.commit()

        service = ConfigService(db_session)

        with pytest.raises(ValueError, match="must be one of"):
            await service.set_value("post_training_suggestions_mode", "invalid_mode")

    @pytest.mark.asyncio
    async def test_set_value_when_allowed_values_satisfied_then_succeeds(self, db_session) -> None:
        """set_value succeeds when value is in allowed_values list."""
        config = _make_config(
            "post_training_suggestions_mode",
            "all",
            data_type=ConfigDataType.STRING.value,
            allowed_values=["all", "top_n", "none"],
        )
        db_session.add(config)
        await db_session.commit()

        service = ConfigService(db_session)
        updated = await service.set_value("post_training_suggestions_mode", "top_n")

        assert updated.value == "top_n"


class TestConfigServiceGetAllByCategory:
    """Tests for ConfigService.get_all_by_category()."""

    @pytest.mark.asyncio
    async def test_get_all_by_category_returns_matching_records(self, db_session) -> None:
        """get_all_by_category returns only records with matching category."""
        c1 = _make_config("key_a", "1.0", category="face_matching")
        c2 = _make_config("key_b", "2.0", category="face_matching")
        c3 = _make_config("key_c", "3.0", category="other_category")
        db_session.add_all([c1, c2, c3])
        await db_session.commit()

        service = ConfigService(db_session)
        results = await service.get_all_by_category("face_matching")

        assert len(results) == 2
        keys = {r.key for r in results}
        assert keys == {"key_a", "key_b"}

    @pytest.mark.asyncio
    async def test_get_all_by_category_when_none_match_then_returns_empty(self, db_session) -> None:
        """get_all_by_category returns empty list when no records match."""
        config = _make_config("key_a", "1.0", category="face_matching")
        db_session.add(config)
        await db_session.commit()

        service = ConfigService(db_session)
        results = await service.get_all_by_category("nonexistent_category")

        assert results == []


class TestConfigServiceEdgeCases:
    """Tests for edge cases: empty strings, special characters, long values."""

    @pytest.mark.asyncio
    async def test_get_string_when_empty_string_value(self, db_session) -> None:
        """get_string returns empty string when DB value is empty."""
        config = _make_config(
            "test_empty_key",
            "",
            data_type=ConfigDataType.STRING.value,
        )
        db_session.add(config)
        await db_session.commit()

        service = ConfigService(db_session)
        result = await service.get_string("test_empty_key")

        assert result == ""

    @pytest.mark.asyncio
    async def test_get_string_when_value_has_special_characters(self, db_session) -> None:
        """get_string handles special characters in values correctly."""
        config = _make_config(
            "test_special_key",
            "value with spaces & symbols! @#$%",
            data_type=ConfigDataType.STRING.value,
        )
        db_session.add(config)
        await db_session.commit()

        service = ConfigService(db_session)
        result = await service.get_string("test_special_key")

        assert result == "value with spaces & symbols! @#$%"

    @pytest.mark.asyncio
    async def test_get_string_when_value_is_very_long(self, db_session) -> None:
        """get_string handles long values (up to column limit)."""
        long_value = "x" * 499  # Just under the 500-char column limit
        config = _make_config(
            "test_long_key",
            long_value,
            data_type=ConfigDataType.STRING.value,
        )
        db_session.add(config)
        await db_session.commit()

        service = ConfigService(db_session)
        result = await service.get_string("test_long_key")

        assert result == long_value
        assert len(result) == 499

    @pytest.mark.asyncio
    async def test_get_bool_when_empty_string_then_returns_false(self, db_session) -> None:
        """get_bool returns False for empty string value."""
        config = _make_config(
            "test_bool_empty",
            "",
            data_type=ConfigDataType.BOOLEAN.value,
        )
        db_session.add(config)
        await db_session.commit()

        service = ConfigService(db_session)
        result = await service.get_bool("test_bool_empty")

        assert result is False


class TestConfigServiceDefaults:
    """Tests verifying DEFAULTS are used as fallback when DB has no entry."""

    @pytest.mark.asyncio
    async def test_all_float_defaults_are_accessible(self, db_session) -> None:
        """All float keys in DEFAULTS are retrievable via get_float without DB."""
        service = ConfigService(db_session)

        float_keys = [
            "face_auto_assign_threshold",
            "face_suggestion_threshold",
            "face_prototype_min_quality",
            "unknown_person_default_threshold",
        ]

        for key in float_keys:
            result = await service.get_float(key)
            expected = float(ConfigService.DEFAULTS[key])
            assert result == expected, f"Default for {key} was {result}, expected {expected}"

    @pytest.mark.asyncio
    async def test_all_int_defaults_are_accessible(self, db_session) -> None:
        """All int keys in DEFAULTS are retrievable via get_int without DB."""
        service = ConfigService(db_session)

        int_keys = [
            "face_suggestion_max_results",
            "face_suggestion_expiry_days",
            "face_prototype_max_exemplars",
            "face_suggestion_groups_per_page",
            "face_suggestion_items_per_group",
            "post_training_suggestions_top_n_count",
            "centroid_min_faces_for_suggestions",
            "unknown_person_min_display_count",
            "unknown_person_max_faces",
            "unknown_person_chunk_size",
        ]

        for key in int_keys:
            result = await service.get_int(key)
            expected = int(ConfigService.DEFAULTS[key])
            assert result == expected, f"Default for {key} was {result}, expected {expected}"


# ============ Sync ConfigService Tests ============


class TestSyncConfigServiceGet:
    """Tests for SyncConfigService get methods using sync_db_session."""

    def test_get_float_when_key_exists_in_db_then_returns_db_value(self, sync_db_session) -> None:
        """SyncConfigService.get_float returns stored DB value."""
        config = _make_config("face_auto_assign_threshold", "0.92")
        sync_db_session.add(config)
        sync_db_session.commit()

        service = SyncConfigService(sync_db_session)
        result = service.get_float("face_auto_assign_threshold")

        assert result == 0.92
        assert isinstance(result, float)

    def test_get_float_when_key_missing_from_db_then_returns_default(self, sync_db_session) -> None:
        """SyncConfigService.get_float falls back to DEFAULTS."""
        service = SyncConfigService(sync_db_session)
        result = service.get_float("face_auto_assign_threshold")

        assert result == 0.85

    def test_get_float_when_key_unknown_then_raises_value_error(self, sync_db_session) -> None:
        """SyncConfigService.get_float raises ValueError for unknown key."""
        service = SyncConfigService(sync_db_session)

        with pytest.raises(ValueError, match="Unknown configuration key"):
            service.get_float("nonexistent_float_key")

    def test_get_int_when_key_exists_in_db_then_returns_db_value(self, sync_db_session) -> None:
        """SyncConfigService.get_int returns stored DB value."""
        config = _make_config(
            "face_suggestion_max_results",
            "100",
            data_type=ConfigDataType.INT.value,
        )
        sync_db_session.add(config)
        sync_db_session.commit()

        service = SyncConfigService(sync_db_session)
        result = service.get_int("face_suggestion_max_results")

        assert result == 100
        assert isinstance(result, int)

    def test_get_int_when_key_missing_from_db_then_returns_default(self, sync_db_session) -> None:
        """SyncConfigService.get_int falls back to DEFAULTS."""
        service = SyncConfigService(sync_db_session)
        result = service.get_int("face_suggestion_max_results")

        assert result == 50

    def test_get_int_when_key_unknown_then_raises_value_error(self, sync_db_session) -> None:
        """SyncConfigService.get_int raises ValueError for unknown key."""
        service = SyncConfigService(sync_db_session)

        with pytest.raises(ValueError, match="Unknown configuration key"):
            service.get_int("nonexistent_int_key")

    def test_get_string_when_key_exists_in_db_then_returns_db_value(self, sync_db_session) -> None:
        """SyncConfigService.get_string returns stored DB value."""
        config = _make_config(
            "post_training_suggestions_mode",
            "top_n",
            data_type=ConfigDataType.STRING.value,
        )
        sync_db_session.add(config)
        sync_db_session.commit()

        service = SyncConfigService(sync_db_session)
        result = service.get_string("post_training_suggestions_mode")

        assert result == "top_n"

    def test_get_string_when_key_missing_from_db_then_returns_default(
        self, sync_db_session
    ) -> None:
        """SyncConfigService.get_string falls back to DEFAULTS."""
        service = SyncConfigService(sync_db_session)
        result = service.get_string("post_training_suggestions_mode")

        assert result == "all"

    def test_get_string_when_key_unknown_then_raises_value_error(self, sync_db_session) -> None:
        """SyncConfigService.get_string raises ValueError for unknown key."""
        service = SyncConfigService(sync_db_session)

        with pytest.raises(ValueError, match="Unknown configuration key"):
            service.get_string("nonexistent_string_key")


class TestSyncConfigServiceGetBool:
    """Tests for SyncConfigService.get_bool() coercion logic."""

    def test_get_bool_when_value_is_true_then_returns_true(self, sync_db_session) -> None:
        """SyncConfigService.get_bool returns True for 'true'."""
        config = _make_config(
            "post_training_use_centroids",
            "true",
            data_type=ConfigDataType.BOOLEAN.value,
        )
        sync_db_session.add(config)
        sync_db_session.commit()

        service = SyncConfigService(sync_db_session)
        result = service.get_bool("post_training_use_centroids")

        assert result is True

    def test_get_bool_when_value_is_1_then_returns_true(self, sync_db_session) -> None:
        """SyncConfigService.get_bool returns True for '1'."""
        config = _make_config(
            "post_training_use_centroids",
            "1",
            data_type=ConfigDataType.BOOLEAN.value,
        )
        sync_db_session.add(config)
        sync_db_session.commit()

        service = SyncConfigService(sync_db_session)
        result = service.get_bool("post_training_use_centroids")

        assert result is True

    def test_get_bool_when_value_is_yes_then_returns_true(self, sync_db_session) -> None:
        """SyncConfigService.get_bool returns True for 'yes'."""
        config = _make_config(
            "post_training_use_centroids",
            "yes",
            data_type=ConfigDataType.BOOLEAN.value,
        )
        sync_db_session.add(config)
        sync_db_session.commit()

        service = SyncConfigService(sync_db_session)
        result = service.get_bool("post_training_use_centroids")

        assert result is True

    def test_get_bool_when_value_is_false_then_returns_false(self, sync_db_session) -> None:
        """SyncConfigService.get_bool returns False for 'false'."""
        config = _make_config(
            "post_training_use_centroids",
            "false",
            data_type=ConfigDataType.BOOLEAN.value,
        )
        sync_db_session.add(config)
        sync_db_session.commit()

        service = SyncConfigService(sync_db_session)
        result = service.get_bool("post_training_use_centroids")

        assert result is False

    def test_get_bool_when_value_is_0_then_returns_false(self, sync_db_session) -> None:
        """SyncConfigService.get_bool returns False for '0'."""
        config = _make_config(
            "post_training_use_centroids",
            "0",
            data_type=ConfigDataType.BOOLEAN.value,
        )
        sync_db_session.add(config)
        sync_db_session.commit()

        service = SyncConfigService(sync_db_session)
        result = service.get_bool("post_training_use_centroids")

        assert result is False

    def test_get_bool_when_value_unexpected_then_returns_false(self, sync_db_session) -> None:
        """SyncConfigService.get_bool returns False for unexpected values."""
        config = _make_config(
            "post_training_use_centroids",
            "banana",
            data_type=ConfigDataType.BOOLEAN.value,
        )
        sync_db_session.add(config)
        sync_db_session.commit()

        service = SyncConfigService(sync_db_session)
        result = service.get_bool("post_training_use_centroids")

        assert result is False

    def test_get_bool_when_default_is_true_bool_then_coerces_correctly(
        self, sync_db_session
    ) -> None:
        """SyncConfigService.get_bool correctly coerces True default to True."""
        # post_training_use_centroids default is True (Python bool)
        service = SyncConfigService(sync_db_session)
        result = service.get_bool("post_training_use_centroids")

        assert result is True


class TestSyncConfigServiceSharesDefaults:
    """Tests that SyncConfigService shares the same DEFAULTS as ConfigService."""

    def test_sync_defaults_are_same_object_as_async_defaults(self) -> None:
        """SyncConfigService.DEFAULTS is the same dict as ConfigService.DEFAULTS."""
        assert SyncConfigService.DEFAULTS is ConfigService.DEFAULTS

    def test_sync_defaults_contain_all_keys(self) -> None:
        """SyncConfigService.DEFAULTS has all keys from ConfigService.DEFAULTS."""
        assert set(SyncConfigService.DEFAULTS.keys()) == set(ConfigService.DEFAULTS.keys())
