"""Regression tests for configuration key synchronization.

These tests prevent the recurring issue where ConfigService.DEFAULTS keys are not
seeded in the database, causing 400 "Unknown configuration key" errors.

**The Problem**: Configuration keys defined in `ConfigService.DEFAULTS` must also be
seeded in the `system_config` database table via migrations. When they're missing,
PUT config endpoints fail with 400 errors.

**Test Coverage**:
1. DEFAULTS completeness: All config keys used by endpoints exist in DEFAULTS
2. Migration coverage: All DEFAULTS keys have corresponding INSERTs in migrations
3. Endpoint-to-DEFAULTS mapping: Endpoint keys match DEFAULTS keys
"""

import re
from pathlib import Path
from typing import Any

import pytest

from image_search_service.services.config_service import ConfigService

# ============ Test Configuration Keys in DEFAULTS ============


def test_defaults_contains_all_face_matching_keys() -> None:
    """Test that DEFAULTS contains all face matching configuration keys.

    Verifies that all keys used by the /face-matching endpoint are defined
    in ConfigService.DEFAULTS to prevent 400 errors.
    """
    required_keys = {
        "face_auto_assign_threshold",
        "face_suggestion_threshold",
        "face_suggestion_max_results",
        "face_suggestion_expiry_days",
        "face_prototype_min_quality",
        "face_prototype_max_exemplars",
    }

    defaults_keys = set(ConfigService.DEFAULTS.keys())

    missing_keys = required_keys - defaults_keys
    assert not missing_keys, (
        f"Missing config keys in ConfigService.DEFAULTS: {missing_keys}. "
        "These keys are required by the /face-matching endpoint."
    )


def test_defaults_contains_all_face_suggestion_settings_keys() -> None:
    """Test that DEFAULTS contains all face suggestion pagination keys.

    Verifies that all keys used by the /face-suggestions endpoint are defined
    in ConfigService.DEFAULTS.
    """
    required_keys = {
        "face_suggestion_groups_per_page",
        "face_suggestion_items_per_group",
    }

    defaults_keys = set(ConfigService.DEFAULTS.keys())

    missing_keys = required_keys - defaults_keys
    assert not missing_keys, (
        f"Missing config keys in ConfigService.DEFAULTS: {missing_keys}. "
        "These keys are required by the /face-suggestions endpoint."
    )


# ============ Test Migration Coverage ============


def _find_migrations_dir() -> Path:
    """Find the migrations directory in the project.

    Returns:
        Path to migrations/versions directory

    Raises:
        FileNotFoundError: If migrations directory not found
    """
    # Start from this test file and search upward
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        migrations_dir = parent / "src" / "image_search_service" / "db" / "migrations" / "versions"
        if migrations_dir.exists():
            return migrations_dir

    raise FileNotFoundError(
        "Could not find migrations directory. Expected at "
        "src/image_search_service/db/migrations/versions/"
    )


def _parse_migration_inserts(migration_file: Path) -> set[str]:
    """Parse INSERT statements from a migration file to extract config keys.

    Args:
        migration_file: Path to migration file

    Returns:
        Set of configuration keys found in INSERT statements
    """
    content = migration_file.read_text()

    # Find all INSERT INTO system_configs statements
    # Matches multi-line INSERT statements with VALUES
    insert_pattern = re.compile(
        r"INSERT INTO system_configs.*?VALUES(.*?)(?:ON CONFLICT|;|\)[\s]*$)",
        re.DOTALL | re.IGNORECASE,
    )

    keys = set()
    for match in insert_pattern.finditer(content):
        values_section = match.group(1)

        # Extract key values from tuples like ('key_name', 'value', ...)
        # Match quoted strings that look like config keys (contain underscores)
        key_pattern = re.compile(r"'([a-z_]+)'")
        potential_keys = key_pattern.findall(values_section)

        # Filter to only keys that match our naming convention
        # (start with face_, contain underscores, are likely config keys)
        for key in potential_keys:
            # Skip data type values like 'float', 'int', 'face_matching' category
            if key in ("float", "int", "boolean", "string", "face_matching", "general"):
                continue
            # Only include keys that start with our known prefixes
            if key.startswith("face_"):
                keys.add(key)

    return keys


def test_all_defaults_have_migration_inserts() -> None:
    """Test that every key in DEFAULTS has a corresponding INSERT in migrations.

    This prevents the bug where new keys are added to DEFAULTS but not seeded
    in the database, causing 400 errors when endpoints try to update them.

    **Critical**: When adding new config keys:
    1. Add to ConfigService.DEFAULTS
    2. Create migration with INSERT statement
    3. This test will verify both exist
    """
    migrations_dir = _find_migrations_dir()

    # Parse all migration files to find INSERT statements
    all_migrated_keys = set()
    migration_files = sorted(migrations_dir.glob("*.py"))

    assert migration_files, (
        f"No migration files found in {migrations_dir}. "
        "This test requires migration files to verify database seeding."
    )

    for migration_file in migration_files:
        keys = _parse_migration_inserts(migration_file)
        all_migrated_keys.update(keys)

    # Compare against DEFAULTS
    defaults_keys = set(ConfigService.DEFAULTS.keys())

    # Keys in DEFAULTS but not in any migration
    missing_migrations = defaults_keys - all_migrated_keys

    assert not missing_migrations, (
        f"Config keys in DEFAULTS missing database migrations: {missing_migrations}\n\n"
        "These keys are defined in ConfigService.DEFAULTS but have no INSERT statement "
        "in any migration file. This causes 400 'Unknown configuration key' errors.\n\n"
        "To fix:\n"
        "1. Create a new migration: make makemigrations\n"
        "2. Add INSERT INTO system_configs statement for these keys\n"
        "3. Follow the pattern in a1b2c3d4e5f6_add_system_configs_table.py\n\n"
        f"Expected migration directory: {migrations_dir}"
    )


def test_no_orphaned_migrations() -> None:
    """Test that all migrated config keys are still in DEFAULTS.

    This catches cleanup issues where migrations exist but keys are removed
    from DEFAULTS without cleanup migrations.
    """
    migrations_dir = _find_migrations_dir()

    # Parse all migration files
    all_migrated_keys = set()
    for migration_file in sorted(migrations_dir.glob("*.py")):
        keys = _parse_migration_inserts(migration_file)
        all_migrated_keys.update(keys)

    defaults_keys = set(ConfigService.DEFAULTS.keys())

    # Keys in migrations but not in DEFAULTS (potential orphans)
    orphaned_keys = all_migrated_keys - defaults_keys

    # This is a warning, not an error - migrations might seed keys no longer in use
    if orphaned_keys:
        pytest.fail(
            f"Config keys in migrations but not in DEFAULTS: {orphaned_keys}\n\n"
            "These keys are seeded in database migrations but not defined in "
            "ConfigService.DEFAULTS. This might indicate:\n"
            "1. Removed keys that should have cleanup migrations\n"
            "2. Keys that should still be in DEFAULTS\n"
            "3. Legacy keys that are no longer used\n\n"
            "Consider:\n"
            "- Adding these keys back to DEFAULTS if still needed\n"
            "- Creating cleanup migration to remove from database if obsolete"
        )


# ============ Test Endpoint-to-DEFAULTS Mapping ============


def test_face_matching_endpoint_keys_match_defaults() -> None:
    """Test that /face-matching endpoint keys exactly match DEFAULTS.

    This ensures the endpoint can read/write all its configuration values
    without encountering missing key errors.

    **Based on**: api/routes/config.py::get_face_matching_config()
    """
    # Keys used by GET /api/v1/config/face-matching
    endpoint_keys = {
        "face_auto_assign_threshold",  # Line 132
        "face_suggestion_threshold",  # Line 133
        "face_suggestion_max_results",  # Line 134
        "face_suggestion_expiry_days",  # Line 135
        "face_prototype_min_quality",  # Line 136
        "face_prototype_max_exemplars",  # Line 137
    }

    defaults_keys = set(ConfigService.DEFAULTS.keys())

    # All endpoint keys must be in DEFAULTS
    missing = endpoint_keys - defaults_keys
    assert not missing, (
        f"Face matching endpoint uses keys not in DEFAULTS: {missing}. "
        "GET /api/v1/config/face-matching will fail with 400 error."
    )


def test_face_suggestions_endpoint_keys_match_defaults() -> None:
    """Test that /face-suggestions endpoint keys exactly match DEFAULTS.

    **Based on**: api/routes/config.py::get_face_suggestion_settings()
    """
    # Keys used by GET /api/v1/config/face-suggestions
    endpoint_keys = {
        "face_suggestion_groups_per_page",  # Line 202
        "face_suggestion_items_per_group",  # Line 203
    }

    defaults_keys = set(ConfigService.DEFAULTS.keys())

    missing = endpoint_keys - defaults_keys
    assert not missing, (
        f"Face suggestions endpoint uses keys not in DEFAULTS: {missing}. "
        "GET /api/v1/config/face-suggestions will fail with 400 error."
    )


def test_defaults_data_types_are_correct() -> None:
    """Test that DEFAULTS values match their expected types.

    This prevents runtime type errors when config values are cast.
    """
    expected_types: dict[str, type[Any]] = {
        "face_auto_assign_threshold": float,
        "face_suggestion_threshold": float,
        "face_suggestion_max_results": int,
        "face_suggestion_expiry_days": int,
        "face_prototype_min_quality": float,
        "face_prototype_max_exemplars": int,
        "face_suggestion_groups_per_page": int,
        "face_suggestion_items_per_group": int,
    }

    for key, expected_type in expected_types.items():
        value = ConfigService.DEFAULTS.get(key)
        assert value is not None, f"Key '{key}' missing from DEFAULTS"

        assert isinstance(value, expected_type), (
            f"DEFAULTS['{key}'] has type {type(value).__name__}, "
            f"expected {expected_type.__name__}. "
            f"Value: {value}"
        )


def test_defaults_values_are_in_valid_ranges() -> None:
    """Test that DEFAULTS values are within documented constraint ranges.

    This prevents migration issues where default values violate constraints.
    """
    validations = [
        # (key, min_value, max_value)
        ("face_auto_assign_threshold", 0.5, 1.0),
        ("face_suggestion_threshold", 0.3, 0.95),
        ("face_suggestion_max_results", 1, 200),
        ("face_suggestion_expiry_days", 1, 365),
        ("face_prototype_min_quality", 0.0, 1.0),
        ("face_prototype_max_exemplars", 1, 20),
        ("face_suggestion_groups_per_page", 1, 50),
        ("face_suggestion_items_per_group", 1, 50),
    ]

    for key, min_val, max_val in validations:
        value = ConfigService.DEFAULTS.get(key)
        assert value is not None, f"Key '{key}' missing from DEFAULTS"

        # Convert to float for comparison (handles both int and float)
        numeric_value = float(value)

        assert min_val <= numeric_value <= max_val, (
            f"DEFAULTS['{key}'] = {value} is outside valid range "
            f"[{min_val}, {max_val}]. Update DEFAULTS or migration constraints."
        )


def test_suggestion_threshold_less_than_auto_assign_threshold() -> None:
    """Test business logic constraint: suggestion threshold < auto-assign threshold.

    This constraint is validated in the API but should also hold in DEFAULTS.
    """
    suggestion = ConfigService.DEFAULTS.get("face_suggestion_threshold")
    auto_assign = ConfigService.DEFAULTS.get("face_auto_assign_threshold")

    assert suggestion is not None and auto_assign is not None, "Missing threshold keys in DEFAULTS"

    # Type narrowing: we know these are floats from previous tests
    assert isinstance(suggestion, (int, float)) and isinstance(
        auto_assign, (int, float)
    ), "Threshold values must be numeric"

    assert float(suggestion) < float(auto_assign), (
        f"DEFAULTS violates business constraint: "
        f"face_suggestion_threshold ({suggestion}) must be < "
        f"face_auto_assign_threshold ({auto_assign})"
    )


# ============ Documentation Tests ============


def test_all_defaults_keys_are_documented_in_migration() -> None:
    """Test that every DEFAULTS key has description in migration.

    This is a code quality check - all config keys should be documented
    in their migration INSERT statements.
    """
    migrations_dir = _find_migrations_dir()

    # Build map of keys to their migrations with descriptions
    keys_with_descriptions = set()

    for migration_file in sorted(migrations_dir.glob("*.py")):
        content = migration_file.read_text()

        # Look for INSERT statements with description field
        # Pattern: 'key_name', 'value', 'type', 'Some description'
        if "system_configs" not in content:
            continue

        # Extract all config keys from this migration
        insert_pattern = re.compile(
            r"INSERT INTO system_configs.*?VALUES(.*?)(?:ON CONFLICT|;|\)[\s]*$)",
            re.DOTALL | re.IGNORECASE,
        )

        for match in insert_pattern.finditer(content):
            values_section = match.group(1)

            # Find tuples with description field
            # Format: ('key', 'value', 'type', 'description', ...)
            tuple_pattern = re.compile(
                r"\(\s*'([a-z_]+)'[^)]*?,\s*'[^']*?'\s*,\s*'[^']*?'\s*,\s*'([^']+)'",
                re.DOTALL,
            )

            for tuple_match in tuple_pattern.finditer(values_section):
                key = tuple_match.group(1)
                description = tuple_match.group(2)

                # Only count if it's a real config key with a description
                if key.startswith("face_") and len(description) > 10:
                    keys_with_descriptions.add(key)

    defaults_keys = set(ConfigService.DEFAULTS.keys())

    # Keys in DEFAULTS but missing descriptions in migrations
    undocumented = defaults_keys - keys_with_descriptions

    assert not undocumented, (
        f"Config keys in DEFAULTS without migration descriptions: {undocumented}\n\n"
        "All configuration keys should have descriptions in their migration INSERT "
        "statements. This helps users understand what each setting controls.\n\n"
        "To fix: Add description field to INSERT statement in migration."
    )
