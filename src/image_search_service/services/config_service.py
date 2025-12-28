"""Database-backed configuration service.

Provides both async (for API endpoints) and sync (for background workers)
interfaces to read and write system configuration values.
"""

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session as SyncSession

from image_search_service.db.models import ConfigDataType, SystemConfig

logger = logging.getLogger(__name__)


class ConfigService:
    """Async configuration service for API endpoints."""

    # Default values (fallback when DB unavailable or key missing)
    DEFAULTS: dict[str, float | int | str | bool] = {
        "face_auto_assign_threshold": 0.85,
        "face_suggestion_threshold": 0.70,
        "face_suggestion_max_results": 50,
        "face_suggestion_expiry_days": 30,
        "face_prototype_min_quality": 0.5,
        "face_prototype_max_exemplars": 5,
    }

    def __init__(self, db_session: AsyncSession):
        """Initialize config service with async database session.

        Args:
            db_session: Async SQLAlchemy session for database operations
        """
        self.db = db_session

    async def get_float(self, key: str) -> float:
        """Get a float configuration value.

        Args:
            key: Configuration key name

        Returns:
            Float value from database or default
        """
        return await self._get_value(key, float)

    async def get_int(self, key: str) -> int:
        """Get an integer configuration value.

        Args:
            key: Configuration key name

        Returns:
            Integer value from database or default
        """
        return await self._get_value(key, int)

    async def get_string(self, key: str) -> str:
        """Get a string configuration value.

        Args:
            key: Configuration key name

        Returns:
            String value from database or default
        """
        return await self._get_value(key, str)

    async def get_bool(self, key: str) -> bool:
        """Get a boolean configuration value.

        Args:
            key: Configuration key name

        Returns:
            Boolean value from database or default
        """
        value = await self._get_value(key, str)
        return str(value).lower() in ("true", "1", "yes")

    async def _get_value(self, key: str, cast_type: type) -> Any:
        """Get and cast a configuration value.

        Args:
            key: Configuration key name
            cast_type: Type to cast the value to

        Returns:
            Configuration value cast to the specified type

        Raises:
            ValueError: If key is unknown and has no default
        """
        config = await self._get_config(key)
        if config:
            return cast_type(config.value)
        default = self.DEFAULTS.get(key)
        if default is not None:
            return cast_type(default)
        raise ValueError(f"Unknown configuration key: {key}")

    async def _get_config(self, key: str) -> SystemConfig | None:
        """Retrieve config record from database.

        Args:
            key: Configuration key name

        Returns:
            SystemConfig record or None if not found
        """
        query = select(SystemConfig).where(SystemConfig.key == key)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def set_value(self, key: str, value: Any) -> SystemConfig:
        """Set a configuration value with validation.

        Args:
            key: Configuration key name
            value: New value to set

        Returns:
            Updated SystemConfig record

        Raises:
            ValueError: If key is unknown or value fails validation
        """
        config = await self._get_config(key)
        if not config:
            raise ValueError(f"Unknown configuration key: {key}")

        # Validate value against constraints
        self._validate_value(config, value)

        # Update
        config.value = str(value)
        await self.db.commit()
        await self.db.refresh(config)

        logger.info(f"Updated config {key}={value}")
        return config

    async def get_all_by_category(self, category: str) -> list[SystemConfig]:
        """Get all configurations for a category.

        Args:
            category: Category name to filter by

        Returns:
            List of SystemConfig records in the category
        """
        query = select(SystemConfig).where(SystemConfig.category == category)
        result = await self.db.execute(query)
        return list(result.scalars().all())

    def _validate_value(self, config: SystemConfig, value: Any) -> None:
        """Validate value against config constraints.

        Args:
            config: SystemConfig record with constraints
            value: Value to validate

        Raises:
            ValueError: If value fails validation
        """
        # Type check
        if config.data_type == ConfigDataType.FLOAT.value:
            if not isinstance(value, (int, float)):
                raise ValueError(f"Value must be a number for {config.key}")
            value = float(value)
        elif config.data_type == ConfigDataType.INT.value:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"Value must be an integer for {config.key}")
            value = int(value)
        elif config.data_type == ConfigDataType.BOOLEAN.value:
            if not isinstance(value, bool):
                raise ValueError(f"Value must be a boolean for {config.key}")

        # Range check for numeric types
        if config.min_value is not None and config.data_type in ("float", "int"):
            min_val = float(config.min_value)
            if float(value) < min_val:
                raise ValueError(f"{config.key} must be >= {min_val}")

        if config.max_value is not None and config.data_type in ("float", "int"):
            max_val = float(config.max_value)
            if float(value) > max_val:
                raise ValueError(f"{config.key} must be <= {max_val}")

        # Allowed values check
        if config.allowed_values:
            if str(value) not in config.allowed_values:
                raise ValueError(
                    f"{config.key} must be one of: {config.allowed_values}"
                )


class SyncConfigService:
    """Synchronous configuration service for background workers (RQ jobs).

    This class provides the same interface as ConfigService but uses
    synchronous database operations for compatibility with RQ workers.
    """

    DEFAULTS = ConfigService.DEFAULTS

    def __init__(self, db_session: SyncSession):
        """Initialize sync config service with database session.

        Args:
            db_session: Synchronous SQLAlchemy session for database operations
        """
        self.db = db_session

    def get_float(self, key: str) -> float:
        """Get a float configuration value synchronously.

        Args:
            key: Configuration key name

        Returns:
            Float value from database or default
        """
        config = self.db.execute(
            select(SystemConfig).where(SystemConfig.key == key)
        ).scalar_one_or_none()

        if config:
            return float(config.value)
        default = self.DEFAULTS.get(key)
        if default is not None:
            return float(default)
        raise ValueError(f"Unknown configuration key: {key}")

    def get_int(self, key: str) -> int:
        """Get an integer configuration value synchronously.

        Args:
            key: Configuration key name

        Returns:
            Integer value from database or default
        """
        config = self.db.execute(
            select(SystemConfig).where(SystemConfig.key == key)
        ).scalar_one_or_none()

        if config:
            return int(config.value)
        default = self.DEFAULTS.get(key)
        if default is not None:
            return int(default)
        raise ValueError(f"Unknown configuration key: {key}")

    def get_string(self, key: str) -> str:
        """Get a string configuration value synchronously.

        Args:
            key: Configuration key name

        Returns:
            String value from database or default
        """
        config = self.db.execute(
            select(SystemConfig).where(SystemConfig.key == key)
        ).scalar_one_or_none()

        if config:
            return str(config.value)
        default = self.DEFAULTS.get(key)
        if default is not None:
            return str(default)
        raise ValueError(f"Unknown configuration key: {key}")

    def get_bool(self, key: str) -> bool:
        """Get a boolean configuration value synchronously.

        Args:
            key: Configuration key name

        Returns:
            Boolean value from database or default
        """
        value = self.get_string(key)
        return value.lower() in ("true", "1", "yes")
