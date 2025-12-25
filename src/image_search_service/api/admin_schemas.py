"""Admin operation schemas for destructive operations."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DeleteAllDataRequest(BaseModel):
    """Request schema for deleting all application data.

    DANGER: This is a highly destructive operation that deletes ALL data
    from both Qdrant and PostgreSQL (except Alembic migration tracking).

    Requires double confirmation for safety.
    """

    model_config = ConfigDict(populate_by_name=True)

    confirm: bool = Field(
        description="First confirmation flag - must be true"
    )
    confirmation_text: str = Field(
        alias="confirmationText",
        serialization_alias="confirmationText",
        description="Second confirmation - must exactly match 'DELETE ALL DATA'",
    )
    reason: str | None = Field(
        None,
        description="Optional reason for deletion (for audit trail)",
    )


class DeleteAllDataResponse(BaseModel):
    """Response schema for delete all data operation."""

    model_config = ConfigDict(populate_by_name=True)

    qdrant_collections_deleted: dict[str, int] = Field(
        serialization_alias="qdrantCollectionsDeleted",
        description="Map of collection name to number of vectors deleted",
    )
    postgres_tables_truncated: dict[str, int] = Field(
        serialization_alias="postgresTablesTruncated",
        description="Map of table name to number of rows deleted",
    )
    alembic_version_preserved: str = Field(
        serialization_alias="alembicVersionPreserved",
        description="Alembic migration version that was preserved",
    )
    message: str = Field(
        description="Human-readable status message"
    )
    timestamp: datetime = Field(
        description="Timestamp when operation completed"
    )
