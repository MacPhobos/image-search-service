"""Admin operation schemas for destructive operations."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


def to_camel(string: str) -> str:
    """Convert snake_case to camelCase."""
    words = string.split("_")
    return words[0] + "".join(word.capitalize() for word in words[1:])


class CamelCaseModel(BaseModel):
    """Base model with camelCase aliases for JSON serialization."""

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
    )


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


# ============ Person Metadata Export/Import Schemas ============


class BoundingBoxExport(CamelCaseModel):
    """Bounding box coordinates for face export/import."""

    x: int
    y: int
    width: int
    height: int


class FaceMappingExport(CamelCaseModel):
    """Face mapping data for export/import."""

    image_path: str
    bounding_box: BoundingBoxExport
    detection_confidence: float
    quality_score: float | None = None


class PersonExport(CamelCaseModel):
    """Person data for export."""

    name: str
    status: str
    face_mappings: list[FaceMappingExport]


class ExportMetadata(CamelCaseModel):
    """Metadata about the export."""

    total_persons: int
    total_face_mappings: int
    export_format: str = Field(default="seed")


class PersonMetadataExport(CamelCaseModel):
    """Complete person metadata export structure."""

    version: str = Field(default="1.0")
    exported_at: datetime
    metadata: ExportMetadata
    persons: list[PersonExport]


class ExportOptions(CamelCaseModel):
    """Options for exporting person metadata."""

    verify_paths: bool = Field(
        default=False,
        description="If true, only export faces where image file exists on filesystem",
    )


class ImportOptions(CamelCaseModel):
    """Options for importing person metadata."""

    dry_run: bool = Field(default=False)
    tolerance_pixels: int = Field(default=10, ge=1, le=50)
    skip_missing_images: bool = Field(default=True)
    auto_ingest_images: bool = Field(
        default=True,
        description="Automatically ingest images that exist on filesystem but not in database",
    )


class ImportRequest(CamelCaseModel):
    """Request to import person metadata."""

    data: PersonMetadataExport
    options: ImportOptions = Field(default_factory=ImportOptions)


class FaceMappingResult(CamelCaseModel):
    """Result of attempting to match a single face mapping."""

    image_path: str
    status: str  # "matched", "not_found", "image_missing", "detection_failed"
    matched_face_id: str | None = None
    error: str | None = None


class PersonImportResult(CamelCaseModel):
    """Result of importing a single person."""

    name: str
    status: str  # "created", "existing", "error"
    person_id: str | None = None
    faces_matched: int = Field(default=0)
    faces_not_found: int = Field(default=0)
    images_missing: int = Field(default=0)
    details: list[FaceMappingResult] = Field(default_factory=list)


class ImportResponse(CamelCaseModel):
    """Response from person metadata import operation."""

    success: bool
    dry_run: bool
    persons_created: int = Field(default=0)
    persons_existing: int = Field(default=0)
    total_faces_matched: int = Field(default=0)
    total_faces_not_found: int = Field(default=0)
    total_images_missing: int = Field(default=0)
    person_results: list[PersonImportResult] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    timestamp: datetime
