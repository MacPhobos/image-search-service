"""Database models."""

import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB as PG_JSONB
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from sqlalchemy.types import Enum as SQLEnum


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


# Use JSONB for PostgreSQL, JSON for SQLite compatibility in tests
JSONB = JSON().with_variant(PG_JSONB(astext_type=Text()), "postgresql")


class TrainingStatus(str, Enum):
    """Training status enum for image assets."""

    PENDING = "pending"
    QUEUED = "queued"
    TRAINING = "training"
    TRAINED = "trained"
    FAILED = "failed"


class SessionStatus(str, Enum):
    """Status enum for training sessions."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class JobStatus(str, Enum):
    """Status enum for training jobs."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SubdirectoryStatus(str, Enum):
    """Status enum for training subdirectories."""

    PENDING = "pending"
    TRAINING = "training"
    TRAINED = "trained"
    FAILED = "failed"


class PersonStatus(str, Enum):
    """Status enum for person entities."""

    ACTIVE = "active"
    MERGED = "merged"
    HIDDEN = "hidden"


class PrototypeRole(str, Enum):
    """Role enum for person prototypes."""

    CENTROID = "centroid"  # Computed average/centroid
    EXEMPLAR = "exemplar"  # High-quality representative face


class Category(Base):
    """Category model for organizing training sessions."""

    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    color: Mapped[str | None] = mapped_column(String(7), nullable=True)  # Hex color like #3B82F6
    is_default: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    training_sessions: Mapped[list["TrainingSession"]] = relationship(
        "TrainingSession", back_populates="category"
    )

    __table_args__ = (
        Index("idx_categories_name", "name"),
        Index("idx_categories_is_default", "is_default"),
    )

    def __repr__(self) -> str:
        return f"<Category(id={self.id}, name={self.name}, is_default={self.is_default})>"


class ImageAsset(Base):
    """Image asset model storing metadata and file paths."""

    __tablename__ = "image_assets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    path: Mapped[str] = mapped_column(String(500), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    indexed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # New columns for production training system
    thumbnail_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    file_size: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    mime_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    file_modified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    training_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=TrainingStatus.PENDING.value
    )

    # Relationships
    training_jobs: Mapped[list["TrainingJob"]] = relationship(
        "TrainingJob", back_populates="asset", cascade="all, delete-orphan"
    )
    training_evidence: Mapped[list["TrainingEvidence"]] = relationship(
        "TrainingEvidence", back_populates="asset", cascade="all, delete-orphan"
    )
    face_instances: Mapped[list["FaceInstance"]] = relationship(
        "FaceInstance", back_populates="asset", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_image_assets_training_status", "training_status"),
        Index("idx_image_assets_file_modified_at", "file_modified_at"),
    )

    def __repr__(self) -> str:
        return f"<ImageAsset(id={self.id}, path={self.path})>"


class TrainingSession(Base):
    """Training session model for managing batch training operations."""

    __tablename__ = "training_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=SessionStatus.PENDING.value
    )
    root_path: Mapped[str] = mapped_column(String(500), nullable=False)
    category_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("categories.id", ondelete="SET NULL"), nullable=True
    )
    config: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)

    # Progress tracking
    total_images: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    processed_images: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_images: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    paused_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    reset_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    reset_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    category: Mapped["Category | None"] = relationship(
        "Category", back_populates="training_sessions"
    )
    subdirectories: Mapped[list["TrainingSubdirectory"]] = relationship(
        "TrainingSubdirectory", back_populates="session", cascade="all, delete-orphan"
    )
    jobs: Mapped[list["TrainingJob"]] = relationship(
        "TrainingJob", back_populates="session", cascade="all, delete-orphan"
    )
    evidence: Mapped[list["TrainingEvidence"]] = relationship(
        "TrainingEvidence", back_populates="session", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_training_sessions_status", "status"),
        Index("idx_training_sessions_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<TrainingSession(id={self.id}, name={self.name}, status={self.status})>"


class TrainingSubdirectory(Base):
    """Subdirectory selection for training sessions."""

    __tablename__ = "training_subdirectories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("training_sessions.id", ondelete="CASCADE"), nullable=False
    )
    path: Mapped[str] = mapped_column(String(500), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    selected: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    image_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    trained_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=SubdirectoryStatus.PENDING.value
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    session: Mapped["TrainingSession"] = relationship(
        "TrainingSession", back_populates="subdirectories"
    )

    __table_args__ = (
        Index("idx_training_subdirectories_session", "session_id"),
        Index("idx_training_subdirectories_status", "status"),
    )

    def __repr__(self) -> str:
        return f"<TrainingSubdirectory(id={self.id}, path={self.path}, selected={self.selected})>"


class TrainingJob(Base):
    """Individual background job tracking for training operations."""

    __tablename__ = "training_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("training_sessions.id", ondelete="CASCADE"), nullable=False
    )
    asset_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("image_assets.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=JobStatus.PENDING.value
    )
    rq_job_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    processing_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    session: Mapped["TrainingSession"] = relationship(
        "TrainingSession", back_populates="jobs"
    )
    asset: Mapped["ImageAsset"] = relationship("ImageAsset", back_populates="training_jobs")

    __table_args__ = (
        Index("idx_training_jobs_session", "session_id"),
        Index("idx_training_jobs_asset", "asset_id"),
        Index("idx_training_jobs_status", "status"),
        Index("idx_training_jobs_rq_job_id", "rq_job_id"),
    )

    def __repr__(self) -> str:
        return f"<TrainingJob(id={self.id}, session_id={self.session_id}, status={self.status})>"


class TrainingEvidence(Base):
    """Training evidence and metadata for inspection and debugging."""

    __tablename__ = "training_evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("image_assets.id", ondelete="CASCADE"), nullable=False
    )
    session_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("training_sessions.id", ondelete="CASCADE"), nullable=False
    )
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    embedding_checksum: Mapped[str | None] = mapped_column(String(64), nullable=True)
    device: Mapped[str] = mapped_column(String(20), nullable=False)
    processing_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict[str, object] | None] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    asset: Mapped["ImageAsset"] = relationship(
        "ImageAsset", back_populates="training_evidence"
    )
    session: Mapped["TrainingSession"] = relationship(
        "TrainingSession", back_populates="evidence"
    )

    __table_args__ = (
        Index("idx_training_evidence_asset", "asset_id"),
        Index("idx_training_evidence_session", "session_id"),
        Index("idx_training_evidence_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<TrainingEvidence(id={self.id}, "
            f"asset_id={self.asset_id}, model={self.model_name})>"
        )


class VectorDeletionLog(Base):
    """Audit log for vector deletion operations."""

    __tablename__ = "vector_deletion_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    deletion_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # DIRECTORY, ASSET, SESSION, CATEGORY, ORPHAN, FULL_RESET
    deletion_target: Mapped[str] = mapped_column(
        Text, nullable=False
    )  # Path prefix, asset_id, session_id, etc.
    vector_count: Mapped[int] = mapped_column(
        Integer, nullable=False
    )  # Number of vectors deleted
    deletion_reason: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # Optional user-provided reason
    metadata_json: Mapped[dict[str, object] | None] = mapped_column(
        "metadata", JSON, nullable=True
    )  # Additional context (column name "metadata", attribute name "metadata_json")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("idx_vector_deletion_logs_deletion_type", "deletion_type"),
        Index("idx_vector_deletion_logs_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<VectorDeletionLog(id={self.id}, "
            f"type={self.deletion_type}, count={self.vector_count})>"
        )


class Person(Base):
    """Person entity for face recognition and labeling."""

    __tablename__ = "persons"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[PersonStatus] = mapped_column(
        SQLEnum(
            PersonStatus,
            name="person_status",
            create_type=False,  # We create it in migration
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
        default=PersonStatus.ACTIVE,
    )
    merged_into_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("persons.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    face_instances: Mapped[list["FaceInstance"]] = relationship(
        "FaceInstance", back_populates="person"
    )
    prototypes: Mapped[list["PersonPrototype"]] = relationship(
        "PersonPrototype", back_populates="person"
    )

    __table_args__ = (
        Index("ix_persons_name_lower", func.lower(name), unique=True),
        Index("ix_persons_status", "status"),
    )

    def __repr__(self) -> str:
        return f"<Person(id={self.id}, name={self.name}, status={self.status})>"


class FaceInstance(Base):
    """Face instance detected in an image asset."""

    __tablename__ = "face_instances"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    asset_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("image_assets.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Bounding box (pixel coordinates)
    bbox_x: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_y: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_w: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_h: Mapped[int] = mapped_column(Integer, nullable=False)

    # Detection metadata
    landmarks: Mapped[dict[str, object] | None] = mapped_column(
        JSONB, nullable=True
    )  # 5-point facial landmarks
    detection_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Vector storage reference
    qdrant_point_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), unique=True, nullable=False, default=uuid.uuid4
    )

    # Clustering and person assignment
    cluster_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    person_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("persons.id", ondelete="SET NULL"),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    asset: Mapped["ImageAsset"] = relationship("ImageAsset", back_populates="face_instances")
    person: Mapped["Person | None"] = relationship("Person", back_populates="face_instances")

    __table_args__ = (
        # Idempotency: same face detected at same location won't duplicate
        UniqueConstraint(
            "asset_id",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            name="uq_face_instance_location",
        ),
        Index("ix_face_instances_asset_id", "asset_id"),
        Index("ix_face_instances_cluster_id", "cluster_id"),
        Index("ix_face_instances_person_id", "person_id"),
        Index("ix_face_instances_quality", "quality_score"),
    )

    def __repr__(self) -> str:
        return (
            f"<FaceInstance(id={self.id}, asset_id={self.asset_id}, "
            f"person_id={self.person_id})>"
        )


class PersonPrototype(Base):
    """Person prototype for face recognition (centroid or exemplar)."""

    __tablename__ = "person_prototypes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    person_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("persons.id", ondelete="CASCADE"),
        nullable=False,
    )
    face_instance_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("face_instances.id", ondelete="SET NULL"),
        nullable=True,
    )
    qdrant_point_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )
    role: Mapped[PrototypeRole] = mapped_column(
        SQLEnum(
            PrototypeRole,
            name="prototype_role",
            create_type=False,  # We create it in migration
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    person: Mapped["Person"] = relationship("Person", back_populates="prototypes")
    face_instance: Mapped["FaceInstance | None"] = relationship("FaceInstance")

    __table_args__ = (Index("ix_person_prototypes_person_id", "person_id"),)

    def __repr__(self) -> str:
        return (
            f"<PersonPrototype(id={self.id}, person_id={self.person_id}, "
            f"role={self.role})>"
        )


class FaceAssignmentEvent(Base):
    """Audit log for face assignment changes and person management operations."""

    __tablename__ = "face_assignment_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    actor: Mapped[str | None] = mapped_column(String(255), nullable=True)
    operation: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # REMOVE_FROM_PERSON, MOVE_TO_PERSON
    from_person_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("persons.id", ondelete="SET NULL"),
        nullable=True,
    )
    to_person_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("persons.id", ondelete="SET NULL"),
        nullable=True,
    )
    affected_photo_ids: Mapped[list[int] | None] = mapped_column(
        JSONB, nullable=True
    )  # Array of asset_ids
    affected_face_instance_ids: Mapped[list[str] | None] = mapped_column(
        JSONB, nullable=True
    )  # Array of UUIDs as strings
    face_count: Mapped[int] = mapped_column(Integer, nullable=False)
    photo_count: Mapped[int] = mapped_column(Integer, nullable=False)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_face_assignment_events_created_at", "created_at"),
        Index("ix_face_assignment_events_operation", "operation"),
        Index("ix_face_assignment_events_from_person_id", "from_person_id"),
        Index("ix_face_assignment_events_to_person_id", "to_person_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<FaceAssignmentEvent(id={self.id}, operation={self.operation}, "
            f"face_count={self.face_count})>"
        )


class FaceDetectionSessionStatus(str, Enum):
    """Status enum for face detection sessions."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class FaceSuggestionStatus(str, Enum):
    """Status for face suggestions."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EXPIRED = "expired"


class FaceDetectionSession(Base):
    """Face detection session for tracking batch face detection operations."""

    __tablename__ = "face_detection_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    training_session_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("training_sessions.id", ondelete="SET NULL"), nullable=True
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=FaceDetectionSessionStatus.PENDING.value
    )

    # Progress tracking
    total_images: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    processed_images: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_images: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    faces_detected: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    faces_assigned: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )  # Auto-assigned to known persons

    # Detection configuration
    min_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    min_face_size: Mapped[int] = mapped_column(Integer, nullable=False, default=20)
    batch_size: Mapped[int] = mapped_column(Integer, nullable=False, default=16)

    # Error tracking
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Background job reference
    job_id: Mapped[str | None] = mapped_column(
        String(100), nullable=True
    )  # RQ job ID

    __table_args__ = (
        Index("ix_face_detection_sessions_status", "status"),
        Index("ix_face_detection_sessions_training_session_id", "training_session_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<FaceDetectionSession(id={self.id}, status={self.status}, "
            f"total_images={self.total_images}, faces_detected={self.faces_detected})>"
        )


class FaceSuggestion(Base):
    """Suggested face-to-person assignment based on similarity."""

    __tablename__ = "face_suggestions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    face_instance_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("face_instances.id", ondelete="CASCADE"),
        nullable=False,
    )
    suggested_person_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("persons.id", ondelete="CASCADE"),
        nullable=False,
    )
    confidence: Mapped[float] = mapped_column(
        Float, nullable=False
    )  # Cosine similarity score
    source_face_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("face_instances.id", ondelete="CASCADE"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(20), default=FaceSuggestionStatus.PENDING.value, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Indexes for efficient querying
    __table_args__ = (
        Index("ix_face_suggestions_face_instance_id", "face_instance_id"),
        Index("ix_face_suggestions_suggested_person_id", "suggested_person_id"),
        Index("ix_face_suggestions_status", "status"),
    )

    def __repr__(self) -> str:
        return (
            f"<FaceSuggestion(id={self.id}, face_instance_id={self.face_instance_id}, "
            f"suggested_person_id={self.suggested_person_id}, confidence={self.confidence})>"
        )


class ConfigDataType(str, Enum):
    """Data types for configuration values."""

    FLOAT = "float"
    INT = "int"
    STRING = "string"
    BOOLEAN = "boolean"


class SystemConfig(Base):
    """System configuration table for database-backed settings.

    This table provides a flexible key-value store for application configuration
    with built-in type safety and validation constraints. It supports multiple
    data types and can enforce min/max ranges or allowed value lists.
    """

    __tablename__ = "system_configs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    value: Mapped[str] = mapped_column(String(500), nullable=False)
    data_type: Mapped[str] = mapped_column(String(20), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Validation constraints (stored as strings, parsed based on data_type)
    min_value: Mapped[str | None] = mapped_column(String(50), nullable=True)
    max_value: Mapped[str | None] = mapped_column(String(50), nullable=True)
    allowed_values: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)

    # Metadata
    category: Mapped[str] = mapped_column(
        String(50), nullable=False, default="general"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_system_configs_key", "key"),
        Index("ix_system_configs_category", "category"),
    )

    def __repr__(self) -> str:
        return f"<SystemConfig(key={self.key}, value={self.value}, category={self.category})>"
