"""Database models."""

from datetime import datetime
from enum import Enum

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


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

    # Relationships
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
