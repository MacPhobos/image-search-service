"""Application configuration management."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/image_search"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = Field(default="image_assets", alias="QDRANT_COLLECTION")
    qdrant_face_collection: str = Field(default="faces", alias="QDRANT_FACE_COLLECTION")

    @property
    def qdrant_host(self) -> str:
        """Extract host from Qdrant URL."""
        # Parse host from URL (e.g., http://localhost:6333 -> localhost)
        url = self.qdrant_url.replace("http://", "").replace("https://", "")
        return url.split(":")[0]

    @property
    def qdrant_port(self) -> int:
        """Extract port from Qdrant URL."""
        # Parse port from URL (e.g., http://localhost:6333 -> 6333)
        url = self.qdrant_url.replace("http://", "").replace("https://", "")
        if ":" in url:
            return int(url.split(":")[1].split("/")[0])
        return 6333  # Default Qdrant port

    # CLIP model settings
    clip_model_name: str = Field(default="ViT-B-32", alias="CLIP_MODEL_NAME")
    clip_pretrained: str = Field(default="laion2b_s34b_b79k", alias="CLIP_PRETRAINED")
    embedding_dim: int = Field(default=512, alias="EMBEDDING_DIM")

    # Application
    log_level: str = "INFO"
    enable_cors: bool = True  # Set ENABLE_CORS=false to disable CORS

    # Training system settings
    image_root_dir: str = Field(default="", alias="IMAGE_ROOT_DIR")
    thumbnail_dir: str = Field(default="/tmp/thumbnails", alias="THUMBNAIL_DIR")
    thumbnail_size: int = Field(default=256, alias="THUMBNAIL_SIZE")
    watch_enabled: bool = Field(default=False, alias="WATCH_ENABLED")
    watch_interval_seconds: int = Field(default=60, alias="WATCH_INTERVAL_SECONDS")
    watch_debounce_seconds: float = Field(default=1.0, alias="WATCH_DEBOUNCE_SECONDS")
    watch_auto_train: bool = Field(default=False, alias="WATCH_AUTO_TRAIN")
    training_batch_size: int = Field(default=32, alias="TRAINING_BATCH_SIZE")

    # Face recognition model settings
    face_model_name: str = Field(default="buffalo_l", alias="FACE_MODEL_NAME")
    face_model_checkpoint: str = Field(default="", alias="FACE_MODEL_CHECKPOINT")
    face_training_enabled: bool = Field(default=False, alias="FACE_TRAINING_ENABLED")

    # Training hyperparameters
    face_triplet_margin: float = Field(default=0.2, alias="FACE_TRIPLET_MARGIN")
    face_training_epochs: int = Field(default=20, alias="FACE_TRAINING_EPOCHS")
    face_batch_size: int = Field(default=32, alias="FACE_BATCH_SIZE")
    face_learning_rate: float = Field(default=0.0001, alias="FACE_LEARNING_RATE")

    # Supervised clustering (known people)
    face_person_match_threshold: float = Field(default=0.7, alias="FACE_PERSON_MATCH_THRESHOLD")

    # Unsupervised clustering (unknown faces)
    face_unknown_clustering_method: str = Field(
        default="hdbscan", alias="FACE_UNKNOWN_CLUSTERING_METHOD"
    )
    face_unknown_min_cluster_size: int = Field(
        default=3, alias="FACE_UNKNOWN_MIN_CLUSTER_SIZE"
    )
    face_unknown_eps: float = Field(default=0.5, alias="FACE_UNKNOWN_EPS")

    # Face suggestion settings
    face_suggestion_min_confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        alias="FACE_SUGGESTION_MIN_CONFIDENCE",
        description="Minimum confidence threshold for face suggestions (0.0-1.0)",
    )
    face_suggestion_max_results: int = Field(
        default=5,
        ge=1,
        le=10,
        alias="FACE_SUGGESTION_MAX_RESULTS",
        description="Maximum number of face suggestions to return",
    )

    # Prototype creation settings
    face_prototype_min_quality: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        alias="FACE_PROTOTYPE_MIN_QUALITY",
        description="Minimum quality score for prototype creation (0.0-1.0)",
    )
    face_prototype_max_exemplars: int = Field(
        default=5,
        ge=1,
        le=20,
        alias="FACE_PROTOTYPE_MAX_EXEMPLARS",
        description="Maximum exemplar prototypes per person",
    )

    # Temporal prototype settings
    face_prototype_max_total: int = Field(
        default=12,
        alias="FACE_PROTOTYPE_MAX_TOTAL",
        description="Maximum total prototypes per person (including all types)",
    )
    face_prototype_temporal_slots: int = Field(
        default=6,
        alias="FACE_PROTOTYPE_TEMPORAL_SLOTS",
        description="Number of slots reserved for temporal prototypes",
    )
    face_prototype_primary_slots: int = Field(
        default=3,
        alias="FACE_PROTOTYPE_PRIMARY_SLOTS",
        description="Number of slots reserved for primary (pinned) prototypes",
    )
    face_prototype_temporal_mode: bool = Field(
        default=True,
        alias="FACE_PROTOTYPE_TEMPORAL_MODE",
        description="Enable temporal prototype mode (age-era based selection)",
    )

    # Age-era bucket ranges
    age_era_infant_max: int = Field(
        default=3,
        alias="AGE_ERA_INFANT_MAX",
        description="Maximum age for infant era (0-3 years)",
    )
    age_era_child_max: int = Field(
        default=12,
        alias="AGE_ERA_CHILD_MAX",
        description="Maximum age for child era (4-12 years)",
    )
    age_era_teen_max: int = Field(
        default=19,
        alias="AGE_ERA_TEEN_MAX",
        description="Maximum age for teen era (13-19 years)",
    )
    age_era_young_adult_max: int = Field(
        default=35,
        alias="AGE_ERA_YOUNG_ADULT_MAX",
        description="Maximum age for young adult era (20-35 years)",
    )
    age_era_adult_max: int = Field(
        default=55,
        alias="AGE_ERA_ADULT_MAX",
        description="Maximum age for adult era (36-55 years)",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
