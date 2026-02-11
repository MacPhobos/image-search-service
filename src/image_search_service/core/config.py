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
    qdrant_centroid_collection: str = Field(
        default="person_centroids", alias="QDRANT_CENTROID_COLLECTION"
    )
    qdrant_strict_startup: bool = Field(
        default=True,
        alias="QDRANT_STRICT_STARTUP",
        description=(
            "Exit on missing Qdrant collections at startup (default: true). "
            "Set to false for development flexibility (service starts with warnings)."
        ),
    )

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

    # CLIP model settings (kept for backward compatibility)
    clip_model_name: str = Field(default="ViT-B-32", alias="CLIP_MODEL_NAME")
    clip_pretrained: str = Field(default="laion2b_s34b_b79k", alias="CLIP_PRETRAINED")
    embedding_dim: int = Field(default=512, alias="EMBEDDING_DIM")

    # SigLIP model settings
    siglip_model_name: str = Field(default="ViT-B-16-SigLIP", alias="SIGLIP_MODEL_NAME")
    siglip_pretrained: str = Field(default="webli", alias="SIGLIP_PRETRAINED")
    siglip_embedding_dim: int = Field(default=768, alias="SIGLIP_EMBEDDING_DIM")
    siglip_collection: str = Field(default="image_assets_siglip", alias="SIGLIP_COLLECTION")

    # Feature flags for gradual rollout
    use_siglip: bool = Field(default=False, alias="USE_SIGLIP")
    siglip_rollout_percentage: int = Field(
        default=0, ge=0, le=100, alias="SIGLIP_ROLLOUT_PERCENTAGE"
    )

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

    # GPU memory management (for MPS/CUDA)
    # Size of batches for GPU inference (default 16 for CUDA, consider 8 for MPS)
    gpu_batch_size: int = Field(default=16, alias="GPU_BATCH_SIZE")
    # Enable explicit GPU memory cleanup (delete tensors, call gc.collect())
    gpu_memory_cleanup_enabled: bool = Field(default=True, alias="GPU_MEMORY_CLEANUP_ENABLED")
    # Interval for periodic garbage collection (every N images processed)
    gpu_memory_cleanup_interval: int = Field(default=50, alias="GPU_MEMORY_CLEANUP_INTERVAL")

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

    # Unknown face clustering display settings
    unknown_face_cluster_min_confidence: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        alias="UNKNOWN_FACE_CLUSTER_MIN_CONFIDENCE",
        description=(
            "Minimum intra-cluster confidence threshold for displaying unknown "
            "faces (0.0-1.0)"
        ),
    )
    unknown_face_cluster_min_size: int = Field(
        default=2,
        ge=1,
        le=100,
        alias="UNKNOWN_FACE_CLUSTER_MIN_SIZE",
        description="Minimum number of faces required per cluster for displaying unknown faces",
    )

    # Unknown person discovery settings
    unknown_person_min_display_count: int = Field(
        default=5,
        ge=2,
        le=50,
        alias="UNKNOWN_PERSON_MIN_DISPLAY_COUNT",
        description="Minimum faces per candidate group to display (Admin UI configurable)",
    )
    unknown_person_default_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        alias="UNKNOWN_PERSON_DEFAULT_THRESHOLD",
        description="Default confidence threshold for unknown person groups",
    )
    unknown_person_max_faces: int = Field(
        default=50000,
        ge=100,
        le=100000,
        alias="UNKNOWN_PERSON_MAX_FACES",
        description="Maximum unassigned faces to process during discovery",
    )
    unknown_person_chunk_size: int = Field(
        default=10000,
        ge=1000,
        le=20000,
        alias="UNKNOWN_PERSON_CHUNK_SIZE",
        description="Chunk size for batched clustering at 50K+ scale",
    )

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
    face_suggestions_auto_rescan_on_recompute: bool = Field(
        default=False,
        alias="FACE_SUGGESTIONS_AUTO_RESCAN_ON_RECOMPUTE",
        description="Automatically regenerate suggestions when prototypes are recomputed",
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

    # Person centroid settings
    centroid_model_version: str = Field(
        default="arcface_r100_glint360k_v1",
        alias="CENTROID_MODEL_VERSION",
        description="Embedding model version for centroids",
    )
    centroid_algorithm_version: int = Field(
        default=2,
        alias="CENTROID_ALGORITHM_VERSION",
        description="Centroid computation algorithm version",
    )
    centroid_min_faces: int = Field(
        default=2,
        ge=1,
        alias="CENTROID_MIN_FACES",
        description="Minimum number of faces required to compute centroid",
    )
    centroid_clustering_min_faces: int = Field(
        default=200,
        ge=50,
        alias="CENTROID_CLUSTERING_MIN_FACES",
        description="Minimum faces for cluster-based centroid computation",
    )
    centroid_trim_threshold_small: float = Field(
        default=0.05,
        ge=0.0,
        le=0.2,
        alias="CENTROID_TRIM_THRESHOLD_SMALL",
        description="Outlier trim threshold for 50-300 faces (5% default)",
    )
    centroid_trim_threshold_large: float = Field(
        default=0.10,
        ge=0.0,
        le=0.3,
        alias="CENTROID_TRIM_THRESHOLD_LARGE",
        description="Outlier trim threshold for 300+ faces (10% default)",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
