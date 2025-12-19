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
    disable_cors: bool = False  # Set DISABLE_CORS=true to disable CORS


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
