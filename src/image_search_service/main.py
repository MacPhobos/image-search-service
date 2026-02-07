"""FastAPI application factory with lazy initialization."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from image_search_service.api.routes import api_v1_router, router
from image_search_service.core.config import get_settings
from image_search_service.core.logging import configure_logging, get_logger
from image_search_service.db.session import close_db
from image_search_service.vector.qdrant import close_qdrant

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup and shutdown.

    Note: Database and external clients are lazily initialized on first use,
    so we only need to handle cleanup here.

    Model Preloading
    ================
    The embedding model is preloaded during startup in the main process.
    This ensures it's cached in memory before RQ workers fork, avoiding
    Metal compiler service initialization issues in subprocesses on macOS.
    Workers inherit the already-loaded model object, allowing safe inference
    without re-initialization.
    """
    from image_search_service.services.embedding import preload_embedding_model
    from image_search_service.services.watcher_manager import WatcherManager

    logger.info("Application starting up")

    # Preload embedding model in main process before workers fork
    # This avoids Metal compiler service issues on macOS in subprocesses
    try:
        preload_embedding_model()
    except Exception as e:
        logger.warning(f"Failed to preload embedding model: {e}. Will load on first use.")

    # Validate Qdrant collections exist at startup
    logger.info("Validating Qdrant collections...")
    try:
        from image_search_service.vector.qdrant import validate_qdrant_collections

        settings = get_settings()
        missing_collections = validate_qdrant_collections()

        if missing_collections:
            error_message = (
                f"\n{'='*80}\n"
                f"FATAL: Missing {len(missing_collections)} required Qdrant collection(s):\n"
                f"{chr(10).join('  - ' + name for name in missing_collections)}\n"
                f"\n"
                f"These collections are required for the service to function.\n"
                f"\n"
                f"To fix, run:\n"
                f"  make bootstrap-qdrant\n"
                f"\n"
                f"Or manually create collections:\n"
                f"  cd image-search-service\n"
                f"  uv run python -m image_search_service.scripts.bootstrap_qdrant init\n"
                f"\n"
                f"To disable strict validation (NOT recommended for production):\n"
                f"  export QDRANT_STRICT_STARTUP=false\n"
                f"{'='*80}\n"
            )

            if settings.qdrant_strict_startup:
                logger.critical(error_message)
                logger.critical("Exiting due to missing collections (QDRANT_STRICT_STARTUP=true)")
                import sys

                sys.exit(1)
            else:
                logger.warning(error_message)
                logger.warning(
                    "Continuing startup despite missing collections (QDRANT_STRICT_STARTUP=false). "
                    "Endpoints using these collections will fail at runtime."
                )
        else:
            logger.info("✅ All required Qdrant collections validated")

    except Exception as e:
        settings = get_settings()
        error_message = (
            f"\n{'='*80}\n"
            f"FATAL: Failed to validate Qdrant collections: {e}\n"
            f"\n"
            f"Possible causes:\n"
            f"  - Qdrant service is not running\n"
            f"  - Qdrant URL is incorrect (current: {settings.qdrant_url})\n"
            f"  - Network connectivity issues\n"
            f"\n"
            f"To fix:\n"
            f"  1. Ensure Qdrant is running: docker-compose up -d\n"
            f"  2. Check Qdrant URL: curl {settings.qdrant_url}/health\n"
            f"  3. Run: make bootstrap-qdrant\n"
            f"{'='*80}\n"
        )

        if settings.qdrant_strict_startup:
            logger.critical(error_message)
            import sys

            sys.exit(1)
        else:
            logger.warning(error_message)

    # Start file watcher if enabled
    watcher = WatcherManager.get_instance()
    watcher.start()

    yield

    logger.info("Application shutting down")

    # Stop file watcher
    watcher.stop()

    await close_db()
    close_qdrant()


def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    settings = get_settings()

    app = FastAPI(
        title="Image Search Service",
        description="Vector similarity search for images",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Add CORS middleware (enabled by default, disable with ENABLE_CORS=false)
    if not settings.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.info("CORS middleware enabled")
    else:
        logger.info("CORS middleware disabled via DISABLE_CORS=true")

    # Register routes
    app.include_router(router)
    app.include_router(api_v1_router)

    return app


app = create_app()
