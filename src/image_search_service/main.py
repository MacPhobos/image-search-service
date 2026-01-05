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
