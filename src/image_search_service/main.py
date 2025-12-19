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
    """
    logger.info("Application starting up")
    yield
    logger.info("Application shutting down")
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

    # Add CORS middleware (enabled by default, disable with DISABLE_CORS=true)
    if not settings.disable_cors:
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
