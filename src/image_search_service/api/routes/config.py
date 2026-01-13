"""Configuration API routes.

Provides endpoints for reading and updating system configuration values,
with a specialized endpoint for face matching settings.
"""

import logging
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.ext.asyncio import AsyncSession


def to_camel(string: str) -> str:
    """Convert snake_case to camelCase."""
    words = string.split("_")
    return words[0] + "".join(word.capitalize() for word in words[1:])

from image_search_service.db.session import get_db
from image_search_service.services.config_service import ConfigService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/config", tags=["Configuration"])


# Response Models
class ConfigItemResponse(BaseModel):
    """Single configuration item response."""

    key: str
    value: str
    data_type: str
    description: str | None
    min_value: str | None
    max_value: str | None
    allowed_values: list[str] | None
    category: str


class ConfigListResponse(BaseModel):
    """List of configuration items."""

    items: list[ConfigItemResponse]
    category: str


class FaceMatchingConfigResponse(BaseModel):
    """Face matching specific configuration."""

    auto_assign_threshold: float
    suggestion_threshold: float
    max_suggestions: int
    suggestion_expiry_days: int
    prototype_min_quality: float
    prototype_max_exemplars: int
    # Post-training suggestions
    post_training_suggestions_mode: str
    post_training_suggestions_top_n_count: int


class FaceSuggestionSettingsResponse(BaseModel):
    """Face suggestion pagination settings."""

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
    )

    groups_per_page: int = Field(..., ge=1, le=50, description="Number of person groups per page")
    items_per_group: int = Field(..., ge=1, le=50, description="Number of suggestions per group")


# Request Models
class ConfigUpdateRequest(BaseModel):
    """Request to update a configuration value."""

    value: Any = Field(..., description="New value for the configuration")


class ConfigUpdateResponse(BaseModel):
    """Response after updating configuration."""

    key: str
    value: str
    updated: bool


class FaceMatchingConfigUpdateRequest(BaseModel):
    """Request to update face matching configuration."""

    auto_assign_threshold: float = Field(..., ge=0.5, le=1.0)
    suggestion_threshold: float = Field(..., ge=0.3, le=0.95)
    max_suggestions: int = Field(default=50, ge=1, le=200)
    suggestion_expiry_days: int = Field(default=30, ge=1, le=365)
    prototype_min_quality: float = Field(default=0.5, ge=0.0, le=1.0)
    prototype_max_exemplars: int = Field(default=5, ge=1, le=20)
    # Post-training suggestions (optional)
    post_training_suggestions_mode: Literal["all", "top_n"] | None = None
    post_training_suggestions_top_n_count: int | None = Field(None, ge=1, le=100)

    @model_validator(mode="after")
    def validate_thresholds(self) -> "FaceMatchingConfigUpdateRequest":
        """Ensure suggestion_threshold < auto_assign_threshold."""
        if self.suggestion_threshold >= self.auto_assign_threshold:
            raise ValueError(
                "suggestion_threshold must be less than auto_assign_threshold"
            )
        return self


class FaceSuggestionSettingsUpdateRequest(BaseModel):
    """Request to update face suggestion pagination settings."""

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
    )

    groups_per_page: int = Field(..., ge=1, le=50)
    items_per_group: int = Field(..., ge=1, le=50)


class UnknownFaceClusteringConfigResponse(BaseModel):
    """Unknown face clustering display configuration."""

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
    )

    min_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Minimum intra-cluster confidence threshold",
    )
    min_cluster_size: int = Field(
        ..., ge=1, le=100, description="Minimum number of faces required per cluster"
    )


class UnknownFaceClusteringConfigUpdateRequest(BaseModel):
    """Request to update unknown face clustering configuration."""

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
    )

    min_confidence: float = Field(..., ge=0.0, le=1.0)
    min_cluster_size: int = Field(..., ge=1, le=100)


# Endpoints
@router.get("/face-matching", response_model=FaceMatchingConfigResponse)
async def get_face_matching_config(
    db: AsyncSession = Depends(get_db),
) -> FaceMatchingConfigResponse:
    """Get face matching configuration settings.

    Returns the current configuration for automatic face-to-person matching,
    including confidence thresholds, suggestion limits, and prototype settings.
    """
    service = ConfigService(db)

    return FaceMatchingConfigResponse(
        auto_assign_threshold=await service.get_float("face_auto_assign_threshold"),
        suggestion_threshold=await service.get_float("face_suggestion_threshold"),
        max_suggestions=await service.get_int("face_suggestion_max_results"),
        suggestion_expiry_days=await service.get_int("face_suggestion_expiry_days"),
        prototype_min_quality=await service.get_float("face_prototype_min_quality"),
        prototype_max_exemplars=await service.get_int("face_prototype_max_exemplars"),
        # Post-training suggestions
        post_training_suggestions_mode=await service.get_string("post_training_suggestions_mode"),
        post_training_suggestions_top_n_count=await service.get_int("post_training_suggestions_top_n_count"),
    )


@router.put("/face-matching", response_model=FaceMatchingConfigResponse)
async def update_face_matching_config(
    request: FaceMatchingConfigUpdateRequest,
    db: AsyncSession = Depends(get_db),
) -> FaceMatchingConfigResponse:
    """Update face matching configuration settings.

    Updates all face matching configuration values atomically.
    Validates that suggestion_threshold < auto_assign_threshold.
    """
    service = ConfigService(db)

    try:
        await service.set_value(
            "face_auto_assign_threshold", request.auto_assign_threshold
        )
        await service.set_value(
            "face_suggestion_threshold", request.suggestion_threshold
        )
        await service.set_value("face_suggestion_max_results", request.max_suggestions)
        await service.set_value(
            "face_suggestion_expiry_days", request.suggestion_expiry_days
        )
        await service.set_value(
            "face_prototype_min_quality", request.prototype_min_quality
        )
        await service.set_value(
            "face_prototype_max_exemplars", request.prototype_max_exemplars
        )

        # Post-training suggestions (optional updates)
        if request.post_training_suggestions_mode is not None:
            await service.set_value(
                "post_training_suggestions_mode",
                request.post_training_suggestions_mode
            )

        if request.post_training_suggestions_top_n_count is not None:
            await service.set_value(
                "post_training_suggestions_top_n_count",
                str(request.post_training_suggestions_top_n_count)
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(
        f"Updated face matching config: auto={request.auto_assign_threshold}, "
        f"suggestion={request.suggestion_threshold}, "
        f"prototype_min_quality={request.prototype_min_quality}, "
        f"prototype_max_exemplars={request.prototype_max_exemplars}"
    )

    # Return updated config (fetch from DB to get all current values)
    return await get_face_matching_config(db)


@router.get("/face-suggestions", response_model=FaceSuggestionSettingsResponse)
async def get_face_suggestion_settings(
    db: AsyncSession = Depends(get_db),
) -> FaceSuggestionSettingsResponse:
    """Get face suggestion pagination settings.

    Returns the current configuration for group-based pagination
    of face suggestions.
    """
    service = ConfigService(db)

    return FaceSuggestionSettingsResponse(
        groups_per_page=await service.get_int("face_suggestion_groups_per_page"),
        items_per_group=await service.get_int("face_suggestion_items_per_group"),
    )


@router.put("/face-suggestions", response_model=FaceSuggestionSettingsResponse)
async def update_face_suggestion_settings(
    request: FaceSuggestionSettingsUpdateRequest,
    db: AsyncSession = Depends(get_db),
) -> FaceSuggestionSettingsResponse:
    """Update face suggestion pagination settings.

    Updates the group-based pagination configuration for face suggestions.
    """
    service = ConfigService(db)

    try:
        await service.set_value("face_suggestion_groups_per_page", request.groups_per_page)
        await service.set_value("face_suggestion_items_per_group", request.items_per_group)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(
        f"Updated face suggestion settings: groups_per_page={request.groups_per_page}, "
        f"items_per_group={request.items_per_group}"
    )

    return FaceSuggestionSettingsResponse(
        groups_per_page=request.groups_per_page,
        items_per_group=request.items_per_group,
    )


@router.get("/face-clustering-unknown", response_model=UnknownFaceClusteringConfigResponse)
async def get_unknown_clustering_config(
    db: AsyncSession = Depends(get_db),
) -> UnknownFaceClusteringConfigResponse:
    """Get configuration for unknown face clustering display.

    Returns the current filtering configuration for the Unknown Faces view,
    including confidence threshold and minimum cluster size settings.
    """
    from image_search_service.core.config import get_settings

    settings = get_settings()

    return UnknownFaceClusteringConfigResponse(
        min_confidence=settings.unknown_face_cluster_min_confidence,
        min_cluster_size=settings.unknown_face_cluster_min_size,
    )


@router.put("/face-clustering-unknown", response_model=UnknownFaceClusteringConfigResponse)
async def update_unknown_clustering_config(
    request: UnknownFaceClusteringConfigUpdateRequest,
    db: AsyncSession = Depends(get_db),
) -> UnknownFaceClusteringConfigResponse:
    """Update configuration for unknown face clustering display.

    Updates filtering thresholds for the Unknown Faces view. Note that these
    settings are currently stored in environment variables and will require
    service restart to take effect. Future versions will support runtime updates.

    Args:
        request: Updated configuration values
        db: Database session (for future database-backed config)

    Returns:
        Updated configuration values
    """
    # Note: Currently settings are loaded from environment variables
    # Future enhancement: Store in database for runtime updates
    logger.warning(
        "Unknown clustering config update requested but currently requires restart. "
        f"Requested: min_confidence={request.min_confidence}, "
        f"min_cluster_size={request.min_cluster_size}"
    )

    # For now, return the requested values as acknowledgment
    # TODO: Implement database-backed config storage for runtime updates
    return UnknownFaceClusteringConfigResponse(
        min_confidence=request.min_confidence,
        min_cluster_size=request.min_cluster_size,
    )


@router.get("/{category}", response_model=ConfigListResponse)
async def get_config_by_category(
    category: str,
    db: AsyncSession = Depends(get_db),
) -> ConfigListResponse:
    """Get all configuration items for a category.

    Returns all configuration items belonging to the specified category,
    including their current values and validation constraints.
    """
    service = ConfigService(db)
    configs = await service.get_all_by_category(category)

    return ConfigListResponse(
        items=[
            ConfigItemResponse(
                key=c.key,
                value=c.value,
                data_type=c.data_type,
                description=c.description,
                min_value=c.min_value,
                max_value=c.max_value,
                allowed_values=c.allowed_values,
                category=c.category,
            )
            for c in configs
        ],
        category=category,
    )


@router.put("/{key}", response_model=ConfigUpdateResponse)
async def update_config(
    key: str,
    request: ConfigUpdateRequest,
    db: AsyncSession = Depends(get_db),
) -> ConfigUpdateResponse:
    """Update a single configuration value.

    Updates the specified configuration key with the new value.
    Validates the value against the configuration's constraints.
    """
    service = ConfigService(db)

    try:
        config = await service.set_value(key, request.value)
        return ConfigUpdateResponse(key=key, value=config.value, updated=True)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
