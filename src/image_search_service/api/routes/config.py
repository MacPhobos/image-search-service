"""Configuration API routes.

Provides endpoints for reading and updating system configuration values,
with a specialized endpoint for face matching settings.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.ext.asyncio import AsyncSession

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

    @model_validator(mode="after")
    def validate_thresholds(self) -> "FaceMatchingConfigUpdateRequest":
        """Ensure suggestion_threshold < auto_assign_threshold."""
        if self.suggestion_threshold >= self.auto_assign_threshold:
            raise ValueError(
                "suggestion_threshold must be less than auto_assign_threshold"
            )
        return self


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
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(
        f"Updated face matching config: auto={request.auto_assign_threshold}, "
        f"suggestion={request.suggestion_threshold}, "
        f"prototype_min_quality={request.prototype_min_quality}, "
        f"prototype_max_exemplars={request.prototype_max_exemplars}"
    )

    return FaceMatchingConfigResponse(
        auto_assign_threshold=request.auto_assign_threshold,
        suggestion_threshold=request.suggestion_threshold,
        max_suggestions=request.max_suggestions,
        suggestion_expiry_days=request.suggestion_expiry_days,
        prototype_min_quality=request.prototype_min_quality,
        prototype_max_exemplars=request.prototype_max_exemplars,
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
