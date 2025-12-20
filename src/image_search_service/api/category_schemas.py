"""Pydantic schemas for category API."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class CategoryCreate(BaseModel):
    """Request schema for creating a category."""

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(max_length=100, description="Category name")
    description: str | None = Field(None, description="Category description")
    color: str | None = Field(
        None, pattern=r"^#[0-9A-Fa-f]{6}$", description="Hex color code"
    )


class CategoryUpdate(BaseModel):
    """Request schema for updating a category."""

    model_config = ConfigDict(populate_by_name=True)

    name: str | None = Field(None, max_length=100)
    description: str | None = None
    color: str | None = Field(None, pattern=r"^#[0-9A-Fa-f]{6}$")


class CategoryResponse(BaseModel):
    """Response schema for category."""

    model_config = ConfigDict(populate_by_name=True, from_attributes=True)

    id: int
    name: str
    description: str | None = None
    color: str | None = None
    is_default: bool = Field(alias="isDefault", serialization_alias="isDefault")
    created_at: datetime = Field(alias="createdAt", serialization_alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt", serialization_alias="updatedAt")
    session_count: int | None = Field(
        None, alias="sessionCount", serialization_alias="sessionCount"
    )
