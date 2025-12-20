"""Category management endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.api.category_schemas import (
    CategoryCreate,
    CategoryResponse,
    CategoryUpdate,
)
from image_search_service.api.schemas import PaginatedResponse
from image_search_service.core.logging import get_logger
from image_search_service.db.models import Category, TrainingSession
from image_search_service.db.session import get_db

logger = get_logger(__name__)
router = APIRouter(prefix="/categories", tags=["categories"])


@router.get("", response_model=PaginatedResponse[CategoryResponse])
async def list_categories(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> PaginatedResponse[CategoryResponse]:
    """List all categories with pagination.

    Args:
        page: Page number (1-indexed)
        page_size: Number of items per page (max 100)
        db: Database session

    Returns:
        Paginated list of categories with session counts
    """
    # Query with session count
    query = (
        select(
            Category,
            func.count(TrainingSession.id).label("session_count"),
        )
        .outerjoin(TrainingSession, Category.id == TrainingSession.category_id)
        .group_by(Category.id)
        .order_by(Category.created_at.desc())
    )

    # Get total count
    count_query = select(func.count()).select_from(Category)
    total_result = await db.execute(count_query)
    total = total_result.scalar_one()

    # Apply pagination
    offset = (page - 1) * page_size
    paginated_query = query.offset(offset).limit(page_size)
    result = await db.execute(paginated_query)
    rows = result.all()

    # Build response with session counts
    items = []
    for category, session_count in rows:
        category_dict = CategoryResponse.model_validate(category).model_dump()
        category_dict["sessionCount"] = session_count
        items.append(CategoryResponse.model_validate(category_dict))

    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        pageSize=page_size,
        hasMore=(page * page_size) < total,
    )


@router.post("", response_model=CategoryResponse, status_code=status.HTTP_201_CREATED)
async def create_category(
    data: CategoryCreate, db: AsyncSession = Depends(get_db)
) -> CategoryResponse:
    """Create a new category.

    Args:
        data: Category creation data
        db: Database session

    Returns:
        Created category

    Raises:
        HTTPException: 409 if category name already exists
    """
    category = Category(
        name=data.name,
        description=data.description,
        color=data.color,
    )

    db.add(category)

    try:
        await db.commit()
        await db.refresh(category)

        # Add session_count field
        response = CategoryResponse.model_validate(category)
        response.session_count = 0
        return response

    except IntegrityError as e:
        await db.rollback()
        logger.warning(f"Category name conflict: {data.name}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Category with name '{data.name}' already exists",
        ) from e


@router.get("/{category_id}", response_model=CategoryResponse)
async def get_category(
    category_id: int, db: AsyncSession = Depends(get_db)
) -> CategoryResponse:
    """Get a single category by ID.

    Args:
        category_id: Category ID
        db: Database session

    Returns:
        Category details with session count

    Raises:
        HTTPException: 404 if category not found
    """
    # Query with session count
    query = (
        select(
            Category,
            func.count(TrainingSession.id).label("session_count"),
        )
        .outerjoin(TrainingSession, Category.id == TrainingSession.category_id)
        .where(Category.id == category_id)
        .group_by(Category.id)
    )

    result = await db.execute(query)
    row = result.one_or_none()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Category {category_id} not found",
        )

    category, session_count = row
    category_dict = CategoryResponse.model_validate(category).model_dump()
    category_dict["sessionCount"] = session_count

    return CategoryResponse.model_validate(category_dict)


@router.patch("/{category_id}", response_model=CategoryResponse)
async def update_category(
    category_id: int,
    data: CategoryUpdate,
    db: AsyncSession = Depends(get_db),
) -> CategoryResponse:
    """Update a category.

    Args:
        category_id: Category ID
        data: Update data
        db: Database session

    Returns:
        Updated category

    Raises:
        HTTPException: 404 if category not found, 409 if duplicate name
    """
    # Get category
    query = select(Category).where(Category.id == category_id)
    result = await db.execute(query)
    category = result.scalar_one_or_none()

    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Category {category_id} not found",
        )

    # Update fields (exclude unset to allow partial updates)
    update_data = data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(category, field, value)

    try:
        await db.commit()
        await db.refresh(category)

        # Get session count
        count_query = (
            select(func.count())
            .select_from(TrainingSession)
            .where(TrainingSession.category_id == category_id)
        )
        count_result = await db.execute(count_query)
        session_count = count_result.scalar_one()

        response = CategoryResponse.model_validate(category)
        response.session_count = session_count
        return response

    except IntegrityError as e:
        await db.rollback()
        logger.warning(f"Category name conflict during update: {data.name}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Category with name '{data.name}' already exists",
        ) from e


@router.delete("/{category_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_category(
    category_id: int, db: AsyncSession = Depends(get_db)
) -> None:
    """Delete a category.

    Args:
        category_id: Category ID
        db: Database session

    Raises:
        HTTPException: 404 if not found, 400 if default category, 409 if has sessions
    """
    # Get category
    query = select(Category).where(Category.id == category_id)
    result = await db.execute(query)
    category = result.scalar_one_or_none()

    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Category {category_id} not found",
        )

    # Prevent deletion of default category
    if category.is_default:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete default category",
        )

    # Check for sessions using this category
    count_query = (
        select(func.count())
        .select_from(TrainingSession)
        .where(TrainingSession.category_id == category_id)
    )
    count_result = await db.execute(count_query)
    session_count = count_result.scalar_one()

    if session_count > 0:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot delete category with {session_count} training session(s)",
        )

    # Delete category
    await db.delete(category)
    await db.commit()

    logger.info(f"Deleted category {category_id} ({category.name})")
