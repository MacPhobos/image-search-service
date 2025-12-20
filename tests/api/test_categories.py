"""Test category management endpoints."""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from image_search_service.db.models import Category, TrainingSession


@pytest.fixture
async def default_category(db_session: AsyncSession) -> Category:
    """Create default category for tests."""
    category = Category(
        name="General",
        description="Default category",
        is_default=True,
    )
    db_session.add(category)
    await db_session.commit()
    await db_session.refresh(category)
    return category


@pytest.fixture
async def test_category(db_session: AsyncSession) -> Category:
    """Create a test category."""
    category = Category(
        name="Test Category",
        description="For testing",
        color="#3B82F6",
        is_default=False,
    )
    db_session.add(category)
    await db_session.commit()
    await db_session.refresh(category)
    return category


@pytest.fixture
async def category_with_session(
    db_session: AsyncSession, test_category: Category
) -> tuple[Category, TrainingSession]:
    """Create a category with an associated training session."""
    session = TrainingSession(
        name="Test Session",
        root_path="/test/path",
        category_id=test_category.id,
    )
    db_session.add(session)
    await db_session.commit()
    await db_session.refresh(session)
    return test_category, session


# List Categories Tests


@pytest.mark.asyncio
async def test_list_categories_empty(test_client: AsyncClient) -> None:
    """Test that empty list returns empty items."""
    response = await test_client.get("/api/v1/categories")

    assert response.status_code == 200
    data = response.json()

    assert data["items"] == []
    assert data["total"] == 0
    assert data["page"] == 1
    assert data["pageSize"] == 50
    assert data["hasMore"] is False


@pytest.mark.asyncio
async def test_list_categories_returns_correct_data(
    test_client: AsyncClient, test_category: Category
) -> None:
    """Test list with categories returns correct count and data."""
    response = await test_client.get("/api/v1/categories")

    assert response.status_code == 200
    data = response.json()

    assert data["total"] == 1
    assert len(data["items"]) == 1

    item = data["items"][0]
    assert item["id"] == test_category.id
    assert item["name"] == test_category.name
    assert item["description"] == test_category.description
    assert item["color"] == test_category.color
    assert item["isDefault"] is False
    assert item["sessionCount"] == 0


@pytest.mark.asyncio
async def test_list_categories_includes_session_count(
    test_client: AsyncClient, category_with_session: tuple[Category, TrainingSession]
) -> None:
    """Test that list includes session count for categories."""
    category, _ = category_with_session

    response = await test_client.get("/api/v1/categories")

    assert response.status_code == 200
    data = response.json()

    item = data["items"][0]
    assert item["id"] == category.id
    assert item["sessionCount"] == 1


@pytest.mark.asyncio
async def test_list_categories_pagination_works(
    test_client: AsyncClient, db_session: AsyncSession
) -> None:
    """Test pagination works with page and page_size."""
    # Create multiple categories
    for i in range(5):
        category = Category(
            name=f"Category {i}",
            description=f"Description {i}",
        )
        db_session.add(category)
    await db_session.commit()

    # Test page 1 with page_size 2
    response = await test_client.get("/api/v1/categories?page=1&page_size=2")
    assert response.status_code == 200
    data = response.json()

    assert data["total"] == 5
    assert len(data["items"]) == 2
    assert data["page"] == 1
    assert data["pageSize"] == 2
    assert data["hasMore"] is True

    # Test page 2
    response = await test_client.get("/api/v1/categories?page=2&page_size=2")
    assert response.status_code == 200
    data = response.json()

    assert len(data["items"]) == 2
    assert data["page"] == 2
    assert data["hasMore"] is True

    # Test page 3 (last page)
    response = await test_client.get("/api/v1/categories?page=3&page_size=2")
    assert response.status_code == 200
    data = response.json()

    assert len(data["items"]) == 1
    assert data["page"] == 3
    assert data["hasMore"] is False


# Create Category Tests


@pytest.mark.asyncio
async def test_create_category_returns_201(test_client: AsyncClient) -> None:
    """Test creating valid category returns 201."""
    response = await test_client.post(
        "/api/v1/categories",
        json={
            "name": "New Category",
            "description": "A new category",
            "color": "#FF5733",
        },
    )

    assert response.status_code == 201
    data = response.json()

    assert data["name"] == "New Category"
    assert data["description"] == "A new category"
    assert data["color"] == "#FF5733"
    assert data["isDefault"] is False
    assert data["sessionCount"] == 0
    assert "id" in data
    assert "createdAt" in data
    assert "updatedAt" in data


@pytest.mark.asyncio
async def test_create_category_minimal_fields(test_client: AsyncClient) -> None:
    """Test created category has correct fields with minimal data."""
    response = await test_client.post(
        "/api/v1/categories",
        json={"name": "Minimal"},
    )

    assert response.status_code == 201
    data = response.json()

    assert data["name"] == "Minimal"
    assert data["description"] is None
    assert data["color"] is None
    assert data["isDefault"] is False


@pytest.mark.asyncio
async def test_create_category_duplicate_name_returns_409(
    test_client: AsyncClient, test_category: Category
) -> None:
    """Test duplicate name returns 409."""
    response = await test_client.post(
        "/api/v1/categories",
        json={"name": test_category.name},
    )

    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]


@pytest.mark.asyncio
async def test_create_category_invalid_color_returns_422(test_client: AsyncClient) -> None:
    """Test invalid color format returns 422."""
    response = await test_client.post(
        "/api/v1/categories",
        json={
            "name": "Bad Color",
            "color": "not-a-color",
        },
    )

    assert response.status_code == 422


# Get Category Tests


@pytest.mark.asyncio
async def test_get_category_returns_200_with_session_count(
    test_client: AsyncClient, category_with_session: tuple[Category, TrainingSession]
) -> None:
    """Test get existing category returns 200 with session_count."""
    category, _ = category_with_session

    response = await test_client.get(f"/api/v1/categories/{category.id}")

    assert response.status_code == 200
    data = response.json()

    assert data["id"] == category.id
    assert data["name"] == category.name
    assert data["sessionCount"] == 1


@pytest.mark.asyncio
async def test_get_category_zero_sessions(
    test_client: AsyncClient, test_category: Category
) -> None:
    """Test get category with zero sessions."""
    response = await test_client.get(f"/api/v1/categories/{test_category.id}")

    assert response.status_code == 200
    data = response.json()

    assert data["sessionCount"] == 0


@pytest.mark.asyncio
async def test_get_category_not_found_returns_404(test_client: AsyncClient) -> None:
    """Test get non-existent category returns 404."""
    response = await test_client.get("/api/v1/categories/99999")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


# Update Category Tests


@pytest.mark.asyncio
async def test_update_category_name_works(
    test_client: AsyncClient, test_category: Category
) -> None:
    """Test update name works."""
    response = await test_client.patch(
        f"/api/v1/categories/{test_category.id}",
        json={"name": "Updated Name"},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["name"] == "Updated Name"
    assert data["description"] == test_category.description
    assert data["color"] == test_category.color


@pytest.mark.asyncio
async def test_update_category_partial_update_color_only(
    test_client: AsyncClient, test_category: Category
) -> None:
    """Test partial update (only color) works."""
    original_name = test_category.name

    response = await test_client.patch(
        f"/api/v1/categories/{test_category.id}",
        json={"color": "#00FF00"},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["name"] == original_name  # Unchanged
    assert data["color"] == "#00FF00"  # Changed


@pytest.mark.asyncio
async def test_update_category_all_fields(
    test_client: AsyncClient, test_category: Category
) -> None:
    """Test updating all fields at once."""
    response = await test_client.patch(
        f"/api/v1/categories/{test_category.id}",
        json={
            "name": "Completely New",
            "description": "New description",
            "color": "#ABCDEF",
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["name"] == "Completely New"
    assert data["description"] == "New description"
    assert data["color"] == "#ABCDEF"


@pytest.mark.asyncio
async def test_update_category_not_found_returns_404(test_client: AsyncClient) -> None:
    """Test update non-existent returns 404."""
    response = await test_client.patch(
        "/api/v1/categories/99999",
        json={"name": "Does not matter"},
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


@pytest.mark.asyncio
async def test_update_category_duplicate_name_returns_409(
    test_client: AsyncClient, db_session: AsyncSession, test_category: Category
) -> None:
    """Test update to duplicate name returns 409."""
    # Create another category
    other_category = Category(name="Other Category", description="Other")
    db_session.add(other_category)
    await db_session.commit()
    await db_session.refresh(other_category)

    # Try to rename test_category to other_category's name
    response = await test_client.patch(
        f"/api/v1/categories/{test_category.id}",
        json={"name": other_category.name},
    )

    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]


# Delete Category Tests


@pytest.mark.asyncio
async def test_delete_empty_category_returns_204(
    test_client: AsyncClient, test_category: Category
) -> None:
    """Test delete empty category returns 204."""
    response = await test_client.delete(f"/api/v1/categories/{test_category.id}")

    assert response.status_code == 204

    # Verify category is deleted
    get_response = await test_client.get(f"/api/v1/categories/{test_category.id}")
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_delete_category_not_found_returns_404(test_client: AsyncClient) -> None:
    """Test delete non-existent returns 404."""
    response = await test_client.delete("/api/v1/categories/99999")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


@pytest.mark.asyncio
async def test_delete_default_category_returns_400(
    test_client: AsyncClient, default_category: Category
) -> None:
    """Test delete default category returns 400."""
    response = await test_client.delete(f"/api/v1/categories/{default_category.id}")

    assert response.status_code == 400
    assert "Cannot delete default category" in response.json()["detail"]


@pytest.mark.asyncio
async def test_delete_category_with_sessions_returns_409(
    test_client: AsyncClient, category_with_session: tuple[Category, TrainingSession]
) -> None:
    """Test delete category with sessions returns 409."""
    category, _ = category_with_session

    response = await test_client.delete(f"/api/v1/categories/{category.id}")

    assert response.status_code == 409
    assert "Cannot delete category with" in response.json()["detail"]
    assert "training session" in response.json()["detail"]
