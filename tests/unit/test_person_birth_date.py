"""Tests for person birth_date feature and age calculation."""

from datetime import date, datetime

from image_search_service.api.routes.faces import calculate_age_at_date


class TestCalculateAgeAtDate:
    """Tests for age calculation helper function."""

    def test_basic_age_calculation(self):
        """Test basic age calculation with full years."""
        birth_date = date(1990, 6, 15)
        photo_date = datetime(2020, 8, 1)
        assert calculate_age_at_date(birth_date, photo_date) == 30

    def test_age_before_birthday_this_year(self):
        """Test age calculation when birthday hasn't occurred yet in photo year."""
        birth_date = date(1990, 6, 15)
        photo_date = datetime(2020, 3, 1)  # Before June 15
        assert calculate_age_at_date(birth_date, photo_date) == 29

    def test_age_on_birthday(self):
        """Test age calculation on exact birthday."""
        birth_date = date(1990, 6, 15)
        photo_date = datetime(2020, 6, 15)
        assert calculate_age_at_date(birth_date, photo_date) == 30

    def test_age_after_birthday(self):
        """Test age calculation after birthday has passed."""
        birth_date = date(1990, 6, 15)
        photo_date = datetime(2020, 12, 31)
        assert calculate_age_at_date(birth_date, photo_date) == 30

    def test_none_birth_date(self):
        """Test that None birth_date returns None."""
        photo_date = datetime(2020, 1, 1)
        assert calculate_age_at_date(None, photo_date) is None

    def test_none_photo_date(self):
        """Test that None photo_date returns None."""
        birth_date = date(1990, 6, 15)
        assert calculate_age_at_date(birth_date, None) is None

    def test_both_none(self):
        """Test that both None returns None."""
        assert calculate_age_at_date(None, None) is None

    def test_infant_age(self):
        """Test age calculation for infants."""
        birth_date = date(2019, 1, 1)
        photo_date = datetime(2020, 6, 1)
        assert calculate_age_at_date(birth_date, photo_date) == 1

    def test_zero_age(self):
        """Test age calculation for photos taken in birth year before birthday."""
        birth_date = date(2020, 6, 15)
        photo_date = datetime(2020, 3, 1)
        assert calculate_age_at_date(birth_date, photo_date) == 0

    def test_leap_year_birthday(self):
        """Test age calculation for leap year birthdays."""
        birth_date = date(2000, 2, 29)
        photo_date = datetime(2020, 3, 1)
        assert calculate_age_at_date(birth_date, photo_date) == 20

    def test_elderly_age(self):
        """Test age calculation for elderly persons."""
        birth_date = date(1930, 1, 1)
        photo_date = datetime(2020, 12, 31)
        assert calculate_age_at_date(birth_date, photo_date) == 90

    def test_negative_age_returns_zero(self):
        """Test that photo taken before birth returns 0 (edge case)."""
        birth_date = date(2020, 1, 1)
        photo_date = datetime(2019, 6, 1)
        # Should never happen in real data, but function should handle gracefully
        assert calculate_age_at_date(birth_date, photo_date) == 0


# Integration tests for PATCH endpoint would go here
# These require async test fixtures and a test database
# Example structure:
#
# @pytest.mark.asyncio
# async def test_update_person_birth_date(async_client, test_db):
#     """Test updating person's birth_date via PATCH endpoint."""
#     # Create test person
#     # PATCH with birth_date
#     # Verify updated
#
# @pytest.mark.asyncio
# async def test_update_person_name_only(async_client, test_db):
#     """Test updating person's name only."""
#     pass
#
# @pytest.mark.asyncio
# async def test_person_photos_includes_age(async_client, test_db):
#     """Test that person photos endpoint includes age_at_photo."""
#     pass
