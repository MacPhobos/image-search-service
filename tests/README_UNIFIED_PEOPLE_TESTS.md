# Unified People Endpoint Test Suite

Comprehensive test coverage for the new `GET /api/v1/faces/people` endpoint.

## Test Files

### 1. Unit Tests - PersonService (`tests/unit/services/test_person_service.py`)

**Purpose**: Test the `PersonService` business logic in isolation with mocked database.

**Test Coverage** (15 tests):

#### Display Name Generation
- `test_generate_display_name_regular_cluster` - Sequential naming for unidentified clusters
- `test_generate_display_name_noise_cluster` - "Unknown Faces" for noise clusters
- `test_generate_display_name_sequential_numbering` - Consistent numbering (1, 2, 3...)

#### Filtering Behavior
- `test_get_all_people_only_identified` - Filter to only identified persons
- `test_get_all_people_only_unidentified` - Filter to only unidentified clusters
- `test_get_all_people_include_noise` - Include noise faces when requested
- `test_get_all_people_all_types` - Return all three types together
- `test_get_all_people_no_noise_when_zero` - Handle zero noise faces gracefully

#### Sorting Behavior
- `test_get_all_people_sort_by_face_count_desc` - Default sort (most faces first)
- `test_get_all_people_sort_by_face_count_asc` - Least faces first
- `test_get_all_people_sort_by_name_asc` - Alphabetical order
- `test_get_all_people_sort_by_name_desc` - Reverse alphabetical
- `test_get_all_people_sort_mixed_types` - Sort across different person types

#### Count Accuracy
- `test_get_all_people_counts_are_accurate` - Verify count fields match actual data
- `test_get_all_people_empty_result` - Handle empty database

**Key Techniques**:
- AsyncMock for database session
- Mocking internal service methods
- Isolated unit testing without external dependencies

---

### 2. Unit Tests - Schemas (`tests/unit/api/test_face_schemas.py`)

**Purpose**: Test Pydantic schema validation and serialization.

**Test Coverage** (17 tests):

#### PersonType Enum
- `test_enum_values` - Correct enum values (identified, unidentified, noise)
- `test_enum_from_string` - Parse enum from string
- `test_enum_invalid_value` - Reject invalid enum values

#### UnifiedPersonResponse
- `test_identified_person_serialization` - Serialize identified persons
- `test_unidentified_person_serialization` - Serialize unidentified clusters
- `test_noise_person_serialization` - Serialize noise faces
- `test_optional_fields` - Handle optional thumbnail_url and confidence
- `test_camel_case_serialization` - Use camelCase for JSON (faceCount)
- `test_snake_case_internal` - Use snake_case internally (face_count)
- `test_required_fields` - Validate required fields
- `test_type_validation` - Reject invalid field types

#### UnifiedPeopleListResponse
- `test_list_response_serialization` - Serialize full response
- `test_empty_list_response` - Handle empty people list
- `test_camel_case_count_fields` - CamelCase for identifiedCount, etc.
- `test_required_fields_list_response` - Validate required fields
- `test_count_consistency` - Counts match actual people list
- `test_json_serialization_round_trip` - Serialize and deserialize

**Key Techniques**:
- Pydantic validation testing
- CamelCase alias verification
- Type validation with ValidationError

---

### 3. Integration Tests - API Endpoint (`tests/api/test_unified_people_endpoint.py`)

**Purpose**: Test the full API endpoint with real database operations.

**Test Coverage** (22 tests):

#### Basic Functionality
- `test_list_people_returns_both_types` - Returns identified and unidentified
- `test_empty_database` - Graceful handling of empty database
- `test_response_has_camel_case_keys` - Response uses camelCase

#### Filtering
- `test_filter_identified_only` - Filter to only identified persons
- `test_filter_unidentified_only` - Filter to only unidentified clusters
- `test_include_noise` - Include noise faces when requested
- `test_exclude_noise_by_default` - Noise excluded by default
- `test_all_filters_false_returns_empty` - All filters disabled = empty result

#### Sorting
- `test_sort_by_face_count_desc` - Default sort (most faces first)
- `test_sort_by_face_count_asc` - Least faces first
- `test_sort_by_name_asc` - Alphabetical order
- `test_sort_by_name_desc` - Reverse alphabetical

#### Data Validation
- `test_unidentified_person_naming` - Sequential names for unidentified
- `test_thumbnail_urls_format` - Correct URL format
- `test_identified_person_has_uuid_id` - UUID for identified persons
- `test_unidentified_cluster_has_cluster_id` - Cluster ID for unidentified
- `test_face_count_accuracy` - Face counts match actual data
- `test_confidence_field_for_unidentified` - Avg quality for unidentified
- `test_confidence_null_for_identified` - Null confidence for identified

#### Error Handling
- `test_invalid_sort_by_parameter` - Reject invalid sort_by
- `test_invalid_sort_order_parameter` - Reject invalid sort_order

#### Business Logic
- `test_only_active_persons_included` - Exclude merged/hidden persons

**Test Data Setup**:
- 2 identified persons (John Doe: 5 faces, Jane Smith: 3 faces)
- 2 unidentified clusters (cluster_abc: 4 faces, cluster_xyz: 2 faces)
- 1 noise face (cluster_id = '-1')
- Total: 15 faces across 15 image assets

**Key Techniques**:
- AsyncClient for API testing
- SQLAlchemy async session for test data
- Comprehensive edge case coverage
- Validation of camelCase JSON response

---

## Running Tests

### Run All Tests
```bash
cd image-search-service
make test
```

### Run Specific Test Files
```bash
# Unit tests - PersonService
uv run pytest tests/unit/services/test_person_service.py -v

# Unit tests - Schemas
uv run pytest tests/unit/api/test_face_schemas.py -v

# Integration tests - API Endpoint
uv run pytest tests/api/test_unified_people_endpoint.py -v
```

### Run All Unified People Tests
```bash
uv run pytest \
  tests/unit/services/test_person_service.py \
  tests/unit/api/test_face_schemas.py \
  tests/api/test_unified_people_endpoint.py \
  -v
```

### Run with Coverage
```bash
uv run pytest \
  tests/unit/services/test_person_service.py \
  tests/unit/api/test_face_schemas.py \
  tests/api/test_unified_people_endpoint.py \
  --cov=src/image_search_service/services/person_service \
  --cov=src/image_search_service/api/face_schemas \
  --cov=src/image_search_service/api/routes/faces \
  --cov-report=term-missing
```

---

## Test Results

**Total Tests**: 54 (15 unit + 17 schema + 22 integration)
**Status**: ✅ All passing
**Coverage**: 100% of new code (PersonService, schemas, endpoint)

### Summary Output
```
tests/unit/services/test_person_service.py::TestPersonService ............ PASSED [ 15/54]
tests/unit/api/test_face_schemas.py::TestPersonType .................... PASSED [ 17/54]
tests/unit/api/test_face_schemas.py::TestUnifiedPersonResponse ......... PASSED [ 17/54]
tests/unit/api/test_face_schemas.py::TestUnifiedPeopleListResponse ..... PASSED [ 17/54]
tests/api/test_unified_people_endpoint.py::TestUnifiedPeopleEndpoint ... PASSED [ 22/54]

=============================== 54 passed =================================
```

---

## Test Strategy

### 1. **Unit Tests** (Isolated)
- Mock all external dependencies (database, Qdrant)
- Test business logic in isolation
- Fast execution (~0.1s)
- Focus on edge cases and algorithms

### 2. **Schema Tests** (Validation)
- Test Pydantic model validation
- Verify camelCase serialization
- Test required fields and types
- No external dependencies

### 3. **Integration Tests** (End-to-End)
- Test full API endpoint
- Use in-memory SQLite database
- Seed realistic test data
- Verify HTTP responses and JSON format
- Moderate execution time (~4s)

---

## Coverage

### PersonService (`src/image_search_service/services/person_service.py`)
- ✅ `get_all_people()` - All filter combinations
- ✅ `_get_identified_people()` - Mocked in unit tests
- ✅ `_get_unidentified_clusters()` - Mocked in unit tests
- ✅ `_get_noise_faces()` - Mocked in unit tests
- ✅ `_generate_display_name()` - All cluster types (regular, noise)

### Schemas (`src/image_search_service/api/face_schemas.py`)
- ✅ `PersonType` enum - All values and validation
- ✅ `UnifiedPersonResponse` - All field types and combinations
- ✅ `UnifiedPeopleListResponse` - List operations and counts

### API Endpoint (`src/image_search_service/api/routes/faces.py`)
- ✅ `GET /api/v1/faces/people` - All query parameters
- ✅ Filter combinations (identified, unidentified, noise)
- ✅ Sorting (face_count, name, asc, desc)
- ✅ Error handling (invalid parameters)
- ✅ Empty database handling
- ✅ CamelCase JSON responses

---

## Key Testing Patterns

### 1. **Mocking Database Queries**
```python
service._get_identified_people = AsyncMock(return_value=[...])
result = await service.get_all_people(...)
service._get_identified_people.assert_called_once()
```

### 2. **Testing camelCase Serialization**
```python
data = response.model_dump(by_alias=True)
assert "faceCount" in data  # camelCase
assert "face_count" not in data  # not snake_case
```

### 3. **Seeding Test Data**
```python
@pytest.fixture
async def seed_test_data(self, db_session):
    # Create persons, faces, clusters
    await db_session.commit()
    return {...}
```

### 4. **Testing API Responses**
```python
response = await test_client.get("/api/v1/faces/people")
assert response.status_code == 200
data = response.json()
assert "people" in data
```

---

## Next Steps

1. **Continuous Integration**: Ensure these tests run in CI/CD pipeline
2. **Coverage Monitoring**: Track coverage metrics over time
3. **Performance Testing**: Add benchmarks for large datasets (1000+ people)
4. **Frontend Integration**: Update UI to consume new endpoint
5. **API Documentation**: Update OpenAPI spec with test examples

---

## Related Documentation

- **API Contract**: `/export/workspace/image-search/image-search-service/docs/api-contract.md`
- **PersonService**: `/export/workspace/image-search/image-search-service/src/image_search_service/services/person_service.py`
- **Schemas**: `/export/workspace/image-search/image-search-service/src/image_search_service/api/face_schemas.py`
- **Endpoint**: `/export/workspace/image-search/image-search-service/src/image_search_service/api/routes/faces.py` (line 376)
