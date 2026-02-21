# Alembic Migration `version_num` Column Truncation Error -- Analysis

**Date**: 2026-02-21
**Status**: Research Complete
**Affected Environment**: tauceti (fresh database)
**Working Environment**: hyperion (incrementally migrated)

---

## 1. Problem Statement

Running `make migrate` (`uv run alembic upgrade head`) on a fresh database (tauceti) fails at migration `013_add_discovering_session_status` with:

```
asyncpg.exceptions.StringDataRightTruncationError: value too long for type character varying(32)
[SQL: UPDATE alembic_version SET version_num='013_add_discovering_session_status'
 WHERE alembic_version.version_num = 'fcc7bcef2a95']
```

The revision ID `013_add_discovering_session_status` is **34 characters**, exceeding the default `alembic_version.version_num VARCHAR(32)` column limit.

---

## 2. Complete Migration Chain

The full migration chain from base to head (36 migrations total):

| # | Revision ID | Length | Exceeds 32? | Down Revision | Type |
|---|-------------|--------|-------------|---------------|------|
| 1 | `001` | 3 | No | `None` | Manual numeric |
| 2 | `ce719ca53e7b` | 12 | No | `001` | Auto-generated hex |
| 3 | `002` | 3 | No | `ce719ca53e7b` | Manual numeric |
| 4 | `003` | 3 | No | `002` | Manual numeric |
| 5 | `004` | 3 | No | `003` | Manual numeric |
| 6 | `005` | 3 | No | `004` | Manual numeric |
| 7 | `006` | 3 | No | `005` | Manual numeric |
| 8 | `007` | 3 | No | `006` | Manual numeric |
| 9 | `008` | 3 | No | `007` | Manual numeric |
| 10 | `009` | 3 | No | `008` | Manual numeric |
| 11 | `e7ccebcb4be7` | 12 | No | `009` | Auto-generated hex |
| 12 | `0ea4fdf7fc41` | 12 | No | `e7ccebcb4be7` | Auto-generated hex |
| 13 | `56f6544da217` | 12 | No | `0ea4fdf7fc41` | Auto-generated hex |
| 14 | `a1b2c3d4e5f6` | 12 | No | `56f6544da217` | Auto-generated hex |
| 15 | `974bfe0f68ed` | 12 | No | `a1b2c3d4e5f6` | Auto-generated hex |
| 16 | `0d2febc7f1d5` | 12 | No | `974bfe0f68ed` | Auto-generated hex |
| 17 | `a5e555cf5477` | 12 | No | `0d2febc7f1d5` | Auto-generated hex |
| 18 | `9511886120f4` | 12 | No | `a5e555cf5477` | Auto-generated hex |
| 19 | `temporal_proto_001` | 18 | No | `9511886120f4` | Manual descriptive |
| 20 | `010` | 3 | No | `temporal_proto_001` | Manual numeric |
| 21 | `8d46a4ba4167` | 12 | No | `010` | Auto-generated hex |
| 22 | `f6a668d072bb` | 12 | No | `8d46a4ba4167` | Auto-generated hex |
| 23a | `a8b9c0d1e2f3` | 12 | No | `f6a668d072bb` | Auto-generated hex (branch A) |
| 23b | `c1d2e3f4g5h6` | 12 | No | `f6a668d072bb` | Auto-generated hex (branch B) |
| 24a | `b9c0d1e2f3g4` | 12 | No | `a8b9c0d1e2f3` | Auto-generated hex (branch A) |
| 25 | `737ef70e7bab` | 12 | No | `(b9c0d1e2f3g4, c1d2e3f4g5h6)` | Merge migration |
| 26 | `011_hash_dedup_fields` | 21 | No | `737ef70e7bab` | Manual descriptive |
| 27 | `012_post_train_suggestions` | 27 | No | `011_hash_dedup_fields` | Manual descriptive |
| 28 | `d1e2f3g4h5i6` | 12 | No | `012_post_train_suggestions` | Auto-generated hex |
| 29 | `c3198aefbaa4` | 12 | No | `d1e2f3g4h5i6` | Auto-generated hex |
| 30 | `cdfb76610d90` | 12 | No | `c3198aefbaa4` | Auto-generated hex |
| 31 | `df737c255b50` | 12 | No | `cdfb76610d90` | Auto-generated hex |
| 32 | `5b9f75181a7d` | 12 | No | `df737c255b50` | Auto-generated hex |
| 33 | `34633924ee16` | 12 | No | `5b9f75181a7d` | Auto-generated hex |
| 34 | `fcc7bcef2a95` | 12 | No | `34633924ee16` | Auto-generated hex |
| **35** | **`013_add_discovering_session_status`** | **34** | **YES** | `fcc7bcef2a95` | **Manual descriptive** |

### Revision IDs Exceeding 32 Characters

| Revision ID | Length | Characters Over |
|-------------|--------|----------------|
| `013_add_discovering_session_status` | 34 | +2 over limit |

### Revision IDs Approaching the 32-Character Limit (>20 chars)

| Revision ID | Length | Margin |
|-------------|--------|--------|
| `012_post_train_suggestions` | 27 | 5 chars remaining |
| `011_hash_dedup_fields` | 21 | 11 chars remaining |

### Down-Revision References to Long IDs

These migrations reference long revision IDs as their `down_revision`, which also get written to `alembic_version`:

| File | Down Revision Referenced | Length |
|------|------------------------|--------|
| `012_post_train_suggestions.py` | `011_hash_dedup_fields` | 21 (safe) |
| `d1e2f3g4h5i6_add_person_centroid_table.py` | `012_post_train_suggestions` | 27 (safe) |

Note: Down-revision values are only compared against the current `version_num` in the table; the critical INSERT/UPDATE happens with the *current* migration's revision ID. So the only ID that causes a truncation error is `013_add_discovering_session_status` (34 chars).

---

## 3. Alembic Configuration Analysis

### alembic.ini

- **File**: `/export/workspace/image-search/image-search-service/alembic.ini`
- **No `version_num_width` setting** -- this parameter does not exist in Alembic
- **No custom `file_template`** configured (the default `%%(rev)s_%%(slug)s` is used)
- Script location: `src/image_search_service/db/migrations`

### env.py

- **File**: `src/image_search_service/db/migrations/env.py`
- **No `version_table` override** -- uses default `alembic_version`
- **No `version_table_schema` override**
- **No `version_table_pk` override**
- Uses async engine via `async_engine_from_config`
- The `context.configure()` calls pass only `url`, `target_metadata`, `literal_binds`, `dialect_opts`, and `connection` -- no version table customization

### script.py.mako

- Standard template, uses `${repr(up_revision)}` for the revision ID
- No custom revision ID logic

### Installed Alembic Version

- **Alembic 1.17.2** -- this version has the `version_table_impl` hook (added in 1.14)

---

## 4. Root Cause Analysis

### Why the Column is VARCHAR(32)

The `alembic_version.version_num` column is created as `VARCHAR(32)` because it is **hardcoded** in the Alembic source code:

```python
# alembic/ddl/impl.py -> DefaultImpl.version_table_impl()
vt = Table(
    version_table,
    MetaData(),
    Column("version_num", String(32), nullable=False),
    schema=version_table_schema,
)
```

There is **no configuration parameter** to change this width. The only way to customize it is:
1. Override `version_table_impl` in a custom dialect implementation (since Alembic 1.14)
2. Manually ALTER the table after creation
3. Pre-create the table with a wider column before running migrations

### Why Some Migrations Have Long IDs

The project uses **three distinct naming conventions** for revision IDs:

**Pattern 1: Manual numeric IDs** (early migrations)
- Examples: `001`, `002`, ..., `010`
- Length: 3 characters
- Created manually, not via `alembic revision --autogenerate`

**Pattern 2: Auto-generated hex hashes** (most migrations)
- Examples: `ce719ca53e7b`, `fcc7bcef2a95`, `737ef70e7bab`
- Length: 12 characters
- Created via `make makemigrations` (`alembic revision --autogenerate`)

**Pattern 3: Manual descriptive IDs** (recent migrations)
- Examples: `temporal_proto_001` (18), `011_hash_dedup_fields` (21), `012_post_train_suggestions` (27), `013_add_discovering_session_status` (34)
- Length: 18-34 characters
- Created manually with human-readable revision IDs

The shift to Pattern 3 began around December 2025 / January 2026. The descriptive IDs follow a convention of `NNN_description` where NNN is a sequential number and description summarizes the change. This is a reasonable convention for readability but **was not validated against the 32-character limit**.

### Why Hyperion Works

On hyperion (existing database), the migration chain was applied **incrementally** as each migration was created. Migration `013_add_discovering_session_status` was the last one applied, and crucially, **the column was never actually tested with a value longer than 32 characters because**:

Wait -- that's not right. If hyperion is at head, it MUST have `013_add_discovering_session_status` stored in the version_num column. This means one of:

1. The `alembic_version.version_num` column on hyperion was **manually widened** at some point
2. PostgreSQL on hyperion has a different behavior for this column
3. The migration on hyperion was applied with a different version of the code

**Most likely explanation**: The `alembic_version` table on hyperion was manually altered to have a wider `version_num` column, either explicitly or as a side effect of some other operation. This would need to be verified by checking the actual column definition on hyperion:

```sql
SELECT column_name, character_maximum_length
FROM information_schema.columns
WHERE table_name = 'alembic_version' AND column_name = 'version_num';
```

**Alternative explanation**: If hyperion was running an older Alembic version that created the table differently, or if the table was manually created with a wider column.

---

## 5. Existing Fixes in Codebase

**None found.** No migration file contains any ALTER TABLE on `alembic_version`. No env.py customization addresses the column width. No configuration file sets a version_num_width.

---

## 6. Impact Assessment

### Current Impact
- **Fresh database deployments are BROKEN** -- cannot run full migration chain
- **Existing databases (hyperion) are UNAFFECTED** -- already past the problematic migration

### Future Impact
- Any new migration with a descriptive ID > 32 chars will also fail on fresh databases
- The Pattern 3 naming convention will continue to produce long IDs as descriptions grow

### Affected Migrations Count
- **Currently breaking**: 1 migration (`013_add_discovering_session_status`, 34 chars)
- **At risk**: `012_post_train_suggestions` (27 chars) -- safe now but fragile
- **Safe**: All other 34 migrations (3-21 chars)
