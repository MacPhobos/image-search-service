# SigLIP Quick Start Guide

**Status:** Phase 2 Complete (Infrastructure Ready)
**Next:** Phase 3 Re-embedding Pipeline

---

## Overview

Phase 2 implements SigLIP model infrastructure with feature flags for gradual rollout. This guide shows how to enable and test SigLIP embeddings.

---

## Step 1: Create SigLIP Collection

Create the parallel Qdrant collection for SigLIP embeddings:

```bash
cd /path/to/image-search-service

# Preview (dry run)
uv run python scripts/create_siglip_collection.py --dry-run

# Create collection
uv run python scripts/create_siglip_collection.py

# Verify creation
uv run python scripts/create_siglip_collection.py --check
```

**Expected Output:**
```
Creating SigLIP collection...
  Collection name: image_assets_siglip
  Embedding dimension: 768
  Qdrant URL: http://localhost:6333

✓ Collection 'image_assets_siglip' created successfully

Collection details:
  Status: green
  Points: 0
  Vector size: 768
  Distance: COSINE
  Quantization enabled: True
  HNSW m: 16
  HNSW ef_construct: 100
```

---

## Step 2: Enable SigLIP (Choose One)

### Option A: Full SigLIP Mode (100% Traffic)

All searches use SigLIP immediately:

```bash
# Set environment variable
export USE_SIGLIP=true

# Restart API
docker-compose restart api
```

### Option B: Gradual Rollout (Percentage-Based)

Route a percentage of searches to SigLIP for A/B testing:

```bash
# 10% of searches use SigLIP
export SIGLIP_ROLLOUT_PERCENTAGE=10

# Restart API
docker-compose restart api
```

**Bucketing:**
- Deterministic by `user_id` (same user always gets same model)
- Random if no `user_id` (anonymous searches)

**Examples:**
```
user_id=23 → bucket=23 → 23 < 10 → SigLIP ✓
user_id=87 → bucket=87 → 87 >= 10 → CLIP ✗
user_id=5  → bucket=5  → 5 < 10 → SigLIP ✓
```

---

## Step 3: Verify Configuration

Check which model is active:

```bash
# Check environment variables
env | grep -E "(USE_SIGLIP|SIGLIP_ROLLOUT)"

# Test search endpoint (should work with both CLIP and SigLIP)
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "sunset over mountains", "limit": 5}'
```

---

## Current Limitations (Phase 2)

⚠️ **Phase 2 is infrastructure-only. To use SigLIP in production, you must:**

1. **Re-embed all images** with SigLIP (Phase 3)
   - Create RQ job for batch re-embedding
   - Enqueue all existing assets
   - Monitor progress

2. **Verify collection populated**
   ```bash
   # Check SigLIP collection has vectors
   uv run python scripts/create_siglip_collection.py --check
   ```

**Current Behavior:**
- Enabling SigLIP without re-embedding → **empty search results**
- SigLIP collection starts with 0 vectors
- Phase 3 will populate the collection

---

## Rollback

### Immediate Rollback (Minutes)

Disable SigLIP and revert to CLIP:

```bash
# Method 1: Disable full SigLIP mode
export USE_SIGLIP=false

# Method 2: Disable gradual rollout
export SIGLIP_ROLLOUT_PERCENTAGE=0

# Restart API
docker-compose restart api
```

All searches immediately use CLIP (legacy behavior).

---

## Testing

### Test Router Logic

```python
from image_search_service.services.embedding_router import get_search_embedding_service

# Test default (should be CLIP)
service, collection = get_search_embedding_service(user_id=None)
print(f"Service: {service.__class__.__name__}")  # EmbeddingService
print(f"Collection: {collection}")  # image_assets

# Test with USE_SIGLIP=true
# (Set env var first)
service, collection = get_search_embedding_service(user_id=None)
print(f"Service: {service.__class__.__name__}")  # SigLIPEmbeddingService
print(f"Collection: {collection}")  # image_assets_siglip
```

### Test Embedding Generation

```python
from image_search_service.services.siglip_embedding import get_siglip_service

service = get_siglip_service()

# Text embedding
text_vector = service.embed_text("a beautiful sunset")
print(f"Text vector dim: {len(text_vector)}")  # 768

# Image embedding
image_vector = service.embed_image("/path/to/image.jpg")
print(f"Image vector dim: {len(image_vector)}")  # 768
```

---

## Next Steps

### Phase 3: Re-embedding Pipeline

**Required for production use:**

1. Create RQ job for batch re-embedding:
   - `jobs/reembed_siglip.py`
   - Batch size: 100 images
   - GPU-based processing

2. Enqueue all assets:
   ```python
   from jobs.reembed_siglip import enqueue_full_reembedding
   enqueue_full_reembedding(batch_size=100)
   ```

3. Monitor progress:
   - Track via admin endpoint `/api/v1/admin/siglip/status`
   - Check Qdrant collection count

4. Verify completeness:
   - SigLIP collection count == CLIP collection count

**Timeline:** 2-3 weeks (depends on dataset size)

---

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SIGLIP_MODEL_NAME` | `ViT-B-16-SigLIP` | Model architecture |
| `SIGLIP_PRETRAINED` | `webli` | Pretrained weights |
| `SIGLIP_EMBEDDING_DIM` | `768` | Vector dimension |
| `SIGLIP_COLLECTION` | `image_assets_siglip` | Qdrant collection |
| `USE_SIGLIP` | `false` | Full SigLIP mode |
| `SIGLIP_ROLLOUT_PERCENTAGE` | `0` | Gradual rollout (0-100) |

### Collection Settings

| Setting | Value | Notes |
|---------|-------|-------|
| Vector size | 768 | vs CLIP's 512 |
| Distance metric | COSINE | Same as CLIP |
| Quantization | INT8 | 75% memory reduction |
| HNSW m | 16 | Neighbor count |
| HNSW ef_construct | 100 | Construction depth |

---

## Troubleshooting

### Empty Search Results

**Symptom:** SigLIP enabled but searches return empty results

**Cause:** SigLIP collection not populated yet

**Solution:**
```bash
# Check collection status
uv run python scripts/create_siglip_collection.py --check

# If points=0, you need to re-embed (Phase 3)
# Temporarily disable SigLIP:
export USE_SIGLIP=false
docker-compose restart api
```

### Model Loading Errors

**Symptom:** `RuntimeError: Model not found`

**Cause:** SigLIP model not downloaded

**Solution:**
```bash
# Manually download model (if needed)
python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-16-SigLIP', pretrained='webli')"
```

### GPU Memory Issues

**Symptom:** CUDA out of memory

**Solution:**
```bash
# Reduce batch size (in config)
export GPU_BATCH_SIZE=8

# Or use CPU (slower)
export DEVICE=cpu
```

---

## Resources

- **Implementation Plan:** `/docs/plans/semantic-search-phase2-model-upgrade.md`
- **Implementation Summary:** `/PHASE2_IMPLEMENTATION_SUMMARY.md`
- **Tests:** `tests/unit/test_siglip_embedding_service.py`
- **Tests:** `tests/unit/test_embedding_router.py`

---

*Last Updated: 2026-01-24*
*Phase 2 Complete - Infrastructure Ready*
