"""
Shared test constants for magic numbers used across the test suite.

These constants document the expected dimensions, lengths, and sizes
used throughout the image search service tests.  They mirror the
production configuration so that a single change propagates everywhere
instead of requiring grep-and-replace across dozens of files.
"""

# ---------------------------------------------------------------------------
# Embedding dimensions
# ---------------------------------------------------------------------------

#: CLIP / SigLIP image-search embedding dimension (production default).
CLIP_EMBEDDING_DIM: int = 768

#: Alias – SigLIP uses the same 768-dim output as CLIP in this project.
SIGLIP_EMBEDDING_DIM: int = 768

#: InsightFace / ArcFace face-recognition embedding dimension.
FACE_EMBEDDING_DIM: int = 512

# ---------------------------------------------------------------------------
# Hash lengths
# ---------------------------------------------------------------------------

#: Length of a SHA-256 hex digest string (64 hex characters).
SHA256_HEX_LENGTH: int = 64

#: Length of the perceptual hash hex string (64-bit dHash → 16 hex chars).
PERCEPTUAL_HASH_LENGTH: int = 16
