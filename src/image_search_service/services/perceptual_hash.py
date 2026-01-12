"""Perceptual hashing service for image deduplication.

Implements dHash (difference hash) algorithm for fast perceptual hashing.
dHash is robust to minor edits and scaling, making it ideal for detecting duplicates.

Algorithm:
1. Resize image to 9x8 grayscale (9 pixels wide to compute 8 differences)
2. Compute horizontal gradient (left pixel > right pixel = 1, else 0)
3. Convert 64-bit binary to 16-character hex string

References:
- http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html
- https://www.pyimagesearch.com/2017/11/27/image-hashing-opencv-python/
"""

from pathlib import Path

from PIL import Image

from image_search_service.core.logging import get_logger

logger = get_logger(__name__)


def compute_perceptual_hash(image_path: str | Path) -> str:
    """Compute perceptual hash (dHash) for an image file.

    Args:
        image_path: Path to image file

    Returns:
        16-character hexadecimal string representing 64-bit dHash

    Raises:
        ValueError: If image cannot be opened or processed
        FileNotFoundError: If image file does not exist
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        with Image.open(image_path) as img:
            return compute_perceptual_hash_from_pil(img)
    except Exception as e:
        logger.error(f"Failed to compute hash for {image_path}: {e}")
        raise ValueError(f"Failed to process image: {e}") from e


def compute_perceptual_hash_from_pil(image: Image.Image) -> str:
    """Compute perceptual hash (dHash) from PIL Image object.

    Args:
        image: PIL Image object (any format, any size)

    Returns:
        16-character hexadecimal string representing 64-bit dHash

    Implementation:
        1. Convert to grayscale
        2. Resize to 9x8 (9 wide for 8 horizontal differences)
        3. Compute gradient: for each row, compare adjacent pixels
        4. Convert to 64-bit integer (1 if left > right, 0 otherwise)
        5. Format as 16-char hex string
    """
    # Convert to grayscale and resize to 9x8
    # 9 pixels wide allows computing 8 differences per row
    gray = image.convert("L")
    resized = gray.resize((9, 8), Image.Resampling.LANCZOS)

    # Get pixel data as flat list
    pixels = list(resized.getdata())

    # Compute horizontal gradient (64 bits total)
    hash_bits = []
    for row in range(8):
        for col in range(8):
            # Compare adjacent pixels in row
            left_idx = row * 9 + col
            right_idx = left_idx + 1
            left_pixel = pixels[left_idx]
            right_pixel = pixels[right_idx]

            # Bit is 1 if left pixel is brighter than right
            hash_bits.append("1" if left_pixel > right_pixel else "0")

    # Convert 64-bit binary string to integer, then to hex
    hash_int = int("".join(hash_bits), 2)
    hash_hex = f"{hash_int:016x}"  # 16-char hex (64 bits / 4 bits per hex char)

    return hash_hex


def compute_hash_hamming_distance(hash1: str, hash2: str) -> int:
    """Compute Hamming distance between two perceptual hashes.

    Hamming distance is the number of bit positions where hashes differ.
    Lower distance = more similar images.

    Args:
        hash1: First hash (16-char hex string)
        hash2: Second hash (16-char hex string)

    Returns:
        Hamming distance (0-64, where 0 = identical, 64 = completely different)

    Raises:
        ValueError: If hashes are not valid 16-char hex strings

    Example:
        >>> h1 = "a1b2c3d4e5f6a7b8"
        >>> h2 = "a1b2c3d4e5f6a7b9"  # Only last bit differs
        >>> compute_hash_hamming_distance(h1, h2)
        1
    """
    if len(hash1) != 16 or len(hash2) != 16:
        raise ValueError(f"Hashes must be 16 characters (got {len(hash1)}, {len(hash2)})")

    try:
        # Convert hex to integers
        int1 = int(hash1, 16)
        int2 = int(hash2, 16)
    except ValueError as e:
        raise ValueError(f"Invalid hex hash: {e}") from e

    # XOR to find differing bits, then count 1s
    xor_result = int1 ^ int2
    return bin(xor_result).count("1")


def are_images_similar(hash1: str, hash2: str, threshold: int = 5) -> bool:
    """Check if two images are perceptually similar based on hash distance.

    Args:
        hash1: First perceptual hash
        hash2: Second perceptual hash
        threshold: Maximum Hamming distance to consider similar (default: 5)
                  Recommended values:
                  - 0: Exact duplicates only
                  - 1-5: Very similar (minor edits, compression)
                  - 6-10: Similar (cropping, color adjustments)
                  - 11+: Different images

    Returns:
        True if Hamming distance <= threshold
    """
    distance = compute_hash_hamming_distance(hash1, hash2)
    return distance <= threshold
