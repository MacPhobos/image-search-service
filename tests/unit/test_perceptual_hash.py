"""Tests for perceptual hash service."""

from pathlib import Path

import pytest
from PIL import Image

from image_search_service.services.perceptual_hash import (
    are_images_similar,
    compute_hash_hamming_distance,
    compute_perceptual_hash,
    compute_perceptual_hash_from_pil,
)


class TestPerceptualHashComputation:
    """Test perceptual hash computation."""

    def test_compute_hash_returns_16_char_hex(self, tmp_path: Path) -> None:
        """Test that hash is 16-character hex string."""
        # Create a simple test image
        img_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (100, 100), color="red")
        img.save(img_path)

        hash_value = compute_perceptual_hash(str(img_path))

        assert len(hash_value) == 16
        assert all(c in "0123456789abcdef" for c in hash_value)

    def test_identical_images_produce_identical_hashes(self, tmp_path: Path) -> None:
        """Test that identical images produce identical hashes."""
        # Create identical images
        img_path1 = tmp_path / "test1.jpg"
        img_path2 = tmp_path / "test2.jpg"

        img = Image.new("RGB", (100, 100), color="blue")
        img.save(img_path1)
        img.save(img_path2)

        hash1 = compute_perceptual_hash(str(img_path1))
        hash2 = compute_perceptual_hash(str(img_path2))

        assert hash1 == hash2

    def test_different_images_produce_different_hashes(self, tmp_path: Path) -> None:
        """Test that different images produce different hashes."""
        img_path1 = tmp_path / "gradient_horizontal.jpg"
        img_path2 = tmp_path / "gradient_vertical.jpg"

        # Create horizontal gradient (left=white, right=black)
        img1 = Image.new("RGB", (100, 100))
        for x in range(100):
            for y in range(100):
                gray = int(255 * (1 - x / 100))  # Gradient from white to black
                img1.putpixel((x, y), (gray, gray, gray))

        # Create vertical gradient (top=white, bottom=black)
        img2 = Image.new("RGB", (100, 100))
        for x in range(100):
            for y in range(100):
                gray = int(255 * (1 - y / 100))  # Gradient from white to black
                img2.putpixel((x, y), (gray, gray, gray))

        img1.save(img_path1)
        img2.save(img_path2)

        hash1 = compute_perceptual_hash(str(img_path1))
        hash2 = compute_perceptual_hash(str(img_path2))

        # Horizontal vs vertical gradients should produce different hashes
        assert hash1 != hash2

    def test_resized_images_produce_similar_hashes(self, tmp_path: Path) -> None:
        """Test that resized versions of same image produce similar hashes."""
        # Create original image
        original = Image.new("RGB", (200, 200))
        for x in range(200):
            for y in range(200):
                original.putpixel((x, y), (x, y, 128))

        # Save original and resized version
        img_path1 = tmp_path / "original.jpg"
        img_path2 = tmp_path / "resized.jpg"

        original.save(img_path1)
        resized = original.resize((100, 100))
        resized.save(img_path2)

        hash1 = compute_perceptual_hash(str(img_path1))
        hash2 = compute_perceptual_hash(str(img_path2))

        # Hashes should be similar (low Hamming distance)
        distance = compute_hash_hamming_distance(hash1, hash2)
        assert distance <= 10  # Allow some variation due to resize

    def test_compute_hash_from_pil_image(self) -> None:
        """Test computing hash from PIL Image object."""
        img = Image.new("RGB", (100, 100), color="green")

        hash_value = compute_perceptual_hash_from_pil(img)

        assert len(hash_value) == 16
        assert all(c in "0123456789abcdef" for c in hash_value)

    def test_file_not_found_raises_error(self, tmp_path: Path) -> None:
        """Test that missing file raises FileNotFoundError."""
        nonexistent = tmp_path / "nonexistent.jpg"

        with pytest.raises(FileNotFoundError):
            compute_perceptual_hash(str(nonexistent))

    def test_invalid_image_raises_error(self, tmp_path: Path) -> None:
        """Test that invalid image file raises ValueError."""
        # Create text file (not an image)
        bad_file = tmp_path / "not_an_image.jpg"
        bad_file.write_text("This is not an image")

        with pytest.raises(ValueError, match="Failed to process image"):
            compute_perceptual_hash(str(bad_file))


class TestHammingDistance:
    """Test Hamming distance computation."""

    def test_identical_hashes_have_zero_distance(self) -> None:
        """Test that identical hashes have Hamming distance of 0."""
        hash1 = "a1b2c3d4e5f6a7b8"
        hash2 = "a1b2c3d4e5f6a7b8"

        distance = compute_hash_hamming_distance(hash1, hash2)

        assert distance == 0

    def test_completely_different_hashes(self) -> None:
        """Test Hamming distance for completely different hashes."""
        hash1 = "0000000000000000"
        hash2 = "ffffffffffffffff"

        distance = compute_hash_hamming_distance(hash1, hash2)

        # All 64 bits differ
        assert distance == 64

    def test_single_bit_difference(self) -> None:
        """Test Hamming distance with single bit difference."""
        # Last hex char differs by 1 (8 -> 9 = 1000 -> 1001)
        hash1 = "a1b2c3d4e5f6a7b8"
        hash2 = "a1b2c3d4e5f6a7b9"

        distance = compute_hash_hamming_distance(hash1, hash2)

        assert distance == 1

    def test_invalid_hash_length_raises_error(self) -> None:
        """Test that invalid hash length raises ValueError."""
        hash1 = "a1b2c3d4"  # Too short
        hash2 = "a1b2c3d4e5f6a7b8"

        with pytest.raises(ValueError, match="must be 16 characters"):
            compute_hash_hamming_distance(hash1, hash2)

    def test_invalid_hex_raises_error(self) -> None:
        """Test that invalid hex string raises ValueError."""
        hash1 = "zzzzzzzzzzzzzzzz"  # Invalid hex
        hash2 = "a1b2c3d4e5f6a7b8"

        with pytest.raises(ValueError, match="Invalid hex hash"):
            compute_hash_hamming_distance(hash1, hash2)


class TestImageSimilarity:
    """Test image similarity comparison."""

    def test_identical_images_are_similar(self) -> None:
        """Test that identical hashes are considered similar."""
        hash1 = "a1b2c3d4e5f6a7b8"
        hash2 = "a1b2c3d4e5f6a7b8"

        assert are_images_similar(hash1, hash2, threshold=5)

    def test_very_different_images_not_similar(self) -> None:
        """Test that very different hashes are not similar."""
        hash1 = "0000000000000000"
        hash2 = "ffffffffffffffff"

        assert not are_images_similar(hash1, hash2, threshold=5)

    def test_threshold_boundary(self) -> None:
        """Test similarity at threshold boundary."""
        # Create hashes with exact threshold distance
        hash1 = "a1b2c3d4e5f6a7b8"
        # Change last byte to differ by exactly 5 bits:
        # b8 = 10111000 (binary)
        # a7 = 10100111 (binary)
        # XOR = 00011111 = 5 bits different (bits 0-4 flipped)
        hash2 = "a1b2c3d4e5f6a7a7"

        distance = compute_hash_hamming_distance(hash1, hash2)
        assert distance == 5

        # Should be similar at threshold=5
        assert are_images_similar(hash1, hash2, threshold=5)

        # Should not be similar at threshold=4
        assert not are_images_similar(hash1, hash2, threshold=4)

    def test_custom_threshold(self) -> None:
        """Test similarity with custom threshold."""
        hash1 = "a1b2c3d4e5f6a7b8"
        hash2 = "a1b2c3d4e5f6a7a7"  # 5 bits different (same as test_threshold_boundary)

        # Strict threshold (exact duplicates only)
        assert not are_images_similar(hash1, hash2, threshold=0)

        # Loose threshold (allow minor variations)
        assert are_images_similar(hash1, hash2, threshold=10)


class TestRealWorldScenarios:
    """Test real-world image scenarios."""

    def test_jpeg_compression_produces_similar_hash(self, tmp_path: Path) -> None:
        """Test that JPEG compression at different qualities produces similar hashes."""
        # Create original image
        img = Image.new("RGB", (200, 200))
        for x in range(200):
            for y in range(200):
                img.putpixel((x, y), (x, y, 128))

        # Save with different JPEG qualities
        high_quality = tmp_path / "high.jpg"
        low_quality = tmp_path / "low.jpg"

        img.save(high_quality, quality=95)
        img.save(low_quality, quality=50)

        hash_high = compute_perceptual_hash(str(high_quality))
        hash_low = compute_perceptual_hash(str(low_quality))

        # Should be similar despite compression differences
        distance = compute_hash_hamming_distance(hash_high, hash_low)
        assert distance <= 10

    def test_grayscale_and_color_versions(self, tmp_path: Path) -> None:
        """Test that color and grayscale versions produce different but related hashes."""
        # Create color image
        color_img = Image.new("RGB", (100, 100))
        for x in range(100):
            for y in range(100):
                color_img.putpixel((x, y), (x, y, x + y))

        # Convert to grayscale
        gray_img = color_img.convert("L").convert("RGB")

        color_path = tmp_path / "color.jpg"
        gray_path = tmp_path / "gray.jpg"

        color_img.save(color_path)
        gray_img.save(gray_path)

        hash_color = compute_perceptual_hash(str(color_path))
        hash_gray = compute_perceptual_hash(str(gray_path))

        # Hashes should be similar (structure is same, just color removed)
        distance = compute_hash_hamming_distance(hash_color, hash_gray)
        assert distance <= 15
