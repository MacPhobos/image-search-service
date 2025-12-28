"""Face detection using InsightFace/RetinaFace."""

import logging
from typing import Any, Optional

import numpy as np

from image_search_service.core.device import get_onnx_providers

logger = logging.getLogger(__name__)

# Lazy loading of insightface
_face_analysis: Any | None = None


def _has_gpu_provider() -> bool:
    """Check if GPU-accelerated ONNX provider is available."""
    providers = get_onnx_providers()
    gpu_providers = {"CUDAExecutionProvider", "CoreMLExecutionProvider"}
    return any(p in gpu_providers for p in providers)


def _ensure_model_loaded() -> Any:
    """Lazy load InsightFace model."""
    global _face_analysis
    if _face_analysis is None:
        try:
            from insightface.app import FaceAnalysis

            providers = get_onnx_providers()
            _face_analysis = FaceAnalysis(
                name="buffalo_l",  # Good balance of speed/accuracy
                providers=providers,
            )
            ctx_id = 0 if _has_gpu_provider() else -1
            _face_analysis.prepare(ctx_id=ctx_id, det_size=(640, 640))
            provider_name = providers[0] if providers else "CPU"
            logger.info(f"Loaded InsightFace model (buffalo_l) with {provider_name}")
        except ImportError as e:
            logger.error("InsightFace not installed. Run: pip install insightface onnxruntime-gpu")
            raise ImportError(
                "InsightFace not installed. Install with: pip install insightface onnxruntime-gpu"
            ) from e
    return _face_analysis


class DetectedFace:
    """Represents a detected face with all metadata."""

    def __init__(
        self,
        bbox: tuple[int, int, int, int],  # (x, y, w, h)
        confidence: float,
        landmarks: "np.ndarray[Any, Any]",  # 5-point landmarks
        embedding: "np.ndarray[Any, Any]",  # 512-d normalized embedding
        aligned_face: Optional["np.ndarray[Any, Any]"] = None,  # Aligned and cropped face image
    ):
        self.bbox = bbox
        self.confidence = confidence
        self.landmarks = landmarks
        self.embedding = embedding
        self.aligned_face = aligned_face

    def landmarks_as_dict(self) -> dict[str, list[float]]:
        """Convert landmarks to JSON-serializable dict."""
        # Return 5 points: left_eye, right_eye, nose, mouth_left, mouth_right
        return {
            "left_eye": self.landmarks[0].tolist(),
            "right_eye": self.landmarks[1].tolist(),
            "nose": self.landmarks[2].tolist(),
            "mouth_left": self.landmarks[3].tolist(),
            "mouth_right": self.landmarks[4].tolist(),
        }

    def compute_quality_score(self) -> float:
        """Compute face quality score (0-1) based on bbox size and confidence."""
        # Simple heuristic: larger faces + higher confidence = better quality
        area = self.bbox[2] * self.bbox[3]  # w * h
        # Normalize area (assume 100x100 = 0.5, 200x200 = 1.0)
        area_score = min(1.0, area / 40000)
        # Combine with confidence
        return (area_score * 0.5) + (self.confidence * 0.5)


def detect_faces(
    image: "np.ndarray[Any, Any]",
    min_confidence: float = 0.5,
    min_face_size: int = 20,
) -> list[DetectedFace]:
    """Detect faces in an image and return embeddings.

    Args:
        image: BGR image as numpy array (OpenCV format)
        min_confidence: Minimum detection confidence threshold
        min_face_size: Minimum face width/height in pixels

    Returns:
        List of DetectedFace objects with bbox, landmarks, and embeddings
    """
    app = _ensure_model_loaded()

    # InsightFace expects BGR format, which is what OpenCV provides
    faces = app.get(image)

    results = []
    for face in faces:
        # Get bounding box (InsightFace gives x1, y1, x2, y2)
        x1, y1, x2, y2 = face.bbox.astype(int)
        w, h = x2 - x1, y2 - y1

        # Filter by confidence and size
        if face.det_score < min_confidence:
            continue
        if w < min_face_size or h < min_face_size:
            continue

        # Get 5-point landmarks
        landmarks = face.kps  # Shape: (5, 2)

        # Get embedding (already normalized by InsightFace)
        embedding = face.embedding

        # InsightFace provides aligned face in normed_embedding attribute
        # but we don't need it for storage, so we pass None
        aligned = None

        results.append(
            DetectedFace(
                bbox=(int(x1), int(y1), int(w), int(h)),
                confidence=float(face.det_score),
                landmarks=landmarks,
                embedding=embedding,
                aligned_face=aligned,
            )
        )

    logger.debug(f"Detected {len(results)} faces in image")
    return results


def detect_faces_from_path(
    image_path: str,
    min_confidence: float = 0.5,
    min_face_size: int = 20,
) -> list[DetectedFace]:
    """Detect faces from an image file path.

    Applies EXIF orientation to ensure correct face detection on rotated images.
    """
    import cv2
    from PIL import Image, ImageOps

    # Read image with PIL first to apply EXIF orientation
    image: "np.ndarray[Any, Any]" | None = None
    try:
        with Image.open(image_path) as pil_img:
            # Apply EXIF orientation transformation
            pil_img = ImageOps.exif_transpose(pil_img) or pil_img
            # Convert PIL RGB to OpenCV BGR format
            image_rgb = np.array(pil_img.convert("RGB"))
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.warning(f"Could not read image with PIL/EXIF: {image_path}, error: {e}")
        # Fallback to cv2.imread without EXIF handling
        image = cv2.imread(image_path)

    if image is None:
        logger.warning(f"Could not read image: {image_path}")
        return []

    return detect_faces(image, min_confidence, min_face_size)
