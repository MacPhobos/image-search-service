"""Face detection, alignment, and embedding module.

This module provides face detection and processing using InsightFace
with lazy loading and efficient batch processing.
"""

from image_search_service.faces.detector import DetectedFace, detect_faces, detect_faces_from_path
from image_search_service.faces.service import FaceProcessingService, get_face_service

__all__ = [
    "DetectedFace",
    "detect_faces",
    "detect_faces_from_path",
    "FaceProcessingService",
    "get_face_service",
]
