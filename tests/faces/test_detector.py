"""Tests for face detection module."""

from unittest.mock import MagicMock

import numpy as np


class TestDetectedFace:
    """Tests for DetectedFace class."""

    def test_landmarks_as_dict(self, mock_detected_face):
        """Test landmark conversion to dict."""
        landmarks = mock_detected_face.landmarks_as_dict()

        assert "left_eye" in landmarks
        assert "right_eye" in landmarks
        assert "nose" in landmarks
        assert "mouth_left" in landmarks
        assert "mouth_right" in landmarks

        # Check all are lists with 2 elements
        for key, value in landmarks.items():
            assert isinstance(value, list)
            assert len(value) == 2

    def test_compute_quality_score(self, mock_detected_face):
        """Test quality score computation."""
        score = mock_detected_face.compute_quality_score()

        assert 0.0 <= score <= 1.0
        # Higher confidence + larger bbox = higher quality
        assert score > 0.5  # Our mock has good confidence

    def test_quality_score_small_face(self):
        """Test quality score for small faces is lower."""
        from image_search_service.faces.detector import DetectedFace

        small_face = DetectedFace(
            bbox=(0, 0, 20, 20),  # Small face
            confidence=0.5,
            landmarks=np.zeros((5, 2)),
            embedding=np.zeros(512),
            aligned_face=None,
        )

        large_face = DetectedFace(
            bbox=(0, 0, 200, 200),  # Large face
            confidence=0.5,
            landmarks=np.zeros((5, 2)),
            embedding=np.zeros(512),
            aligned_face=None,
        )

        assert small_face.compute_quality_score() < large_face.compute_quality_score()

    def test_quality_score_high_confidence(self):
        """Test quality score increases with confidence."""
        from image_search_service.faces.detector import DetectedFace

        low_conf = DetectedFace(
            bbox=(0, 0, 100, 100),
            confidence=0.3,
            landmarks=np.zeros((5, 2)),
            embedding=np.zeros(512),
            aligned_face=None,
        )

        high_conf = DetectedFace(
            bbox=(0, 0, 100, 100),
            confidence=0.95,
            landmarks=np.zeros((5, 2)),
            embedding=np.zeros(512),
            aligned_face=None,
        )

        assert low_conf.compute_quality_score() < high_conf.compute_quality_score()


class TestDetectFaces:
    """Tests for detect_faces function."""

    def test_detect_faces_returns_list(self, mock_insightface):
        """Test that detect_faces returns a list."""
        from image_search_service.faces.detector import detect_faces

        # Mock InsightFace response
        mock_face = MagicMock()
        mock_face.bbox = np.array([100, 150, 180, 230])  # x1, y1, x2, y2
        mock_face.det_score = 0.95
        mock_face.kps = np.zeros((5, 2))
        mock_face.embedding = np.zeros(512)

        mock_insightface.get.return_value = [mock_face]

        image = np.zeros((640, 480, 3), dtype=np.uint8)
        faces = detect_faces(image)

        assert isinstance(faces, list)
        assert len(faces) == 1

    def test_detect_faces_filters_low_confidence(self, mock_insightface):
        """Test that low confidence faces are filtered."""
        from image_search_service.faces.detector import detect_faces

        # Mock faces with varying confidence
        high_conf = MagicMock()
        high_conf.bbox = np.array([100, 100, 200, 200])
        high_conf.det_score = 0.95
        high_conf.kps = np.zeros((5, 2))
        high_conf.embedding = np.zeros(512)

        low_conf = MagicMock()
        low_conf.bbox = np.array([300, 300, 400, 400])
        low_conf.det_score = 0.3
        low_conf.kps = np.zeros((5, 2))
        low_conf.embedding = np.zeros(512)

        mock_insightface.get.return_value = [high_conf, low_conf]

        image = np.zeros((640, 480, 3), dtype=np.uint8)
        faces = detect_faces(image, min_confidence=0.5)

        assert len(faces) == 1
        assert faces[0].confidence == 0.95

    def test_detect_faces_filters_small_faces(self, mock_insightface):
        """Test that small faces are filtered."""
        from image_search_service.faces.detector import detect_faces

        small_face = MagicMock()
        small_face.bbox = np.array([100, 100, 115, 115])  # 15x15 pixels
        small_face.det_score = 0.95
        small_face.kps = np.zeros((5, 2))
        small_face.embedding = np.zeros(512)

        mock_insightface.get.return_value = [small_face]

        image = np.zeros((640, 480, 3), dtype=np.uint8)
        faces = detect_faces(image, min_face_size=20)

        assert len(faces) == 0

    def test_detect_faces_no_faces(self, mock_insightface):
        """Test detection when no faces are found."""
        from image_search_service.faces.detector import detect_faces

        mock_insightface.get.return_value = []

        image = np.zeros((640, 480, 3), dtype=np.uint8)
        faces = detect_faces(image)

        assert len(faces) == 0

    def test_detect_faces_multiple_faces(self, mock_insightface):
        """Test detection of multiple faces."""
        from image_search_service.faces.detector import detect_faces

        # Create 3 mock faces
        faces_data = []
        for i in range(3):
            mock_face = MagicMock()
            mock_face.bbox = np.array([100 * i, 100, 100 * i + 80, 180])
            mock_face.det_score = 0.9
            mock_face.kps = np.zeros((5, 2))
            mock_face.embedding = np.zeros(512)
            faces_data.append(mock_face)

        mock_insightface.get.return_value = faces_data

        image = np.zeros((640, 480, 3), dtype=np.uint8)
        faces = detect_faces(image)

        assert len(faces) == 3

    def test_detect_faces_bbox_conversion(self, mock_insightface):
        """Test that bbox is correctly converted from x1,y1,x2,y2 to x,y,w,h."""
        from image_search_service.faces.detector import detect_faces

        mock_face = MagicMock()
        # x1=100, y1=150, x2=180, y2=230
        # Expected: x=100, y=150, w=80, h=80
        mock_face.bbox = np.array([100, 150, 180, 230])
        mock_face.det_score = 0.95
        mock_face.kps = np.zeros((5, 2))
        mock_face.embedding = np.zeros(512)

        mock_insightface.get.return_value = [mock_face]

        image = np.zeros((640, 480, 3), dtype=np.uint8)
        faces = detect_faces(image)

        assert len(faces) == 1
        assert faces[0].bbox == (100, 150, 80, 80)


class TestDetectFacesFromPath:
    """Tests for detect_faces_from_path function."""

    def test_detect_faces_from_path_invalid_image(self, mock_insightface):
        """Test detection from invalid image path."""
        from image_search_service.faces.detector import detect_faces_from_path

        faces = detect_faces_from_path("/nonexistent/path.jpg")

        assert len(faces) == 0

    def test_detect_faces_from_path_valid_image(self, mock_insightface, tmp_path):
        """Test detection from valid image path."""
        from PIL import Image

        from image_search_service.faces.detector import detect_faces_from_path

        # Create a test image
        image_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (640, 480), color="white")
        img.save(image_path)

        # Mock face detection
        mock_face = MagicMock()
        mock_face.bbox = np.array([100, 150, 180, 230])
        mock_face.det_score = 0.95
        mock_face.kps = np.zeros((5, 2))
        mock_face.embedding = np.zeros(512)

        mock_insightface.get.return_value = [mock_face]

        faces = detect_faces_from_path(str(image_path))

        assert len(faces) == 1
        assert faces[0].confidence == 0.95
