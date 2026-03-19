"""FaceNet-PyTorch adapter for face detection and recognition.

Uses MTCNN for detection/alignment and InceptionResnetV1 (vggface2) for embeddings.
"""

from __future__ import annotations

import logging
from uuid import UUID

import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

from baymax.perception.interfaces import FaceDetector, FaceRecognizer
from baymax.schemas.perception import DetectedFace, FaceBoundingBox, FaceEmbeddingRecord

logger = logging.getLogger(__name__)


class FacenetDetector(FaceDetector):
    """MTCNN-based face detector with InceptionResnetV1 embeddings."""

    def __init__(
        self,
        device: str = "cpu",
        min_face_size: int = 40,
        thresholds: tuple[float, float, float] = (0.6, 0.7, 0.7),
        max_faces: int = 5,
    ) -> None:
        self._device = torch.device(device)
        self._min_face_size = min_face_size
        self._thresholds = list(thresholds)
        self._max_faces = max_faces
        self._mtcnn: MTCNN | None = None
        self._resnet: InceptionResnetV1 | None = None

    def load_model(self) -> None:
        """Initialize MTCNN detector and InceptionResnetV1 embedding model."""
        logger.info("Loading MTCNN on %s (min_face_size=%d)", self._device, self._min_face_size)
        self._mtcnn = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=self._min_face_size,
            thresholds=self._thresholds,
            keep_all=True,
            device=self._device,
        )
        logger.info("Loading InceptionResnetV1 (vggface2) on %s", self._device)
        self._resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self._device)
        logger.info("FacenetDetector models loaded successfully")

    def detect(self, frame: np.ndarray) -> list[DetectedFace]:
        """Detect faces and compute embeddings from an RGB numpy frame.

        Args:
            frame: H x W x 3 numpy array (BGR or RGB).

        Returns:
            List of DetectedFace with bounding boxes and 512-d embeddings.
        """
        if self._mtcnn is None or self._resnet is None:
            self.load_model()
        assert self._mtcnn is not None and self._resnet is not None

        # Runtime frame sources already normalize frames to RGB.
        rgb = frame

        pil_image = Image.fromarray(rgb)

        # Detect faces - returns boxes and probabilities
        boxes, probs = self._mtcnn.detect(pil_image)

        if boxes is None or len(boxes) == 0:
            return []

        # Limit to max_faces
        n = min(len(boxes), self._max_faces)
        boxes = boxes[:n]
        probs = probs[:n]

        # Extract aligned face tensors for embedding
        faces_cropped = self._mtcnn(pil_image)

        results: list[DetectedFace] = []
        if faces_cropped is None:
            return []

        # faces_cropped might be a single tensor if only one face
        if faces_cropped.dim() == 3:
            faces_cropped = faces_cropped.unsqueeze(0)

        for i in range(min(n, faces_cropped.shape[0])):
            box = boxes[i]
            prob = float(probs[i]) if probs[i] is not None else 0.0

            # Compute embedding
            face_tensor = faces_cropped[i].unsqueeze(0).to(self._device)
            with torch.no_grad():
                embedding = self._resnet(face_tensor).cpu().numpy().flatten().tolist()

            results.append(DetectedFace(
                bbox=FaceBoundingBox(
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                ),
                detection_confidence=prob,
                embedding=embedding,
                backend="facenet_pytorch",
            ))

        return results


class CosineRecognizer(FaceRecognizer):
    """Recognize faces by cosine similarity against enrolled embeddings."""

    def best_match(
        self,
        embedding: list[float],
        enrolled: list[FaceEmbeddingRecord],
    ) -> tuple[UUID | None, float]:
        """Return best user candidate and cosine score, even below threshold."""
        if not enrolled or not embedding:
            return None, 0.0

        query = np.array(embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return None, 0.0
        query = query / query_norm

        best_user_id: UUID | None = None
        best_score: float = 0.0

        for record in enrolled:
            ref = np.array(record.embedding, dtype=np.float32)
            ref_norm = np.linalg.norm(ref)
            if ref_norm == 0:
                continue
            ref = ref / ref_norm
            score = float(np.dot(query, ref))
            if score > best_score:
                best_score = score
                best_user_id = record.user_id

        return best_user_id, best_score

    def recognize(
        self,
        embedding: list[float],
        enrolled: list[FaceEmbeddingRecord],
        threshold: float = 0.75,
    ) -> tuple[UUID | None, float]:
        """Match embedding against enrolled users.

        Args:
            embedding: 512-d face embedding vector.
            enrolled: List of enrolled FaceEmbeddingRecord.
            threshold: Minimum cosine similarity for a match.

        Returns:
            (user_id, confidence) if matched, (None, best_score) otherwise.
        """
        best_user_id, best_score = self.best_match(embedding, enrolled)

        if best_score >= threshold:
            return best_user_id, best_score
        return None, best_score
