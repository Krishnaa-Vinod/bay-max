"""Simple IoU + cosine face tracker for frame-to-frame continuity.

This tracker assigns persistent track IDs to faces across frames using a combination
of bounding box IoU overlap and face embedding cosine similarity. It does NOT depend
on external tracking libraries like ByteTrack.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from uuid import UUID

import numpy as np

from baymax.schemas.perception import DetectedFace, FaceBoundingBox


@dataclass
class TrackedFace:
    """A face being tracked across frames."""

    track_id: str
    bbox: FaceBoundingBox
    embedding: list[float] | None = None
    user_id: UUID | None = None
    last_seen: float = field(default_factory=time.time)
    frames_seen: int = 1
    match_confidence: float = 0.0


def _compute_iou(a: FaceBoundingBox, b: FaceBoundingBox) -> float:
    """Compute intersection-over-union between two bounding boxes."""
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = a.width * a.height
    area_b = b.width * b.height
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _cosine_similarity(a: list[float] | None, b: list[float] | None) -> float:
    """Compute cosine similarity between two embedding vectors."""
    if a is None or b is None:
        return 0.0
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


class SimpleTracker:
    """Frame-to-frame face tracker using IoU + cosine similarity."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        cosine_threshold: float = 0.6,
        max_age_seconds: float = 5.0,
        iou_weight: float = 0.4,
        cosine_weight: float = 0.6,
    ) -> None:
        self._iou_threshold = iou_threshold
        self._cosine_threshold = cosine_threshold
        self._max_age = max_age_seconds
        self._iou_weight = iou_weight
        self._cosine_weight = cosine_weight
        self._tracks: dict[str, TrackedFace] = {}
        self._next_id = 0

    def _new_track_id(self) -> str:
        tid = f"face_{self._next_id}"
        self._next_id += 1
        return tid

    def update(self, detections: list[DetectedFace]) -> list[TrackedFace]:
        """Update tracks with new detections and return current tracks.

        Args:
            detections: Faces detected in the current frame.

        Returns:
            List of updated TrackedFace objects.
        """
        now = time.time()

        # Remove stale tracks
        stale = [
            tid for tid, t in self._tracks.items()
            if (now - t.last_seen) > self._max_age
        ]
        for tid in stale:
            del self._tracks[tid]

        if not detections:
            return list(self._tracks.values())

        active_tracks = list(self._tracks.values())
        matched_tracks: set[str] = set()
        matched_dets: set[int] = set()
        assignments: list[tuple[int, str, float]] = []

        # Compute combined scores between all detections and existing tracks
        for di, det in enumerate(detections):
            for track in active_tracks:
                iou = _compute_iou(det.bbox, track.bbox)
                cos = _cosine_similarity(det.embedding, track.embedding)
                combined = self._iou_weight * iou + self._cosine_weight * cos
                if iou >= self._iou_threshold or cos >= self._cosine_threshold:
                    assignments.append((di, track.track_id, combined))

        # Greedy assignment - best scores first
        assignments.sort(key=lambda x: x[2], reverse=True)
        for di, tid, score in assignments:
            if di in matched_dets or tid in matched_tracks:
                continue
            matched_dets.add(di)
            matched_tracks.add(tid)
            det = detections[di]
            track = self._tracks[tid]
            track.bbox = det.bbox
            track.embedding = det.embedding
            track.last_seen = now
            track.frames_seen += 1

        # Create new tracks for unmatched detections
        for di, det in enumerate(detections):
            if di not in matched_dets:
                tid = self._new_track_id()
                self._tracks[tid] = TrackedFace(
                    track_id=tid,
                    bbox=det.bbox,
                    embedding=det.embedding,
                    last_seen=now,
                )

        return list(self._tracks.values())

    @property
    def active_track_ids(self) -> list[str]:
        return list(self._tracks.keys())

    def get_track(self, track_id: str) -> TrackedFace | None:
        return self._tracks.get(track_id)

    def set_user_id(self, track_id: str, user_id: UUID, confidence: float) -> None:
        """Associate a track with a recognized user."""
        track = self._tracks.get(track_id)
        if track:
            track.user_id = user_id
            track.match_confidence = confidence

    def clear(self) -> None:
        """Clear all tracks."""
        self._tracks.clear()
        self._next_id = 0
