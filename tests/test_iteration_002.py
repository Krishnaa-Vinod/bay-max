"""Tests for Iteration 002: perception, tracker, enrollment, frame analysis."""

from uuid import uuid4

import numpy as np
import pytest

from baymax.memory.store_sqlite import SQLiteMetadataStore
from baymax.perception.facenet_adapter import CosineRecognizer
from baymax.perception.interfaces import StubFaceDetector, StubFaceRecognizer
from baymax.perception.tracker import SimpleTracker, _compute_iou, _cosine_similarity
from baymax.schemas.perception import (
    DetectedFace,
    FaceBoundingBox,
    FaceEmbeddingRecord,
    FrameAnalysisResult,
    RecognitionObservation,
    RecognizedFace,
    UnknownFace,
)
from baymax.schemas.user import FaceEnrollment, UserProfile
from baymax.state.manager import StateManager
from baymax.state.models import InteractionState

# --- Schema tests ---


class TestPerceptionSchemas:
    def test_face_bounding_box(self):
        bbox = FaceBoundingBox(x1=10, y1=20, x2=110, y2=220)
        assert bbox.width == 100
        assert bbox.height == 200

    def test_detected_face(self):
        face = DetectedFace(
            bbox=FaceBoundingBox(x1=0, y1=0, x2=100, y2=100),
            detection_confidence=0.99,
            embedding=[0.1] * 512,
        )
        assert face.detection_confidence == 0.99
        assert len(face.embedding) == 512
        assert face.backend == "facenet_pytorch"

    def test_recognized_face(self):
        uid = uuid4()
        face = RecognizedFace(
            user_id=uid,
            user_display_name="Alice",
            bbox=FaceBoundingBox(x1=0, y1=0, x2=50, y2=50),
            detection_confidence=0.95,
            match_confidence=0.88,
        )
        assert face.user_id == uid
        assert face.match_confidence == 0.88

    def test_unknown_face(self):
        face = UnknownFace(
            bbox=FaceBoundingBox(x1=0, y1=0, x2=50, y2=50),
            detection_confidence=0.9,
            best_match_score=0.42,
        )
        assert face.best_match_score == 0.42

    def test_face_embedding_record(self):
        uid = uuid4()
        rec = FaceEmbeddingRecord(
            user_id=uid,
            embedding=[0.5] * 512,
        )
        assert rec.user_id == uid
        assert rec.model_name == "InceptionResnetV1-vggface2"

    def test_frame_analysis_result(self):
        sid = uuid4()
        result = FrameAnalysisResult(session_id=sid, faces_detected=2)
        assert result.faces_detected == 2
        assert result.observations_written == 0

    def test_recognition_observation(self):
        sid = uuid4()
        obs = RecognitionObservation(
            session_id=sid,
            event_type="first_recognition",
            content="Recognized user for the first time.",
        )
        assert obs.event_type == "first_recognition"

    def test_face_enrollment_new_fields(self):
        uid = uuid4()
        enrollment = FaceEnrollment(
            user_id=uid,
            status="enrolled",
            embedding_model="InceptionResnetV1-vggface2",
            face_count_detected=1,
            message="Face enrolled successfully.",
        )
        assert enrollment.embedding_model == "InceptionResnetV1-vggface2"
        assert enrollment.face_count_detected == 1


# --- Tracker tests ---


class TestTracker:
    def test_iou_identical_boxes(self):
        a = FaceBoundingBox(x1=0, y1=0, x2=100, y2=100)
        assert _compute_iou(a, a) == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        a = FaceBoundingBox(x1=0, y1=0, x2=10, y2=10)
        b = FaceBoundingBox(x1=20, y1=20, x2=30, y2=30)
        assert _compute_iou(a, b) == 0.0

    def test_iou_partial_overlap(self):
        a = FaceBoundingBox(x1=0, y1=0, x2=100, y2=100)
        b = FaceBoundingBox(x1=50, y1=50, x2=150, y2=150)
        iou = _compute_iou(a, b)
        assert 0.0 < iou < 1.0

    def test_cosine_similarity_identical(self):
        v = [1.0, 0.0, 0.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_cosine_similarity_none(self):
        assert _cosine_similarity(None, [1.0]) == 0.0

    def test_tracker_creates_tracks(self):
        tracker = SimpleTracker()
        face = DetectedFace(
            bbox=FaceBoundingBox(x1=10, y1=10, x2=60, y2=60),
            detection_confidence=0.9,
            embedding=[0.1] * 128,
        )
        tracks = tracker.update([face])
        assert len(tracks) == 1
        assert tracks[0].track_id == "face_0"

    def test_tracker_maintains_tracks(self):
        tracker = SimpleTracker()
        face1 = DetectedFace(
            bbox=FaceBoundingBox(x1=10, y1=10, x2=60, y2=60),
            detection_confidence=0.9,
            embedding=[0.1] * 128,
        )
        tracker.update([face1])

        # Same position and embedding -> should maintain track
        face2 = DetectedFace(
            bbox=FaceBoundingBox(x1=12, y1=12, x2=62, y2=62),
            detection_confidence=0.85,
            embedding=[0.1] * 128,
        )
        tracks = tracker.update([face2])
        assert len(tracks) == 1
        assert tracks[0].frames_seen == 2

    def test_tracker_creates_new_track_for_different_face(self):
        tracker = SimpleTracker()
        face1 = DetectedFace(
            bbox=FaceBoundingBox(x1=10, y1=10, x2=60, y2=60),
            detection_confidence=0.9,
            embedding=[0.1] * 128,
        )
        tracker.update([face1])

        # Very different position and embedding
        face2 = DetectedFace(
            bbox=FaceBoundingBox(x1=300, y1=300, x2=400, y2=400),
            detection_confidence=0.9,
            embedding=[-0.1] * 128,
        )
        tracks = tracker.update([face1, face2])
        assert len(tracks) == 2

    def test_tracker_set_user_id(self):
        tracker = SimpleTracker()
        face = DetectedFace(
            bbox=FaceBoundingBox(x1=10, y1=10, x2=60, y2=60),
            detection_confidence=0.9,
            embedding=[0.1] * 128,
        )
        tracker.update([face])
        uid = uuid4()
        tracker.set_user_id("face_0", uid, 0.92)
        track = tracker.get_track("face_0")
        assert track is not None
        assert track.user_id == uid
        assert track.match_confidence == 0.92

    def test_tracker_clear(self):
        tracker = SimpleTracker()
        face = DetectedFace(
            bbox=FaceBoundingBox(x1=10, y1=10, x2=60, y2=60),
            detection_confidence=0.9,
        )
        tracker.update([face])
        tracker.clear()
        assert tracker.active_track_ids == []


# --- CosineRecognizer tests ---


class TestCosineRecognizer:
    def test_no_enrolled(self):
        rec = CosineRecognizer()
        uid, score = rec.recognize([0.1] * 512, [], threshold=0.75)
        assert uid is None
        assert score == 0.0

    def test_match_above_threshold(self):
        rec = CosineRecognizer()
        uid = uuid4()
        enrolled = [FaceEmbeddingRecord(
            user_id=uid,
            embedding=[0.1] * 512,
        )]
        matched_uid, score = rec.recognize([0.1] * 512, enrolled, threshold=0.75)
        assert matched_uid == uid
        assert score == pytest.approx(1.0)

    def test_no_match_below_threshold(self):
        rec = CosineRecognizer()
        uid = uuid4()
        # Create orthogonal embeddings
        emb_enrolled = [1.0] + [0.0] * 511
        emb_query = [0.0, 1.0] + [0.0] * 510
        enrolled = [FaceEmbeddingRecord(
            user_id=uid,
            embedding=emb_enrolled,
        )]
        matched_uid, score = rec.recognize(emb_query, enrolled, threshold=0.75)
        assert matched_uid is None

    def test_best_match_among_multiple(self):
        rec = CosineRecognizer()
        uid_a = uuid4()
        uid_b = uuid4()
        enrolled = [
            FaceEmbeddingRecord(user_id=uid_a, embedding=[1.0] + [0.0] * 511),
            FaceEmbeddingRecord(user_id=uid_b, embedding=[0.0, 1.0] + [0.0] * 510),
        ]
        matched_uid, score = rec.recognize([1.0] + [0.0] * 511, enrolled, threshold=0.5)
        assert matched_uid == uid_a
        assert score == pytest.approx(1.0)


# --- State Manager tests ---


class TestStateManagerRecognition:
    def test_update_recognition(self):
        mgr = StateManager()
        sid = uuid4()
        uid = uuid4()
        mgr.get_or_create(sid)
        state = mgr.update_recognition(
            sid,
            user_id=uid,
            user_display_name="Alice",
            identity_confidence=0.92,
            face_count=1,
            active_track_ids=["face_0"],
            frame_summary="Recognized: Alice",
        )
        assert state is not None
        assert state.user_id == uid
        assert state.identity_confidence == 0.92
        assert state.face_count == 1
        assert state.last_seen_at is not None
        assert state.active_track_ids == ["face_0"]
        assert state.last_frame_summary == "Recognized: Alice"
        assert state.is_new_user is False

    def test_interaction_state_defaults(self):
        sid = uuid4()
        state = InteractionState(session_id=sid)
        assert state.identity_confidence == 0.0
        assert state.face_count == 0
        assert state.last_seen_at is None
        assert state.active_track_ids == []
        assert state.last_frame_summary == ""


# --- SQLite store tests ---


class TestSQLiteStoreV2:
    @pytest.fixture(autouse=True)
    async def setup_store(self, tmp_path):
        self.store = SQLiteMetadataStore(db_path=str(tmp_path / "test.db"))
        await self.store.initialize()
        yield
        await self.store.close()

    @pytest.mark.asyncio
    async def test_list_users_empty(self):
        users = await self.store.list_users()
        assert users == []

    @pytest.mark.asyncio
    async def test_list_users_returns_created(self):
        user = UserProfile(display_name="Alice")
        await self.store.create_user(user)
        users = await self.store.list_users()
        assert len(users) == 1
        assert users[0].display_name == "Alice"

    @pytest.mark.asyncio
    async def test_store_and_retrieve_face_embedding(self):
        user = UserProfile(display_name="Bob")
        await self.store.create_user(user)
        emb = FaceEmbeddingRecord(
            user_id=user.id,
            embedding=[0.5] * 512,
        )
        await self.store.store_face_embedding(emb)
        all_embs = await self.store.get_all_face_embeddings()
        assert len(all_embs) == 1
        assert all_embs[0].user_id == user.id
        assert len(all_embs[0].embedding) == 512

    @pytest.mark.asyncio
    async def test_get_face_embeddings_for_user(self):
        user_a = UserProfile(display_name="Alice")
        user_b = UserProfile(display_name="Bob")
        await self.store.create_user(user_a)
        await self.store.create_user(user_b)
        await self.store.store_face_embedding(
            FaceEmbeddingRecord(user_id=user_a.id, embedding=[0.1] * 512)
        )
        await self.store.store_face_embedding(
            FaceEmbeddingRecord(user_id=user_b.id, embedding=[0.2] * 512)
        )
        embs_a = await self.store.get_face_embeddings_for_user(user_a.id)
        assert len(embs_a) == 1
        assert embs_a[0].user_id == user_a.id

    @pytest.mark.asyncio
    async def test_store_observation(self):
        sid = uuid4()
        obs = RecognitionObservation(
            session_id=sid,
            event_type="first_recognition",
            content="Recognized Alice.",
            confidence=0.9,
        )
        stored = await self.store.store_observation(obs)
        assert stored.id == obs.id

    @pytest.mark.asyncio
    async def test_create_face_enrollment_with_new_fields(self):
        user = UserProfile(display_name="Charlie")
        await self.store.create_user(user)
        enrollment = FaceEnrollment(
            user_id=user.id,
            status="enrolled",
            embedding_model="InceptionResnetV1-vggface2",
            face_count_detected=1,
            message="Enrolled.",
        )
        stored = await self.store.create_face_enrollment(enrollment)
        assert stored.status == "enrolled"


# --- Stub detector/recognizer tests ---


class TestStubs:
    def test_stub_detector_returns_empty(self):
        det = StubFaceDetector()
        det.load_model()
        result = det.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        assert result == []

    def test_stub_recognizer_returns_none(self):
        rec = StubFaceRecognizer()
        uid, score = rec.recognize([0.1] * 512, [])
        assert uid is None
        assert score == 0.0
