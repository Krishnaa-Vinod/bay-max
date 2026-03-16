"""Tests for Iteration 003: pose estimation, heuristics, engagement, body-state."""

from uuid import uuid4

import numpy as np
import pytest

from baymax.core.enums import EngagementLevel, LeanLabel, MotionLevel, PostureLabel
from baymax.memory.store_sqlite import SQLiteMetadataStore
from baymax.perception.heuristics import (
    MotionTracker,
    compute_engagement,
    compute_lean,
    compute_posture,
)
from baymax.perception.pose_interface import StubPoseEstimator
from baymax.schemas.perception import (
    AnnotatedArtifactRef,
    BodyStateObservation,
    EngagementResult,
    FrameAnalysisResult,
    PoseLandmark2D,
    PoseResult,
)
from baymax.state.manager import StateManager
from baymax.state.models import InteractionState

# --- Helpers ---

def _make_landmarks_33(
    nose_y: float = 0.20,
    shoulder_y: float = 0.35,
    hip_y: float = 0.60,
    nose_x: float = 0.50,
    shoulder_x: float = 0.50,
    visibility: float = 0.9,
) -> list[PoseLandmark2D]:
    """Create a simplified 33-landmark set with key landmarks positioned.

    Index reference: 0=nose, 11=left_shoulder, 12=right_shoulder,
    23=left_hip, 24=right_hip.
    """
    lms = [
        PoseLandmark2D(x=0.5, y=0.5, visibility=0.0, name=f"lm_{i}")
        for i in range(33)
    ]
    # Nose
    lms[0] = PoseLandmark2D(x=nose_x, y=nose_y, visibility=visibility, name="nose")
    # Shoulders
    lms[11] = PoseLandmark2D(
        x=shoulder_x - 0.08, y=shoulder_y, visibility=visibility, name="left_shoulder"
    )
    lms[12] = PoseLandmark2D(
        x=shoulder_x + 0.08, y=shoulder_y, visibility=visibility, name="right_shoulder"
    )
    # Hips
    lms[23] = PoseLandmark2D(
        x=shoulder_x - 0.06, y=hip_y, visibility=visibility, name="left_hip"
    )
    lms[24] = PoseLandmark2D(
        x=shoulder_x + 0.06, y=hip_y, visibility=visibility, name="right_hip"
    )
    return lms


def _make_pose(
    pose_present: bool = True, **kwargs
) -> PoseResult:
    """Create a PoseResult with configurable landmarks."""
    if not pose_present:
        return PoseResult(pose_present=False, backend="test")
    lms = _make_landmarks_33(**kwargs)
    return PoseResult(
        pose_present=True,
        landmarks=lms,
        landmark_count=len(lms),
        confidence=0.85,
        backend="test",
    )


# --- Enum tests ---


class TestIteration003Enums:
    def test_posture_label_values(self):
        assert len(PostureLabel) == 4
        assert PostureLabel.UPRIGHT == "upright"
        assert PostureLabel.SLOUCHED == "slouched"
        assert PostureLabel.RECLINED == "reclined"
        assert PostureLabel.UNKNOWN == "unknown"

    def test_lean_label_values(self):
        assert len(LeanLabel) == 4
        assert LeanLabel.FORWARD == "forward"
        assert LeanLabel.NEUTRAL == "neutral"
        assert LeanLabel.BACKWARD == "backward"
        assert LeanLabel.UNKNOWN == "unknown"

    def test_motion_level_values(self):
        assert len(MotionLevel) == 4
        assert MotionLevel.LOW == "low"
        assert MotionLevel.MEDIUM == "medium"
        assert MotionLevel.HIGH == "high"
        assert MotionLevel.UNKNOWN == "unknown"


# --- Schema tests ---


class TestIteration003Schemas:
    def test_pose_landmark_2d(self):
        lm = PoseLandmark2D(x=0.5, y=0.3, visibility=0.95, name="nose")
        assert lm.x == 0.5
        assert lm.y == 0.3
        assert lm.visibility == 0.95
        assert lm.name == "nose"

    def test_pose_landmark_2d_defaults(self):
        lm = PoseLandmark2D(x=0.0, y=0.0)
        assert lm.visibility == 0.0
        assert lm.name == ""

    def test_pose_result_defaults(self):
        pr = PoseResult()
        assert pr.pose_present is False
        assert pr.landmarks == []
        assert pr.landmark_count == 0
        assert pr.confidence == 0.0
        assert pr.backend == "mediapipe"

    def test_pose_result_with_landmarks(self):
        lms = [PoseLandmark2D(x=0.1, y=0.2, visibility=0.8, name="nose")]
        pr = PoseResult(
            pose_present=True, landmarks=lms, landmark_count=1, confidence=0.9
        )
        assert pr.pose_present is True
        assert len(pr.landmarks) == 1
        assert pr.landmarks[0].name == "nose"

    def test_engagement_result_defaults(self):
        er = EngagementResult()
        assert er.level == EngagementLevel.MEDIUM
        assert er.posture == PostureLabel.UNKNOWN
        assert er.lean == LeanLabel.UNKNOWN
        assert er.motion == MotionLevel.UNKNOWN
        assert er.face_visible is False
        assert er.score == 0.0

    def test_engagement_result_with_values(self):
        er = EngagementResult(
            level=EngagementLevel.HIGH,
            posture=PostureLabel.UPRIGHT,
            lean=LeanLabel.FORWARD,
            motion=MotionLevel.LOW,
            face_visible=True,
            face_recognized=True,
            pose_visible=True,
            score=0.9,
            evidence=["face_visible", "posture=upright"],
        )
        assert er.level == EngagementLevel.HIGH
        assert er.score == 0.9
        assert len(er.evidence) == 2

    def test_annotated_artifact_ref(self):
        ref = AnnotatedArtifactRef(path="/tmp/test.jpg")
        assert ref.artifact_type == "annotated_frame"
        assert ref.session_id is None

    def test_body_state_observation(self):
        sid = uuid4()
        uid = uuid4()
        obs = BodyStateObservation(
            session_id=sid,
            user_id=uid,
            event_type="posture_change",
            content="Posture changed from upright to slouched.",
            confidence=0.7,
            evidence_refs=[f"session:{sid}"],
        )
        assert obs.session_id == sid
        assert obs.user_id == uid
        assert obs.event_type == "posture_change"
        assert "slouched" in obs.content

    def test_frame_analysis_result_with_pose_engagement(self):
        sid = uuid4()
        pr = PoseResult(pose_present=True, landmark_count=33, confidence=0.9)
        er = EngagementResult(level=EngagementLevel.HIGH, score=0.8)
        result = FrameAnalysisResult(
            session_id=sid,
            faces_detected=1,
            pose_result=pr,
            engagement=er,
        )
        assert result.pose_result is not None
        assert result.pose_result.pose_present is True
        assert result.engagement.level == EngagementLevel.HIGH


# --- Posture heuristic tests ---


class TestComputePosture:
    def test_no_pose_returns_unknown(self):
        pose = PoseResult(pose_present=False)
        assert compute_posture(pose) == PostureLabel.UNKNOWN

    def test_empty_landmarks_returns_unknown(self):
        pose = PoseResult(pose_present=True, landmarks=[])
        assert compute_posture(pose) == PostureLabel.UNKNOWN

    def test_upright_posture(self):
        # Nose well above shoulders, standard upright pose
        pose = _make_pose(nose_y=0.15, shoulder_y=0.35, hip_y=0.60)
        result = compute_posture(pose)
        assert result == PostureLabel.UPRIGHT

    def test_slouched_posture(self):
        # Nose closer to shoulders (small nose-shoulder gap)
        # nose_shoulder_ratio = (0.35 - 0.30) / (0.60 - 0.35) = 0.05/0.25 = 0.20
        # 0.20 < POSTURE_SLOUCH_RATIO_THRESHOLD(0.38) -> slouched
        # 0.20 > POSTURE_RECLINE_RATIO_THRESHOLD(0.15) -> not reclined
        pose = _make_pose(nose_y=0.30, shoulder_y=0.35, hip_y=0.60)
        result = compute_posture(pose)
        assert result == PostureLabel.SLOUCHED

    def test_reclined_posture(self):
        # Nose below shoulders (negative or very low ratio)
        # nose_shoulder_ratio = (0.35 - 0.37) / (0.60 - 0.35) = -0.02/0.25 = -0.08
        # -0.08 < POSTURE_RECLINE_RATIO_THRESHOLD(0.15) -> reclined
        pose = _make_pose(nose_y=0.37, shoulder_y=0.35, hip_y=0.60)
        result = compute_posture(pose)
        assert result == PostureLabel.RECLINED

    def test_shoulders_at_hips_returns_reclined(self):
        # Shoulders at or below hips -> torso_height <= 0.01 -> reclined
        pose = _make_pose(nose_y=0.50, shoulder_y=0.60, hip_y=0.60)
        result = compute_posture(pose)
        assert result == PostureLabel.RECLINED

    def test_low_visibility_landmarks_returns_unknown(self):
        lms = _make_landmarks_33(visibility=0.1)  # Below MIN_VISIBILITY
        pose = PoseResult(pose_present=True, landmarks=lms, landmark_count=33)
        result = compute_posture(pose)
        assert result == PostureLabel.UNKNOWN


# --- Lean heuristic tests ---


class TestComputeLean:
    def test_no_pose_returns_unknown(self):
        pose = PoseResult(pose_present=False)
        assert compute_lean(pose) == LeanLabel.UNKNOWN

    def test_neutral_lean(self):
        # Nose at shoulder level but slightly below -> neutral offset
        # nose_offset = (0.34 - 0.35) / (0.60 - 0.35) = -0.01/0.25 = -0.04
        # -0.04 is not > LEAN_BACKWARD_OFFSET(0.04), not < LEAN_FORWARD_OFFSET(-0.03)
        # Actually -0.04 < -0.03 -> FORWARD
        # Let's pick values that land in neutral:
        # nose_offset = (0.345 - 0.35) / (0.60 - 0.35) = -0.005/0.25 = -0.02
        # -0.02 is not < -0.03 and not > 0.04 -> NEUTRAL
        pose = _make_pose(nose_y=0.345, shoulder_y=0.35, hip_y=0.60)
        result = compute_lean(pose)
        assert result == LeanLabel.NEUTRAL

    def test_forward_lean(self):
        # nose_offset needs to be < LEAN_FORWARD_OFFSET (-0.03)
        # nose_offset = (nose_y - shoulder_mid_y) / torso_height
        # Want nose well above shoulders: nose_y=0.30, shoulder_y=0.40, hip_y=0.65
        # offset = (0.30 - 0.40) / (0.65 - 0.40) = -0.10/0.25 = -0.40
        pose = _make_pose(nose_y=0.30, shoulder_y=0.40, hip_y=0.65)
        result = compute_lean(pose)
        assert result == LeanLabel.FORWARD

    def test_backward_lean(self):
        # nose_offset needs to be > LEAN_BACKWARD_OFFSET (0.04)
        # Nose well below shoulder midpoint
        # offset = (0.42 - 0.35) / (0.60 - 0.35) = 0.07/0.25 = 0.28
        pose = _make_pose(nose_y=0.42, shoulder_y=0.35, hip_y=0.60)
        result = compute_lean(pose)
        assert result == LeanLabel.BACKWARD

    def test_low_visibility_returns_unknown(self):
        lms = _make_landmarks_33(visibility=0.1)
        pose = PoseResult(pose_present=True, landmarks=lms, landmark_count=33)
        result = compute_lean(pose)
        assert result == LeanLabel.UNKNOWN


# --- Motion tracker tests ---


class TestMotionTracker:
    def test_single_frame_returns_unknown(self):
        tracker = MotionTracker(window_size=5)
        pose = _make_pose()
        result = tracker.update(pose)
        assert result == MotionLevel.UNKNOWN

    def test_no_pose_returns_unknown(self):
        tracker = MotionTracker()
        pose = PoseResult(pose_present=False)
        result = tracker.update(pose)
        assert result == MotionLevel.UNKNOWN

    def test_stationary_returns_low(self):
        tracker = MotionTracker(window_size=5)
        pose = _make_pose()
        tracker.update(pose)
        # Same pose again -> zero displacement -> LOW
        result = tracker.update(pose)
        assert result == MotionLevel.LOW

    def test_large_movement_returns_high(self):
        tracker = MotionTracker(window_size=5)
        pose1 = _make_pose(nose_y=0.20, shoulder_y=0.35, hip_y=0.60)
        pose2 = _make_pose(nose_y=0.30, shoulder_y=0.45, hip_y=0.70)
        tracker.update(pose1)
        result = tracker.update(pose2)
        assert result == MotionLevel.HIGH

    def test_small_movement_returns_medium(self):
        tracker = MotionTracker(window_size=5)
        pose1 = _make_pose(nose_y=0.20, shoulder_y=0.35, hip_y=0.60)
        # Small shift: ~0.02 in y for key landmarks
        pose2 = _make_pose(nose_y=0.22, shoulder_y=0.37, hip_y=0.62)
        tracker.update(pose1)
        result = tracker.update(pose2)
        assert result == MotionLevel.MEDIUM

    def test_clear_resets_history(self):
        tracker = MotionTracker(window_size=5)
        pose = _make_pose()
        tracker.update(pose)
        tracker.update(pose)
        tracker.clear()
        # After clear, next frame should be UNKNOWN (no previous)
        result = tracker.update(pose)
        assert result == MotionLevel.UNKNOWN


# --- Engagement compute tests ---


class TestComputeEngagement:
    def test_all_positive_signals(self):
        result = compute_engagement(
            posture=PostureLabel.UPRIGHT,
            lean=LeanLabel.FORWARD,
            motion=MotionLevel.LOW,
            face_visible=True,
            face_recognized=True,
            pose_visible=True,
        )
        assert result.level == EngagementLevel.HIGH
        # 0.20 + 0.10 + 0.10 + 0.20 + 0.15 + 0.15 = 0.90
        assert result.score == 0.90

    def test_all_negative_signals(self):
        result = compute_engagement(
            posture=PostureLabel.RECLINED,
            lean=LeanLabel.BACKWARD,
            motion=MotionLevel.UNKNOWN,
            face_visible=False,
            face_recognized=False,
            pose_visible=False,
        )
        assert result.level == EngagementLevel.LOW
        assert result.score == 0.0

    def test_medium_engagement(self):
        result = compute_engagement(
            posture=PostureLabel.SLOUCHED,
            lean=LeanLabel.NEUTRAL,
            motion=MotionLevel.MEDIUM,
            face_visible=True,
            face_recognized=False,
            pose_visible=True,
        )
        # 0.20 + 0.10 + 0.05 + 0.10 + 0.10 = 0.55
        assert result.level == EngagementLevel.MEDIUM
        assert result.score == 0.55

    def test_unknown_signals_contribute_zero(self):
        result = compute_engagement(
            posture=PostureLabel.UNKNOWN,
            lean=LeanLabel.UNKNOWN,
            motion=MotionLevel.UNKNOWN,
            face_visible=True,
            face_recognized=True,
            pose_visible=True,
        )
        # Only face and pose signals: 0.20 + 0.10 + 0.10 = 0.40
        assert result.score == 0.40
        assert result.level == EngagementLevel.MEDIUM

    def test_evidence_populated(self):
        result = compute_engagement(
            posture=PostureLabel.UPRIGHT,
            lean=LeanLabel.FORWARD,
            motion=MotionLevel.LOW,
            face_visible=True,
            face_recognized=True,
            pose_visible=True,
        )
        assert "face_visible" in result.evidence
        assert "face_recognized" in result.evidence
        assert "pose_visible" in result.evidence
        assert "posture=upright" in result.evidence
        assert "lean=forward" in result.evidence
        assert "motion=low" in result.evidence

    def test_engagement_result_fields_set(self):
        result = compute_engagement(
            posture=PostureLabel.UPRIGHT,
            lean=LeanLabel.NEUTRAL,
            motion=MotionLevel.LOW,
            face_visible=True,
            face_recognized=False,
            pose_visible=True,
        )
        assert result.posture == PostureLabel.UPRIGHT
        assert result.lean == LeanLabel.NEUTRAL
        assert result.motion == MotionLevel.LOW
        assert result.face_visible is True
        assert result.face_recognized is False
        assert result.pose_visible is True


# --- Stub pose estimator tests ---


class TestStubPoseEstimator:
    def test_returns_no_pose(self):
        estimator = StubPoseEstimator()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = estimator.estimate(frame)
        assert result.pose_present is False
        assert result.backend == "stub"

    def test_load_model_is_noop(self):
        estimator = StubPoseEstimator()
        estimator.load_model()  # Should not raise


# --- State manager body-state tests ---


class TestStateManagerBodyState:
    def test_update_body_state(self):
        sm = StateManager()
        sid = uuid4()
        sm.get_or_create(sid)

        engagement = EngagementResult(
            level=EngagementLevel.HIGH,
            posture=PostureLabel.UPRIGHT,
            lean=LeanLabel.FORWARD,
            motion=MotionLevel.LOW,
            face_visible=True,
            face_recognized=True,
            pose_visible=True,
            score=0.9,
        )
        result = sm.update_body_state(sid, engagement)

        assert result is not None
        assert result.pose_visible is True
        assert result.posture == PostureLabel.UPRIGHT
        assert result.lean == LeanLabel.FORWARD
        assert result.motion == MotionLevel.LOW
        assert result.engagement_level == EngagementLevel.HIGH
        assert result.engagement_score == 0.9

    def test_update_body_state_nonexistent_session(self):
        sm = StateManager()
        engagement = EngagementResult(level=EngagementLevel.LOW, score=0.1)
        result = sm.update_body_state(uuid4(), engagement)
        assert result is None

    def test_interaction_state_defaults(self):
        state = InteractionState(session_id=uuid4())
        assert state.pose_visible is False
        assert state.posture == PostureLabel.UNKNOWN
        assert state.lean == LeanLabel.UNKNOWN
        assert state.motion == MotionLevel.UNKNOWN
        assert state.engagement_score == 0.0


# --- SQLite store body-state observation tests ---


class TestSQLiteStoreBodyState:
    @pytest.fixture
    async def store(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        store = SQLiteMetadataStore(db_path=db_path)
        await store.initialize()
        yield store
        await store.close()

    @pytest.mark.asyncio
    async def test_store_body_state_observation(self, store):
        sid = uuid4()
        obs = BodyStateObservation(
            session_id=sid,
            user_id=uuid4(),
            event_type="posture_change",
            content="Posture changed from upright to slouched.",
            confidence=0.7,
            evidence_refs=[f"session:{sid}"],
        )
        result = await store.store_observation(obs)
        assert result.id == obs.id
        assert result.event_type == "posture_change"

    @pytest.mark.asyncio
    async def test_store_engagement_change_observation(self, store):
        sid = uuid4()
        obs = BodyStateObservation(
            session_id=sid,
            event_type="engagement_change",
            content="Engagement changed from medium to high (score: 0.85).",
            confidence=0.7,
            evidence_refs=[f"session:{sid}"],
        )
        result = await store.store_observation(obs)
        assert result.id == obs.id
        assert result.event_type == "engagement_change"

    @pytest.mark.asyncio
    async def test_store_pose_first_seen_observation(self, store):
        sid = uuid4()
        obs = BodyStateObservation(
            session_id=sid,
            event_type="pose_first_seen",
            content="Body pose detected for the first time.",
            confidence=0.8,
        )
        result = await store.store_observation(obs)
        assert result.event_type == "pose_first_seen"
