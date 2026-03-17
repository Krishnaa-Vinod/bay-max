"""Tests for iteration-006: Live webcam continuity runtime."""

import time
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pytest

from baymax.config.settings import BaymaxSettings
from baymax.core.enums import EngagementLevel, PostureLabel
from baymax.live.event_engine import EventEngine
from baymax.live.proactive_scheduler import ProactiveScheduler
from baymax.live.schemas import (
    CompanionEvent,
    CompanionEventType,
    CooldownState,
    LiveArtifactRef,
    LiveFrameResult,
    LiveRunSummary,
    LiveRuntimeStatus,
    LiveSourceType,
    ProactiveDecision,
    SessionLifecycleEvent,
    SessionLifecycleState,
)
from baymax.live.session_supervisor import SessionSupervisor

# ============================================================
# Schema / model tests
# ============================================================


class TestLiveSchemas:
    """Test that all live schemas instantiate and serialize correctly."""

    def test_live_runtime_status_defaults(self):
        status = LiveRuntimeStatus()
        assert status.live_mode_active is False
        assert status.source == ""
        assert status.session_status == SessionLifecycleState.IDLE
        assert status.current_user_id is None
        assert status.frame_count == 0

    def test_live_runtime_status_with_values(self):
        uid = uuid4()
        status = LiveRuntimeStatus(
            live_mode_active=True,
            source="webcam:0",
            session_status=SessionLifecycleState.ACTIVE,
            current_user_id=uid,
            current_user_display_name="Alice",
            last_event="recognized_user_arrived",
            last_response_text="Hello Alice!",
            frame_count=100,
            analysis_count=10,
            uptime_sec=120.5,
        )
        data = status.model_dump(mode="json")
        assert data["live_mode_active"] is True
        assert data["current_user_display_name"] == "Alice"
        assert data["frame_count"] == 100

    def test_live_frame_result(self):
        result = LiveFrameResult(frame_index=5, faces_detected=1, is_analysis_frame=True)
        assert result.frame_index == 5
        assert result.is_analysis_frame is True

    def test_session_lifecycle_event(self):
        sid = uuid4()
        evt = SessionLifecycleEvent(
            event_type="session_started",
            session_id=sid,
            reason="Presence threshold met",
        )
        assert evt.session_id == sid
        assert evt.event_type == "session_started"

    def test_companion_event(self):
        evt = CompanionEvent(
            event_type=CompanionEventType.RECOGNIZED_USER_ARRIVED,
            user_id=uuid4(),
            user_display_name="Bob",
            confidence=0.95,
        )
        assert evt.event_type == CompanionEventType.RECOGNIZED_USER_ARRIVED
        assert evt.suppressed is False

    def test_companion_event_suppressed(self):
        evt = CompanionEvent(
            event_type=CompanionEventType.POSTURE_CHANGED,
            suppressed=True,
            suppression_reason="Cooldown active",
        )
        assert evt.suppressed is True
        assert "Cooldown" in evt.suppression_reason

    def test_proactive_decision(self):
        event = CompanionEvent(event_type=CompanionEventType.RECOGNIZED_USER_ARRIVED)
        decision = ProactiveDecision(
            event=event,
            should_respond=True,
            trigger_reason="arrival_greeting",
        )
        assert decision.should_respond is True

    def test_cooldown_state(self):
        cs = CooldownState()
        assert cs.last_any_response_at is None
        assert cs.last_event_responses == {}

    def test_live_artifact_ref(self):
        ref = LiveArtifactRef(
            path="/tmp/events.jsonl",
            artifact_type="event_log",
        )
        assert ref.artifact_type == "event_log"

    def test_live_run_summary(self):
        summary = LiveRunSummary(
            source="replay",
            total_frames=100,
            total_events=5,
            total_responses=2,
        )
        assert summary.total_frames == 100
        assert summary.sessions_created == 0

    def test_source_type_enum(self):
        assert LiveSourceType.WEBCAM == "webcam"
        assert LiveSourceType.REPLAY == "replay"

    def test_session_lifecycle_state_enum(self):
        assert SessionLifecycleState.IDLE == "idle"
        assert SessionLifecycleState.ACTIVE == "active"
        assert SessionLifecycleState.PAUSED == "paused"

    def test_companion_event_type_enum(self):
        assert CompanionEventType.PERSON_ARRIVED == "person_arrived"
        assert CompanionEventType.QUIET_COMPANIONSHIP_DUE == "quiet_companionship_due"


# ============================================================
# Session Supervisor tests
# ============================================================


class TestSessionSupervisor:
    """Test automatic session lifecycle management."""

    def test_initial_state(self):
        sup = SessionSupervisor()
        assert sup.state == SessionLifecycleState.IDLE
        assert sup.current_session_id is None

    def test_session_start_on_presence(self):
        sup = SessionSupervisor(presence_min_frames=2)
        # Frame 1: presence detected
        events = sup.update(person_present=True)
        assert len(events) == 0  # Not enough consecutive frames

        # Frame 2: presence confirmed
        events = sup.update(person_present=True)
        assert len(events) == 1
        assert events[0].event_type == "session_started"
        assert sup.state == SessionLifecycleState.ACTIVE

    def test_no_session_without_presence(self):
        sup = SessionSupervisor(presence_min_frames=2)
        events = sup.update(person_present=False)
        assert len(events) == 0
        assert sup.state == SessionLifecycleState.IDLE

    def test_session_pause_on_absence(self):
        sup = SessionSupervisor(presence_min_frames=1, absence_timeout_sec=0.1)
        sid = uuid4()

        # Start session
        events = sup.update(person_present=True)
        assert events[0].event_type == "session_started"
        sup.set_session_id(sid)

        # Person leaves
        sup.update(person_present=False)
        time.sleep(0.15)  # Wait for timeout
        events = sup.update(person_present=False)
        assert any(e.event_type == "session_paused" for e in events)
        assert sup.state == SessionLifecycleState.PAUSED

    def test_session_resume_within_window(self):
        sup = SessionSupervisor(
            presence_min_frames=1,
            absence_timeout_sec=0.05,
            resume_window_sec=10.0,
        )
        sid = uuid4()
        uid = uuid4()

        # Start session
        sup.update(person_present=True, user_id=uid)
        sup.set_session_id(sid)

        # Pause
        sup.update(person_present=False)
        time.sleep(0.06)
        sup.update(person_present=False)

        assert sup.state == SessionLifecycleState.PAUSED

        # Return (within resume window)
        events = sup.update(person_present=True, user_id=uid)
        assert any(e.event_type == "session_resumed" for e in events)
        assert sup.state == SessionLifecycleState.ACTIVE

    def test_new_session_after_expired_window(self):
        sup = SessionSupervisor(
            presence_min_frames=1,
            absence_timeout_sec=0.05,
            resume_window_sec=0.1,
        )
        sid = uuid4()

        # Start session
        sup.update(person_present=True)
        sup.set_session_id(sid)

        # Pause
        sup.update(person_present=False)
        time.sleep(0.06)
        sup.update(person_present=False)
        assert sup.state == SessionLifecycleState.PAUSED

        # Wait beyond resume window
        time.sleep(0.12)
        events = sup.update(person_present=True)
        event_types = [e.event_type for e in events]
        assert "session_ended" in event_types
        assert "session_started" in event_types

    def test_force_end(self):
        sup = SessionSupervisor(presence_min_frames=1)
        sid = uuid4()

        sup.update(person_present=True)
        sup.set_session_id(sid)

        evt = sup.force_end()
        assert evt is not None
        assert evt.event_type == "session_ended"
        assert sup.state == SessionLifecycleState.IDLE

    def test_force_end_when_idle(self):
        sup = SessionSupervisor()
        evt = sup.force_end()
        assert evt is None

    def test_set_session_id(self):
        sup = SessionSupervisor()
        sid = uuid4()
        sup.set_session_id(sid)
        assert sup.current_session_id == sid

    def test_reset(self):
        sup = SessionSupervisor(presence_min_frames=1)
        sup.update(person_present=True)
        sup.set_session_id(uuid4())
        sup.reset()
        assert sup.state == SessionLifecycleState.IDLE
        assert sup.current_session_id is None
        assert len(sup.events) == 0


# ============================================================
# Event Engine tests
# ============================================================


class TestEventEngine:
    """Test event detection from state transitions."""

    def test_person_arrived_recognized(self):
        engine = EventEngine()
        uid = uuid4()
        events = engine.update(
            person_present=True,
            user_id=uid,
            user_display_name="Alice",
            identity_confidence=0.9,
            recognized=True,
        )
        types = [e.event_type for e in events]
        assert CompanionEventType.PERSON_ARRIVED in types
        assert CompanionEventType.RECOGNIZED_USER_ARRIVED in types

    def test_person_arrived_unknown(self):
        engine = EventEngine()
        events = engine.update(person_present=True, recognized=False)
        types = [e.event_type for e in events]
        assert CompanionEventType.PERSON_ARRIVED in types
        assert CompanionEventType.UNKNOWN_USER_ARRIVED in types

    def test_person_departed(self):
        engine = EventEngine()
        engine.update(person_present=True)  # Arrive
        events = engine.update(person_present=False)  # Depart
        types = [e.event_type for e in events]
        assert CompanionEventType.PERSON_DEPARTED in types

    def test_no_event_on_stable_presence(self):
        engine = EventEngine()
        engine.update(person_present=True, recognized=True, user_id=uuid4())
        events = engine.update(
            person_present=True, recognized=True, user_id=engine._prev_user_id
        )
        # Should not re-fire arrival events
        arrival_events = [
            e for e in events
            if e.event_type in (
                CompanionEventType.PERSON_ARRIVED,
                CompanionEventType.RECOGNIZED_USER_ARRIVED,
            )
        ]
        assert len(arrival_events) == 0

    def test_recognition_gained(self):
        engine = EventEngine()
        uid = uuid4()
        engine.update(person_present=True, recognized=False)  # Unknown first
        events = engine.update(
            person_present=True, recognized=True, user_id=uid, user_display_name="Bob"
        )
        types = [e.event_type for e in events]
        assert CompanionEventType.RECOGNITION_GAINED in types

    def test_recognition_lost(self):
        engine = EventEngine()
        uid = uuid4()
        engine.update(person_present=True, recognized=True, user_id=uid)
        events = engine.update(person_present=True, recognized=False)
        types = [e.event_type for e in events]
        assert CompanionEventType.RECOGNITION_LOST in types

    def test_posture_change_requires_stability(self):
        engine = EventEngine(stability_sec=0.1)

        # Start with upright
        engine._prev_posture = PostureLabel.UPRIGHT

        # Change to slouched (first frame, starts pending)
        events = engine.update(
            person_present=True,
            posture=PostureLabel.SLOUCHED,
        )
        posture_events = [
            e for e in events if e.event_type == CompanionEventType.POSTURE_CHANGED
        ]
        assert len(posture_events) == 0  # Not stable yet

        # Wait for stability
        time.sleep(0.12)
        events = engine.update(
            person_present=True,
            posture=PostureLabel.SLOUCHED,
        )
        posture_events = [
            e for e in events if e.event_type == CompanionEventType.POSTURE_CHANGED
        ]
        assert len(posture_events) == 1

    def test_engagement_change_requires_stability(self):
        engine = EventEngine(stability_sec=0.1)

        engine.update(
            person_present=True,
            engagement=EngagementLevel.HIGH,
        )

        # Change to LOW (first frame, starts pending)
        events = engine.update(
            person_present=True,
            engagement=EngagementLevel.LOW,
        )
        eng_events = [
            e for e in events if e.event_type == CompanionEventType.ENGAGEMENT_CHANGED
        ]
        assert len(eng_events) == 0

        time.sleep(0.12)
        events = engine.update(
            person_present=True,
            engagement=EngagementLevel.LOW,
        )
        eng_events = [
            e for e in events if e.event_type == CompanionEventType.ENGAGEMENT_CHANGED
        ]
        assert len(eng_events) == 1

    def test_quiet_companionship(self):
        engine = EventEngine(stability_sec=0.1)
        engine.set_quiet_interval(0.1)

        uid = uuid4()
        engine.update(person_present=True, recognized=True, user_id=uid)

        time.sleep(0.12)
        events = engine.update(
            person_present=True, recognized=True, user_id=uid
        )
        quiet_events = [
            e for e in events if e.event_type == CompanionEventType.QUIET_COMPANIONSHIP_DUE
        ]
        assert len(quiet_events) == 1

    def test_reset(self):
        engine = EventEngine()
        engine.update(person_present=True)
        engine.reset()
        assert len(engine.events) == 0
        assert engine._prev_person_present is False


# ============================================================
# Proactive Scheduler tests
# ============================================================


class TestProactiveScheduler:
    """Test cooldown-based response scheduling."""

    def test_first_greeting_allowed(self):
        sched = ProactiveScheduler()
        event = CompanionEvent(event_type=CompanionEventType.RECOGNIZED_USER_ARRIVED)
        decision = sched.evaluate(event)
        assert decision.should_respond is True
        assert decision.trigger_reason == "arrival_greeting"

    def test_departure_does_not_trigger_response(self):
        sched = ProactiveScheduler()
        event = CompanionEvent(event_type=CompanionEventType.PERSON_DEPARTED)
        decision = sched.evaluate(event)
        assert decision.should_respond is False
        assert not decision.suppressed

    def test_cooldown_suppression(self):
        sched = ProactiveScheduler(
            any_response_min_interval_sec=60.0,
            proactive_min_interval_sec=60.0,
        )

        # First response
        event1 = CompanionEvent(event_type=CompanionEventType.RECOGNIZED_USER_ARRIVED)
        decision1 = sched.evaluate(event1)
        assert decision1.should_respond is True
        sched.record_response(event1)

        # Second response should be suppressed
        event2 = CompanionEvent(event_type=CompanionEventType.POSTURE_CHANGED)
        decision2 = sched.evaluate(event2)
        assert decision2.should_respond is False
        assert decision2.suppressed is True
        assert decision2.cooldown_remaining_sec > 0

    def test_response_after_cooldown(self):
        sched = ProactiveScheduler(
            any_response_min_interval_sec=0.05,
            proactive_min_interval_sec=0.05,
        )

        event1 = CompanionEvent(event_type=CompanionEventType.RECOGNIZED_USER_ARRIVED)
        sched.evaluate(event1)
        sched.record_response(event1)

        time.sleep(0.06)

        event2 = CompanionEvent(event_type=CompanionEventType.POSTURE_CHANGED)
        decision2 = sched.evaluate(event2)
        assert decision2.should_respond is True

    def test_record_response_updates_counts(self):
        sched = ProactiveScheduler()
        event = CompanionEvent(event_type=CompanionEventType.RECOGNIZED_USER_ARRIVED)
        assert sched.total_responses == 0
        sched.record_response(event)
        assert sched.total_responses == 1

    def test_suppression_count(self):
        sched = ProactiveScheduler(
            any_response_min_interval_sec=60.0,
            proactive_min_interval_sec=60.0,
        )
        event = CompanionEvent(event_type=CompanionEventType.RECOGNIZED_USER_ARRIVED)
        sched.evaluate(event)
        sched.record_response(event)

        event2 = CompanionEvent(event_type=CompanionEventType.POSTURE_CHANGED)
        sched.evaluate(event2)
        assert sched.total_suppressions == 1

    def test_unknown_user_greeting(self):
        sched = ProactiveScheduler()
        event = CompanionEvent(event_type=CompanionEventType.UNKNOWN_USER_ARRIVED)
        decision = sched.evaluate(event)
        assert decision.should_respond is True
        assert decision.trigger_reason == "arrival_greeting"

    def test_quiet_companionship_trigger(self):
        sched = ProactiveScheduler()
        event = CompanionEvent(event_type=CompanionEventType.QUIET_COMPANIONSHIP_DUE)
        decision = sched.evaluate(event)
        assert decision.should_respond is True
        assert decision.trigger_reason == "quiet_companionship"

    def test_recognition_gained_trigger(self):
        sched = ProactiveScheduler()
        event = CompanionEvent(event_type=CompanionEventType.RECOGNITION_GAINED)
        decision = sched.evaluate(event)
        assert decision.should_respond is True
        assert decision.trigger_reason == "recognition_greeting"

    def test_reset(self):
        sched = ProactiveScheduler()
        event = CompanionEvent(event_type=CompanionEventType.RECOGNIZED_USER_ARRIVED)
        sched.record_response(event)
        sched.reset()
        assert sched.total_responses == 0
        assert sched.cooldown_state.last_any_response_at is None


# ============================================================
# Settings tests
# ============================================================


class TestLiveSettings:
    """Test that iteration-006 settings are properly configured."""

    def test_default_live_settings(self):
        with patch.dict("os.environ", {}, clear=False):
            settings = BaymaxSettings()
            assert settings.live_source == "webcam"
            assert settings.live_camera_index == 0
            assert settings.preview_fps == 8
            assert settings.analysis_interval_sec == 2.0
            assert settings.presence_min_consecutive_frames == 2
            assert settings.absence_timeout_sec == 30.0
            assert settings.session_resume_window_sec == 300.0
            assert settings.proactive_min_interval_sec == 30.0
            assert settings.any_response_min_interval_sec == 10.0
            assert settings.quiet_companionship_interval_sec == 300.0
            assert settings.event_min_stability_sec == 4.0
            assert settings.enable_live_overlay is True
            assert settings.enable_artifact_logging is True
            assert settings.live_max_run_sec == 0.0
            assert settings.live_video_replay_path == ""


# ============================================================
# API endpoint tests
# ============================================================


class TestLiveAPIEndpoints:
    """Test the live status and control API endpoints."""

    @pytest.fixture
    def tmp_db(self, tmp_path):
        return str(tmp_path / "test_iter6_api.db")

    def test_live_status_inactive(self, tmp_db):
        import apps.api.main as api_module
        from fastapi.testclient import TestClient

        client = TestClient(api_module.app)
        resp = client.get("/v1/live/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["live_mode_active"] is False

    def test_live_control_stop_when_inactive(self, tmp_db):
        import apps.api.main as api_module
        from fastapi.testclient import TestClient

        client = TestClient(api_module.app)
        resp = client.post(
            "/v1/live/control",
            json={"action": "stop"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_live_control_start_not_via_api(self, tmp_db):
        import apps.api.main as api_module
        from fastapi.testclient import TestClient

        client = TestClient(api_module.app)
        resp = client.post(
            "/v1/live/control",
            json={"action": "start"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"

    def test_live_control_unknown_action(self, tmp_db):
        import apps.api.main as api_module
        from fastapi.testclient import TestClient

        client = TestClient(api_module.app)
        resp = client.post(
            "/v1/live/control",
            json={"action": "restart"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"


# ============================================================
# Artifact Logger tests
# ============================================================


class TestArtifactLogger:
    """Test artifact logging functionality."""

    def test_log_event(self, tmp_path):
        from baymax.live.artifact_logger import ArtifactLogger

        logger = ArtifactLogger(str(tmp_path), enabled=True)
        event = CompanionEvent(
            event_type=CompanionEventType.RECOGNIZED_USER_ARRIVED,
            user_display_name="Alice",
        )
        logger.log_event(event)

        events_file = tmp_path / "events.jsonl"
        assert events_file.exists()
        import json
        with open(events_file) as f:
            entry = json.loads(f.readline())
        assert entry["event_type"] == "recognized_user_arrived"

    def test_log_response(self, tmp_path):
        from baymax.live.artifact_logger import ArtifactLogger

        logger = ArtifactLogger(str(tmp_path), enabled=True)
        event = CompanionEvent(event_type=CompanionEventType.RECOGNIZED_USER_ARRIVED)
        logger.log_response(
            event=event,
            response_text="Hello!",
            strategy="greet",
            memory_refs=["past session content"],
            backend="rule_based",
            trigger_reason="arrival_greeting",
        )

        resp_file = tmp_path / "responses.jsonl"
        assert resp_file.exists()
        import json
        with open(resp_file) as f:
            entry = json.loads(f.readline())
        assert entry["response_text"] == "Hello!"
        assert entry["trigger_reason"] == "arrival_greeting"

    def test_log_suppression(self, tmp_path):
        from baymax.live.artifact_logger import ArtifactLogger

        logger = ArtifactLogger(str(tmp_path), enabled=True)
        event = CompanionEvent(event_type=CompanionEventType.POSTURE_CHANGED)
        decision = ProactiveDecision(
            event=event,
            suppressed=True,
            suppression_reason="Cooldown active: 25s remaining",
            cooldown_remaining_sec=25.0,
        )
        logger.log_suppression(decision)

        events_file = tmp_path / "events.jsonl"
        assert events_file.exists()

    def test_disabled_logging(self, tmp_path):
        from baymax.live.artifact_logger import ArtifactLogger

        logger = ArtifactLogger(str(tmp_path), enabled=False)
        event = CompanionEvent(event_type=CompanionEventType.PERSON_ARRIVED)
        logger.log_event(event)

        events_file = tmp_path / "events.jsonl"
        assert not events_file.exists()

    def test_write_session_timeline(self, tmp_path):
        from baymax.live.artifact_logger import ArtifactLogger

        logger = ArtifactLogger(str(tmp_path), enabled=True)
        evt = SessionLifecycleEvent(
            event_type="session_started",
            session_id=uuid4(),
            reason="Presence threshold met",
        )
        logger.log_session_event(evt)
        logger.write_session_timeline()

        timeline_file = tmp_path / "session_timeline.json"
        assert timeline_file.exists()

    def test_write_manifest_and_summary(self, tmp_path):
        from baymax.live.artifact_logger import ArtifactLogger

        logger = ArtifactLogger(str(tmp_path), enabled=True)
        summary = LiveRunSummary(
            source="replay",
            total_frames=50,
            total_events=3,
            total_responses=1,
            backend="rule_based",
        )
        logger.write_manifest(summary)
        logger.write_summary(summary)

        assert (tmp_path / "manifest.json").exists()
        assert (tmp_path / "summary.md").exists()


# ============================================================
# Frame Source tests
# ============================================================


class TestFrameSource:
    """Test frame source implementations."""

    def test_folder_frame_source(self, tmp_path):
        """Test folder frame source with synthetic images."""
        from PIL import Image

        from baymax.live.frame_source import FolderFrameSource

        # Create some test frames
        for i in range(3):
            img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
            img.save(tmp_path / f"frame_{i:03d}.jpg")

        # Use very high target_fps so throttle doesn't kick in
        source = FolderFrameSource(str(tmp_path), target_fps=100000)
        source.open()
        assert source.is_open()

        frames = []
        # Read until source is exhausted
        while source.is_open():
            frame = source.read_frame()
            if frame is not None:
                frames.append(frame)
        source.close()

        assert len(frames) == 3
        assert not source.is_open()

    def test_folder_source_missing_dir(self):
        from baymax.live.frame_source import FolderFrameSource

        source = FolderFrameSource("/nonexistent/path")
        with pytest.raises(RuntimeError, match="not found"):
            source.open()

    def test_folder_source_empty_dir(self, tmp_path):
        from baymax.live.frame_source import FolderFrameSource

        source = FolderFrameSource(str(tmp_path))
        with pytest.raises(RuntimeError, match="No image files"):
            source.open()


# ============================================================
# Integration-level tests
# ============================================================


class TestLiveIntegration:
    """Integration tests combining session supervisor + event engine + scheduler."""

    def test_arrival_to_greeting_flow(self):
        """Simulate a user arriving and verify the complete event flow."""
        sup = SessionSupervisor(presence_min_frames=1, absence_timeout_sec=5.0)
        engine = EventEngine(stability_sec=0.1)
        sched = ProactiveScheduler()

        uid = uuid4()
        sid = uuid4()

        # User arrives
        lifecycle_events = sup.update(person_present=True, user_id=uid)
        assert any(e.event_type == "session_started" for e in lifecycle_events)
        sup.set_session_id(sid)

        # Event engine detects arrival
        companion_events = engine.update(
            person_present=True,
            user_id=uid,
            user_display_name="Alice",
            identity_confidence=0.9,
            recognized=True,
            session_id=sid,
        )
        assert any(
            e.event_type == CompanionEventType.RECOGNIZED_USER_ARRIVED
            for e in companion_events
        )

        # Scheduler allows response
        for event in companion_events:
            decision = sched.evaluate(event)
            if event.event_type == CompanionEventType.RECOGNIZED_USER_ARRIVED:
                assert decision.should_respond is True
                sched.record_response(event)

    def test_absence_pause_resume_flow(self):
        """Simulate absence pause and resume."""
        sup = SessionSupervisor(
            presence_min_frames=1,
            absence_timeout_sec=0.05,
            resume_window_sec=10.0,
        )
        uid = uuid4()
        sid = uuid4()

        # Start session
        sup.update(person_present=True, user_id=uid)
        sup.set_session_id(sid)
        assert sup.state == SessionLifecycleState.ACTIVE

        # Person leaves
        sup.update(person_present=False)
        time.sleep(0.06)
        events = sup.update(person_present=False)
        assert any(e.event_type == "session_paused" for e in events)

        # Person returns
        events = sup.update(person_present=True, user_id=uid)
        assert any(e.event_type == "session_resumed" for e in events)
        assert sup.state == SessionLifecycleState.ACTIVE

    def test_cooldown_prevents_spam(self):
        """Verify that repeated events don't cause response spam."""
        sched = ProactiveScheduler(
            proactive_min_interval_sec=60.0,
            any_response_min_interval_sec=60.0,
        )

        # First response is allowed
        event1 = CompanionEvent(event_type=CompanionEventType.RECOGNIZED_USER_ARRIVED)
        assert sched.evaluate(event1).should_respond is True
        sched.record_response(event1)

        # All subsequent events are suppressed
        for _ in range(5):
            event = CompanionEvent(event_type=CompanionEventType.POSTURE_CHANGED)
            decision = sched.evaluate(event)
            assert decision.should_respond is False
            assert decision.suppressed is True

        assert sched.total_suppressions == 5

    def test_known_vs_unknown_event_path(self):
        """Verify that known and unknown users produce different event types."""
        engine = EventEngine()

        # Unknown arrives
        events_unknown = engine.update(person_present=True, recognized=False)
        assert any(
            e.event_type == CompanionEventType.UNKNOWN_USER_ARRIVED
            for e in events_unknown
        )

        engine.reset()

        # Known arrives
        uid = uuid4()
        events_known = engine.update(
            person_present=True,
            recognized=True,
            user_id=uid,
            user_display_name="Alice",
        )
        assert any(
            e.event_type == CompanionEventType.RECOGNIZED_USER_ARRIVED
            for e in events_known
        )


# ============================================================
# project_state.json validity test
# ============================================================


class TestProjectState:
    """Validate that project_state.json is valid and up to date."""

    def test_project_state_valid(self):
        import json
        import os
        state_path = os.path.join(
            os.path.dirname(__file__), "..", "docs", "project_state.json"
        )
        with open(state_path) as f:
            state = json.load(f)
        assert "iteration" in state
        assert "status" in state
        assert "modules_implemented" in state
