"""Tests for memory and schema models."""

from uuid import uuid4

import pytest

from baymax.core.enums import (
    EmotionLabel,
    EngagementLevel,
    MemoryStatus,
    MemoryType,
    PerceptionEventType,
    ResponseStrategy,
    SessionStatus,
)
from baymax.schemas.memory import (
    EpisodicMemory,
    InterventionMemory,
    MemoryQuery,
    MemoryQueryResult,
    ProjectState,
    RetrievalLog,
    SemanticFact,
)
from baymax.schemas.perception import Observation, PerceptionEvent
from baymax.schemas.response import RespondRequest, SupportiveResponse
from baymax.schemas.session import Session
from baymax.schemas.user import FaceEnrollment, UserProfile, UserProfileCreate


class TestUserSchemas:
    def test_create_user_profile(self):
        user = UserProfile(display_name="Alice")
        assert user.display_name == "Alice"
        assert user.is_active is True
        assert user.id is not None

    def test_user_profile_create_validation(self):
        req = UserProfileCreate(display_name="Bob")
        assert req.display_name == "Bob"

    def test_user_profile_create_rejects_empty_name(self):
        with pytest.raises(Exception):
            UserProfileCreate(display_name="")

    def test_face_enrollment(self):
        uid = uuid4()
        enrollment = FaceEnrollment(user_id=uid)
        assert enrollment.user_id == uid
        assert enrollment.status == "pending"
        assert enrollment.confidence == 0.0


class TestSessionSchemas:
    def test_create_session(self):
        session = Session()
        assert session.status == SessionStatus.ACTIVE
        assert session.frame_count == 0

    def test_session_with_user(self):
        uid = uuid4()
        session = Session(user_id=uid)
        assert session.user_id == uid


class TestPerceptionSchemas:
    def test_perception_event(self):
        sid = uuid4()
        event = PerceptionEvent(
            session_id=sid,
            event_type=PerceptionEventType.FACE_DETECTED,
            confidence=0.9,
        )
        assert event.session_id == sid
        assert event.event_type == PerceptionEventType.FACE_DETECTED
        assert event.confidence == 0.9

    def test_observation(self):
        sid = uuid4()
        obs = Observation(
            session_id=sid,
            content="User is looking at the screen",
        )
        assert obs.content == "User is looking at the screen"
        assert obs.source == "perception"


class TestMemorySchemas:
    def test_episodic_memory(self):
        uid = uuid4()
        sid = uuid4()
        mem = EpisodicMemory(
            user_id=uid,
            session_id=sid,
            content="User smiled when greeted",
        )
        assert mem.user_id == uid
        assert mem.salience == 0.5
        assert mem.status == MemoryStatus.ACTIVE
        assert mem.memory_type == MemoryType.EPISODIC

    def test_semantic_fact(self):
        uid = uuid4()
        fact = SemanticFact(
            user_id=uid,
            content="User prefers morning greetings",
        )
        assert fact.memory_type == MemoryType.SEMANTIC
        assert fact.confidence == 0.5

    def test_intervention_memory(self):
        uid = uuid4()
        sid = uuid4()
        intervention = InterventionMemory(
            user_id=uid,
            session_id=sid,
            strategy="empathize",
            content="I can see this might be a difficult moment.",
        )
        assert intervention.memory_type == MemoryType.INTERVENTION

    def test_retrieval_log(self):
        uid = uuid4()
        log = RetrievalLog(user_id=uid, query="recent interactions")
        assert log.results_count == 0
        assert log.retrieval_method == "sqlite"

    def test_memory_query(self):
        uid = uuid4()
        query = MemoryQuery(user_id=uid, query="how are they doing")
        assert MemoryType.EPISODIC in query.memory_types
        assert MemoryType.SEMANTIC in query.memory_types
        assert query.limit == 10

    def test_memory_query_result(self):
        uid = uuid4()
        result = MemoryQueryResult(user_id=uid, query="test")
        assert result.total_count == 0
        assert result.episodic_memories == []

    def test_project_state(self):
        state = ProjectState(
            iteration="001",
            branch_name="bootstrap/iteration-001",
            status="in_progress",
        )
        assert state.iteration == "001"
        assert state.tests_passing is False


class TestResponseSchemas:
    def test_supportive_response(self):
        sid = uuid4()
        response = SupportiveResponse(
            session_id=sid,
            strategy=ResponseStrategy.GREET,
            message="Hello! How are you?",
        )
        assert response.strategy == ResponseStrategy.GREET
        assert response.confidence == 1.0

    def test_respond_request(self):
        sid = uuid4()
        req = RespondRequest(session_id=sid)
        assert req.context == ""


class TestEnums:
    def test_all_session_statuses(self):
        assert len(SessionStatus) == 3

    def test_all_engagement_levels(self):
        assert len(EngagementLevel) == 4

    def test_all_emotion_labels(self):
        assert len(EmotionLabel) == 7

    def test_all_memory_types(self):
        assert len(MemoryType) == 4

    def test_all_response_strategies(self):
        assert len(ResponseStrategy) == 7
