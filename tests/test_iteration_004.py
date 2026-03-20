"""Tests for Iteration 004: Memory Recall & Consolidation."""

from uuid import uuid4

import pytest

from baymax.config.settings import BaymaxSettings
from baymax.core.enums import (
    CorrectionAction,
    EngagementLevel,
    FactStatus,
    MemoryType,
    PostureLabel,
    TurnRole,
)
from baymax.memory.consolidate import (
    consolidate_episodic_to_semantic,
    consolidate_session,
    merge_duplicate_facts,
)
from baymax.memory.embedding import StubTextEmbedder
from baymax.memory.store_sqlite import SQLiteMetadataStore
from baymax.memory.vector_store import StubVectorStore
from baymax.schemas.memory import (
    ChatTurn,
    ConsolidationResult,
    EpisodicMemory,
    MemoryCorrectionRequest,
    MemoryCorrectionResult,
    MemoryEmbeddingRecord,
    MemoryHit,
    MemorySummaryResponse,
    SemanticFact,
    SessionSummary,
)
from baymax.schemas.perception import RecognitionObservation
from baymax.schemas.response import SupportiveResponse
from baymax.schemas.session import Session
from baymax.schemas.user import UserProfile

# ---- Helpers ----


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "test_iter4.db")


@pytest.fixture
async def store(tmp_db):
    s = SQLiteMetadataStore(db_path=tmp_db)
    await s.initialize()
    yield s
    await s.close()


@pytest.fixture
async def user_and_session(store):
    user = UserProfile(display_name="TestUser")
    user = await store.create_user(user)
    session = Session(user_id=user.id)
    session = await store.create_session(session)
    return user, session


# ---- T401: pyproject / settings ----


class TestSettings:
    def test_new_settings_have_defaults(self):
        settings = BaymaxSettings(
            _env_file=None,
            data_dir="/tmp/test",
            cache_dir="/tmp/test",
            model_dir="/tmp/test",
        )
        assert settings.text_embedding_model == "all-MiniLM-L6-v2"
        assert settings.text_embedding_dim == 384
        assert settings.memory_top_k == 5
        assert settings.memory_similarity_threshold == 0.4
        assert settings.session_smoothing_window == 5
        assert settings.enable_memory_consolidation is True
        assert settings.vector_backend == "lancedb"


# ---- T402: Text embedder + Vector store ----


class TestStubTextEmbedder:
    def test_embed_single_returns_correct_dim(self):
        embedder = StubTextEmbedder(dimension=384)
        vec = embedder.embed_single("hello world")
        assert len(vec) == 384

    def test_embed_batch_returns_correct_count(self):
        embedder = StubTextEmbedder(dimension=128)
        vecs = embedder.embed(["a", "b", "c"])
        assert len(vecs) == 3
        assert all(len(v) == 128 for v in vecs)

    def test_dimension_property(self):
        embedder = StubTextEmbedder(dimension=256)
        assert embedder.dimension == 256

    def test_embeddings_are_normalized(self):
        import numpy as np

        embedder = StubTextEmbedder(dimension=64)
        vec = embedder.embed_single("test")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5


class TestStubVectorStore:
    @pytest.mark.asyncio
    async def test_add_and_search(self):
        vs = StubVectorStore()
        mid = uuid4()
        await vs.add(mid, [0.1, 0.2, 0.3], {"user_id": "u1", "content": "hello"})
        results = await vs.search([0.1, 0.2, 0.3], top_k=5)
        assert len(results) == 1
        assert results[0]["memory_id"] == str(mid)

    @pytest.mark.asyncio
    async def test_search_empty(self):
        vs = StubVectorStore()
        results = await vs.search([0.1, 0.2], top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_delete(self):
        vs = StubVectorStore()
        mid = uuid4()
        await vs.add(mid, [1.0, 0.0], {"user_id": "u1"})
        await vs.delete(mid)
        results = await vs.search([1.0, 0.0], top_k=5)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_filter_by_user(self):
        vs = StubVectorStore()
        uid1 = uuid4()
        uid2 = uuid4()
        await vs.add(uuid4(), [1.0, 0.0], {"user_id": str(uid1)})
        await vs.add(uuid4(), [0.0, 1.0], {"user_id": str(uid2)})
        results = await vs.search([1.0, 0.0], top_k=5, filter_user_id=uid1)
        assert len(results) == 1
        assert results[0]["metadata"]["user_id"] == str(uid1)


# ---- T403: Chat turns ----


class TestChatTurns:
    @pytest.mark.asyncio
    async def test_store_and_retrieve_turn(self, store, user_and_session):
        user, session = user_and_session
        turn = ChatTurn(
            session_id=session.id,
            user_id=user.id,
            role=TurnRole.USER,
            text="Hello, how are you?",
        )
        stored = await store.store_chat_turn(turn)
        assert stored.text == "Hello, how are you?"
        assert stored.role == TurnRole.USER

    @pytest.mark.asyncio
    async def test_get_turns_ordered(self, store, user_and_session):
        user, session = user_and_session
        t1 = ChatTurn(
            session_id=session.id, user_id=user.id,
            role=TurnRole.USER, text="Hi",
        )
        t2 = ChatTurn(
            session_id=session.id, user_id=user.id,
            role=TurnRole.SYSTEM, text="Hello!",
        )
        await store.store_chat_turn(t1)
        await store.store_chat_turn(t2)
        turns = await store.get_chat_turns(session.id)
        assert len(turns) == 2
        assert turns[0].role == TurnRole.USER
        assert turns[1].role == TurnRole.SYSTEM

    @pytest.mark.asyncio
    async def test_empty_turns(self, store, user_and_session):
        _, session = user_and_session
        turns = await store.get_chat_turns(session.id)
        assert turns == []

    def test_chat_turn_schema(self):
        turn = ChatTurn(
            session_id=uuid4(),
            user_id=uuid4(),
            role=TurnRole.USER,
            text="test message",
        )
        assert turn.role == TurnRole.USER
        assert turn.text == "test message"

    def test_chat_turn_source_round_trip_sqlite(self, tmp_path):
        import asyncio

        async def _run() -> None:
            db_path = str(tmp_path / "source_roundtrip.db")
            store = SQLiteMetadataStore(db_path=db_path)
            await store.initialize()

            user = await store.create_user(UserProfile(display_name="RoundTripUser"))
            session = await store.create_session(Session(user_id=user.id))

            stored = await store.store_chat_turn(ChatTurn(
                session_id=session.id,
                user_id=user.id,
                role=TurnRole.USER,
                text="spoken turn",
                source="speech",
            ))

            turns = await store.get_chat_turns(session.id)
            assert stored.source == "speech"
            assert turns[0].source == "speech"

            await store.close()

        asyncio.run(_run())


# ---- T404: Session consolidation ----


class TestConsolidation:
    @pytest.mark.asyncio
    async def test_consolidate_with_turns(self, store, user_and_session):
        user, session = user_and_session
        await store.store_chat_turn(ChatTurn(
            session_id=session.id, user_id=user.id,
            role=TurnRole.USER, text="I like morning walks",
        ))
        await store.store_chat_turn(ChatTurn(
            session_id=session.id, user_id=user.id,
            role=TurnRole.SYSTEM, text="That sounds nice!",
        ))
        result = await consolidate_session(store, session.id, user.id)
        assert isinstance(result, ConsolidationResult)
        assert result.episodic_memories_created >= 1
        assert result.summary is not None
        assert result.summary.turn_count == 2

    @pytest.mark.asyncio
    async def test_consolidate_empty_session(self, store, user_and_session):
        user, session = user_and_session
        result = await consolidate_session(store, session.id, user.id)
        assert result.episodic_memories_created == 0
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_consolidate_with_observations(self, store, user_and_session):
        user, session = user_and_session
        obs = RecognitionObservation(
            session_id=session.id,
            user_id=user.id,
            event_type="first_recognition",
            content="Recognized TestUser for the first time.",
            confidence=0.9,
        )
        await store.store_observation(obs)
        result = await consolidate_session(store, session.id, user.id)
        assert result.episodic_memories_created >= 1

    @pytest.mark.asyncio
    async def test_semantic_extraction_from_preference(self, store, user_and_session):
        user, session = user_and_session
        await store.store_chat_turn(ChatTurn(
            session_id=session.id, user_id=user.id,
            role=TurnRole.USER, text="I prefer to study in the evening",
        ))
        result = await consolidate_session(store, session.id, user.id)
        assert result.semantic_facts_created >= 1

    @pytest.mark.asyncio
    async def test_session_summary_stored(self, store, user_and_session):
        user, session = user_and_session
        await store.store_chat_turn(ChatTurn(
            session_id=session.id, user_id=user.id,
            role=TurnRole.USER, text="Hello",
        ))
        await consolidate_session(store, session.id, user.id)
        summaries = await store.get_session_summaries(user.id)
        assert len(summaries) == 1
        assert summaries[0].session_id == session.id


class TestConsolidateEpisodicToSemantic:
    @pytest.mark.asyncio
    async def test_empty_input(self):
        result = await consolidate_episodic_to_semantic([])
        assert result == []

    @pytest.mark.asyncio
    async def test_below_threshold(self):
        uid = uuid4()
        sid = uuid4()
        memories = [
            EpisodicMemory(user_id=uid, session_id=sid, content="mem1"),
        ]
        result = await consolidate_episodic_to_semantic(memories)
        assert result == []

    @pytest.mark.asyncio
    async def test_above_threshold(self):
        uid = uuid4()
        sid = uuid4()
        memories = [
            EpisodicMemory(user_id=uid, session_id=sid, content=f"mem{i}")
            for i in range(5)
        ]
        result = await consolidate_episodic_to_semantic(memories)
        assert len(result) >= 1
        assert result[0].user_id == uid


class TestMergeDuplicateFacts:
    @pytest.mark.asyncio
    async def test_dedup_by_content(self):
        uid = uuid4()
        facts = [
            SemanticFact(user_id=uid, content="User likes walks", confidence=0.4),
            SemanticFact(user_id=uid, content="User likes walks", confidence=0.6),
        ]
        merged = await merge_duplicate_facts(facts)
        assert len(merged) == 1
        assert merged[0].confidence == 0.6

    @pytest.mark.asyncio
    async def test_no_duplicates(self):
        uid = uuid4()
        facts = [
            SemanticFact(user_id=uid, content="fact A"),
            SemanticFact(user_id=uid, content="fact B"),
        ]
        merged = await merge_duplicate_facts(facts)
        assert len(merged) == 2


# ---- T405: Temporal smoothing ----


class TestTemporalSmoothing:
    def test_smoothing_small_window(self):
        from baymax.orchestrator.service import Orchestrator
        orch = Orchestrator.__new__(Orchestrator)
        orch._smoothing_window = 3
        orch._engagement_buffer = {}
        sid = uuid4()
        # First call: not enough data, return raw
        result = orch._smoothed_engagement(sid, EngagementLevel.HIGH)
        assert result == EngagementLevel.HIGH

    def test_smoothing_majority_wins(self):
        from baymax.orchestrator.service import Orchestrator
        orch = Orchestrator.__new__(Orchestrator)
        orch._smoothing_window = 3
        orch._engagement_buffer = {}
        sid = uuid4()
        orch._smoothed_engagement(sid, EngagementLevel.HIGH)
        orch._smoothed_engagement(sid, EngagementLevel.HIGH)
        result = orch._smoothed_engagement(sid, EngagementLevel.LOW)
        assert result == EngagementLevel.HIGH

    def test_posture_smoothing(self):
        from baymax.orchestrator.service import Orchestrator
        orch = Orchestrator.__new__(Orchestrator)
        orch._smoothing_window = 3
        orch._posture_buffer = {}
        sid = uuid4()
        orch._smoothed_posture(sid, PostureLabel.UPRIGHT)
        orch._smoothed_posture(sid, PostureLabel.UPRIGHT)
        result = orch._smoothed_posture(sid, PostureLabel.SLOUCHED)
        assert result == PostureLabel.UPRIGHT


# ---- T406: Memory-aware response ----


class TestMemoryAwareResponse:
    def test_supportive_response_has_memory_refs(self):
        resp = SupportiveResponse(
            session_id=uuid4(),
            strategy="greet",
            message="Hello!",
            memory_refs=["User likes walks"],
            state_summary={"engagement": "high"},
        )
        assert resp.memory_refs == ["User likes walks"]
        assert resp.state_summary["engagement"] == "high"

    def test_supportive_response_defaults_empty(self):
        resp = SupportiveResponse(
            session_id=uuid4(),
            strategy="greet",
            message="Hi",
        )
        assert resp.memory_refs == []
        assert resp.state_summary == {}


# ---- T407: Memory inspection and correction ----


class TestMemoryCorrection:
    @pytest.mark.asyncio
    async def test_get_semantic_fact(self, store, user_and_session):
        user, _ = user_and_session
        fact = SemanticFact(user_id=user.id, content="User likes coffee")
        stored = await store.store_semantic_fact(fact)
        retrieved = await store.get_semantic_fact(stored.id)
        assert retrieved is not None
        assert retrieved.content == "User likes coffee"

    @pytest.mark.asyncio
    async def test_get_nonexistent_fact(self, store):
        result = await store.get_semantic_fact(uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_update_semantic_fact(self, store, user_and_session):
        user, _ = user_and_session
        fact = SemanticFact(user_id=user.id, content="original")
        stored = await store.store_semantic_fact(fact)
        stored.content = "updated"
        stored.confidence = 0.9
        updated = await store.update_semantic_fact(stored)
        assert updated.content == "updated"
        # Verify via re-fetch
        refetched = await store.get_semantic_fact(stored.id)
        assert refetched.content == "updated"
        assert refetched.confidence == 0.9

    @pytest.mark.asyncio
    async def test_count_episodic(self, store, user_and_session):
        user, session = user_and_session
        await store.store_episodic_memory(EpisodicMemory(
            user_id=user.id, session_id=session.id, content="test",
        ))
        count = await store.count_episodic_memories(user.id)
        assert count == 1

    @pytest.mark.asyncio
    async def test_count_semantic(self, store, user_and_session):
        user, _ = user_and_session
        await store.store_semantic_fact(SemanticFact(
            user_id=user.id, content="fact",
        ))
        count = await store.count_semantic_facts(user.id)
        assert count == 1

    def test_correction_request_schema(self):
        req = MemoryCorrectionRequest(
            fact_id=uuid4(),
            action=CorrectionAction.CONFIRM,
        )
        assert req.action == CorrectionAction.CONFIRM

    def test_correction_result_schema(self):
        result = MemoryCorrectionResult(
            fact_id=uuid4(),
            action=CorrectionAction.UPDATE,
            previous_content="old",
            new_content="new",
            new_status=FactStatus.CONFIRMED,
        )
        assert result.new_content == "new"

    def test_memory_summary_response_schema(self):
        resp = MemorySummaryResponse(user_id=uuid4())
        assert resp.episodic_count == 0
        assert resp.semantic_count == 0


# ---- Schema tests ----


class TestNewSchemas:
    def test_session_summary_schema(self):
        ss = SessionSummary(
            session_id=uuid4(),
            user_id=uuid4(),
            summary_text="Test summary",
            turn_count=3,
            observation_count=2,
        )
        assert ss.turn_count == 3

    def test_memory_embedding_record_schema(self):
        rec = MemoryEmbeddingRecord(
            memory_id=uuid4(),
            memory_type=MemoryType.EPISODIC,
            user_id=uuid4(),
            content="test",
            embedding=[0.1, 0.2],
        )
        assert len(rec.embedding) == 2

    def test_memory_hit_schema(self):
        hit = MemoryHit(
            memory_id=uuid4(),
            memory_type=MemoryType.SEMANTIC,
            content="User prefers tea",
            score=0.85,
            user_id=uuid4(),
        )
        assert hit.score == 0.85

    def test_consolidation_result_schema(self):
        r = ConsolidationResult(
            session_id=uuid4(),
            user_id=uuid4(),
            episodic_memories_created=2,
            semantic_facts_created=1,
        )
        assert r.episodic_memories_created == 2

    def test_new_enums(self):
        assert TurnRole.USER == "user"
        assert TurnRole.SYSTEM == "system"
        assert FactStatus.CANDIDATE == "candidate"
        assert FactStatus.CONFIRMED == "confirmed"
        assert FactStatus.REJECTED == "rejected"
        assert CorrectionAction.CONFIRM == "confirm"
        assert CorrectionAction.REJECT == "reject"
        assert CorrectionAction.UPDATE == "update"


# ---- API endpoint tests ----


class TestAPIEndpoints:
    @pytest.fixture
    def client(self, tmp_db):
        from apps.api.main import app, orchestrator
        from httpx import ASGITransport, AsyncClient
        orchestrator.store = SQLiteMetadataStore(db_path=tmp_db)

        async def lifespan_override(a):
            await orchestrator.store.initialize()
            yield
            await orchestrator.store.close()

        app.router.lifespan_context = lifespan_override
        return AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        )

    @pytest.mark.asyncio
    async def test_add_turn_endpoint(self, client):
        async with client as c:
            # Create user and session first
            resp = await c.post("/v1/users", json={"display_name": "TurnTest"})
            user_id = resp.json()["id"]
            resp = await c.post("/v1/sessions", json={"user_id": user_id})
            session_id = resp.json()["id"]

            # Add a turn
            resp = await c.post(
                f"/v1/sessions/{session_id}/turns",
                json={"role": "user", "text": "Hello there!", "user_id": user_id},
            )
            assert resp.status_code == 201
            data = resp.json()
            assert data["text"] == "Hello there!"
            assert data["role"] == "user"

    @pytest.mark.asyncio
    async def test_get_turns_endpoint(self, client):
        async with client as c:
            resp = await c.post("/v1/users", json={"display_name": "TurnGet"})
            user_id = resp.json()["id"]
            resp = await c.post("/v1/sessions", json={"user_id": user_id})
            session_id = resp.json()["id"]

            await c.post(
                f"/v1/sessions/{session_id}/turns",
                json={"role": "user", "text": "msg1"},
            )
            await c.post(
                f"/v1/sessions/{session_id}/turns",
                json={"role": "system", "text": "msg2"},
            )
            resp = await c.get(f"/v1/sessions/{session_id}/turns")
            assert resp.status_code == 200
            turns = resp.json()
            assert len(turns) == 2

    @pytest.mark.asyncio
    async def test_consolidate_endpoint(self, client):
        async with client as c:
            resp = await c.post("/v1/users", json={"display_name": "ConsUser"})
            user_id = resp.json()["id"]
            resp = await c.post("/v1/sessions", json={"user_id": user_id})
            session_id = resp.json()["id"]

            await c.post(
                f"/v1/sessions/{session_id}/turns",
                json={"role": "user", "text": "I like reading books"},
            )
            resp = await c.post(
                f"/v1/sessions/{session_id}/consolidate",
                json={"user_id": user_id},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["session_id"] == session_id
            assert data["episodic_memories_created"] >= 1

    @pytest.mark.asyncio
    async def test_memory_summary_endpoint(self, client):
        async with client as c:
            resp = await c.post("/v1/users", json={"display_name": "MemSum"})
            user_id = resp.json()["id"]
            resp = await c.get(f"/v1/memory/summary/{user_id}")
            assert resp.status_code == 200
            data = resp.json()
            assert "episodic_count" in data
            assert "semantic_count" in data

    @pytest.mark.asyncio
    async def test_memory_search_endpoint(self, client):
        async with client as c:
            resp = await c.post("/v1/users", json={"display_name": "SearchUser"})
            user_id = resp.json()["id"]
            resp = await c.post(
                "/v1/memory/search",
                json={"user_id": user_id, "query": "hello", "top_k": 5},
            )
            assert resp.status_code == 200
            assert isinstance(resp.json(), list)

    @pytest.mark.asyncio
    async def test_memory_correct_endpoint_not_found(self, client):
        async with client as c:
            resp = await c.post(
                "/v1/memory/correct",
                json={
                    "fact_id": str(uuid4()),
                    "action": "confirm",
                },
            )
            assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_respond_includes_memory_refs(self, client):
        async with client as c:
            resp = await c.post("/v1/users", json={"display_name": "RecallUser"})
            user_id = resp.json()["id"]
            resp = await c.post("/v1/sessions", json={"user_id": user_id})
            session_id = resp.json()["id"]

            resp = await c.post("/v1/respond", json={
                "session_id": session_id,
                "user_id": user_id,
                "context": "checking in",
            })
            assert resp.status_code == 200
            data = resp.json()
            assert "memory_refs" in data
            assert "state_summary" in data

    @pytest.mark.asyncio
    async def test_healthz(self, client):
        async with client as c:
            resp = await c.get("/healthz")
            assert resp.status_code == 200
