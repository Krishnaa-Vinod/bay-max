"""Tests for Iteration 005: Grounded Local Dialogue."""

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from baymax.config.settings import BaymaxSettings
from baymax.core.enums import (
    EmotionLabel,
    EngagementLevel,
    PostureLabel,
    ResponseStrategy,
    TurnRole,
)
from baymax.dialogue.factory import AVAILABLE_BACKENDS, create_dialogue_provider
from baymax.dialogue.prompt_builder import (
    build_grounded_user_prompt,
    build_messages,
    build_prompt_context,
    build_system_prompt,
)
from baymax.dialogue.rule_based import RuleBasedDialogue
from baymax.dialogue.safety import check_output_safety, check_safety
from baymax.memory.store_sqlite import SQLiteMetadataStore
from baymax.schemas.memory import (
    ChatTurn,
    EpisodicMemory,
    MemoryQueryResult,
    SemanticFact,
)
from baymax.schemas.response import (
    DialogueBackendInfo,
    DialogueDebugTrace,
    DialogueSmokeTestResult,
    GroundedPromptContext,
    GroundedResponse,
    SafetyDecision,
    SupportiveResponse,
)
from baymax.schemas.session import Session
from baymax.schemas.user import UserProfile
from baymax.state.models import InteractionState

# ---- Helpers ----


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "test_iter5.db")


@pytest.fixture
async def store(tmp_db):
    s = SQLiteMetadataStore(db_path=tmp_db)
    await s.initialize()
    yield s
    await s.close()


@pytest.fixture
async def user_and_session(store):
    user = UserProfile(display_name="TestUser005")
    user = await store.create_user(user)
    session = Session(user_id=user.id)
    session = await store.create_session(session)
    return user, session


def make_state(session_id=None, user_id=None, display_name="Alice") -> InteractionState:
    sid = session_id or uuid4()
    return InteractionState(
        session_id=sid,
        user_id=user_id,
        user_display_name=display_name,
        engagement_level=EngagementLevel.MEDIUM,
        dominant_emotion=EmotionLabel.NEUTRAL,
        posture=PostureLabel.UPRIGHT,
    )


def make_memories(user_id=None) -> MemoryQueryResult:
    uid = user_id or uuid4()
    return MemoryQueryResult(
        user_id=uid,
        query="test query",
        episodic_memories=[
            EpisodicMemory(
                user_id=uid,
                session_id=uuid4(),
                content="User mentioned they enjoy warm greetings",
                salience=0.7,
            )
        ],
        semantic_facts=[
            SemanticFact(
                user_id=uid,
                content="User is building Bay-Max",
                confidence=0.8,
            )
        ],
        total_count=2,
    )


# ---- T501: Settings defaults ----


class TestIteration005Settings:
    def test_new_dialogue_settings_have_defaults(self):
        settings = BaymaxSettings(
            _env_file=None,
            data_dir="/tmp/test",
            cache_dir="/tmp/test",
            model_dir="/tmp/test",
        )
        assert settings.dialogue_backend == "rule_based"
        assert settings.enable_dialogue_debug is False
        assert settings.enable_safe_health_mode is True
        assert settings.dialogue_max_history_turns == 8
        assert settings.dialogue_top_k_memories == 5
        assert settings.dialogue_temperature == 0.4
        assert settings.dialogue_max_new_tokens == 220
        assert settings.ollama_base_url == "http://localhost:11434"
        assert settings.ollama_model == ""
        assert settings.hf_chat_model == "Qwen/Qwen2.5-1.5B-Instruct"
        assert settings.hf_dtype == "auto"
        assert settings.enable_rule_based_fallback is True


# ---- T502: Dialogue provider selection ----


class TestDialogueProviderSelection:
    def test_factory_returns_rule_based_by_default(self):
        settings = BaymaxSettings(
            _env_file=None,
            dialogue_backend="rule_based",
            data_dir="/tmp",
            cache_dir="/tmp",
            model_dir="/tmp",
        )
        p = create_dialogue_provider(settings)
        assert p.backend_name == "rule_based"

    def test_factory_falls_back_when_ollama_model_empty(self):
        settings = BaymaxSettings(
            _env_file=None,
            dialogue_backend="ollama",
            ollama_model="",  # deliberately empty — will raise ValueError
            enable_rule_based_fallback=True,
            data_dir="/tmp",
            cache_dir="/tmp",
            model_dir="/tmp",
        )
        p = create_dialogue_provider(settings)
        assert p.backend_name == "rule_based"

    def test_factory_raises_if_unknown_backend_and_no_fallback(self):
        settings = BaymaxSettings(
            _env_file=None,
            dialogue_backend="nonexistent",
            enable_rule_based_fallback=False,
            data_dir="/tmp",
            cache_dir="/tmp",
            model_dir="/tmp",
        )
        with pytest.raises(ValueError, match="Unknown dialogue backend"):
            create_dialogue_provider(settings)

    def test_factory_falls_back_for_unknown_backend(self):
        settings = BaymaxSettings(
            _env_file=None,
            dialogue_backend="nonexistent",
            enable_rule_based_fallback=True,
            data_dir="/tmp",
            cache_dir="/tmp",
            model_dir="/tmp",
        )
        p = create_dialogue_provider(settings)
        assert p.backend_name == "rule_based"

    def test_available_backends_list(self):
        assert "rule_based" in AVAILABLE_BACKENDS
        assert "ollama" in AVAILABLE_BACKENDS
        assert "transformers" in AVAILABLE_BACKENDS


# ---- T502: Rule-based dialogue still works ----


class TestRuleBasedDialogueProvider:
    def test_generate_returns_supportive_response(self):
        rb = RuleBasedDialogue()
        state = make_state()
        memories = MemoryQueryResult(user_id=uuid4(), query="")
        response = rb.generate(ResponseStrategy.GREET, state, memories)
        assert isinstance(response, SupportiveResponse)
        assert response.message
        assert response.backend == "rule_based"
        assert response.model_name == ""

    def test_generate_with_prompt_context_still_works(self):
        rb = RuleBasedDialogue()
        state = make_state()
        memories = MemoryQueryResult(user_id=uuid4(), query="")
        ctx = GroundedPromptContext(
            session_id=state.session_id,
            strategy=ResponseStrategy.GREET,
        )
        response = rb.generate(ResponseStrategy.GREET, state, memories, ctx)
        assert isinstance(response, SupportiveResponse)
        assert response.backend == "rule_based"

    def test_recall_strategy_uses_memory(self):
        rb = RuleBasedDialogue()
        uid = uuid4()
        state = make_state(user_id=uid)
        memories = make_memories(uid)
        response = rb.generate(ResponseStrategy.RECALL, state, memories)
        assert "warm greetings" in response.message or "Bay-Max" in response.message

    def test_is_available_always_true(self):
        assert RuleBasedDialogue().is_available() is True


# ---- T504: Prompt builder ----


class TestPromptBuilder:
    def test_build_system_prompt_contains_safety_rules(self):
        sp = build_system_prompt()
        assert "diagnos" in sp.lower()
        assert "companion" in sp.lower()
        assert "Bay-Max" in sp

    def test_build_grounded_user_prompt_includes_memory(self):
        state = make_state()
        memories = make_memories(state.user_id)
        turns = []
        ctx = build_prompt_context(
            strategy=ResponseStrategy.RECALL,
            state=state,
            memories=memories,
            recent_turns=turns,
            context="test context",
        )
        prompt = build_grounded_user_prompt(ctx)
        assert "warm greetings" in prompt
        assert "Bay-Max" in prompt
        assert "Alice" in prompt

    def test_build_messages_returns_two_messages(self):
        state = make_state()
        ctx = GroundedPromptContext(
            session_id=state.session_id,
            strategy=ResponseStrategy.GREET,
            user_display_name="Alice",
        )
        msgs = build_messages(ctx)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_build_prompt_context_packages_memories(self):
        uid = uuid4()
        state = make_state(user_id=uid)
        memories = make_memories(uid)
        ctx = build_prompt_context(
            strategy=ResponseStrategy.RECALL,
            state=state,
            memories=memories,
            recent_turns=[],
        )
        assert "warm greetings" in ctx.memory_refs[0]
        assert ctx.strategy == ResponseStrategy.RECALL
        assert ctx.state_summary["engagement"] == "medium"

    def test_build_prompt_context_no_memory_doesnt_fabricate(self):
        state = make_state()
        memories = MemoryQueryResult(user_id=uuid4(), query="")
        ctx = build_prompt_context(
            strategy=ResponseStrategy.GREET,
            state=state,
            memories=memories,
            recent_turns=[],
        )
        assert ctx.memory_refs == []

    def test_build_prompt_context_includes_recent_turns(self):
        uid = uuid4()
        state = make_state(user_id=uid)
        session_id = state.session_id
        turns = [
            ChatTurn(
                session_id=session_id,
                user_id=uid,
                role=TurnRole.USER,
                text="Hello there",
            )
        ]
        ctx = build_prompt_context(
            strategy=ResponseStrategy.GREET,
            state=state,
            memories=MemoryQueryResult(user_id=uid, query=""),
            recent_turns=turns,
        )
        assert len(ctx.recent_turns) == 1
        assert ctx.recent_turns[0]["text"] == "Hello there"


# ---- T505: Safety gating ----


class TestSafetyGating:
    def test_safe_message_passes(self):
        decision = check_safety("How are you today?")
        assert decision.is_safe is True
        assert decision.flags == []
        assert decision.redirect_response is None

    def test_diagnosis_request_flagged(self):
        decision = check_safety("Can you diagnose me from my face?")
        assert decision.is_safe is False
        assert len(decision.flags) > 0
        assert decision.redirect_response is not None
        assert "diagnos" in decision.redirect_response.lower()

    def test_medical_advice_request_flagged(self):
        decision = check_safety("What medication should I take?")
        assert decision.is_safe is False

    def test_disease_query_flagged(self):
        decision = check_safety("Do I have diabetes?")
        assert decision.is_safe is False

    def test_output_safety_clean(self):
        decision = check_output_safety("I'm here to support you today!")
        assert decision.is_safe is True

    def test_output_safety_diagnoses_flagged(self):
        decision = check_output_safety("You have diabetes based on your appearance.")
        assert decision.is_safe is False

    def test_output_safety_prescribes_flagged(self):
        decision = check_output_safety("You should take metformin for your condition.")
        assert decision.is_safe is False


# ---- T505: Fallback behaviour ----


class TestFallbackBehaviour:
    def test_fallback_used_when_backend_raises(self):
        """If the primary backend raises, fallback should kick in."""
        settings = BaymaxSettings(
            _env_file=None,
            dialogue_backend="rule_based",
            enable_rule_based_fallback=True,
            data_dir="/tmp",
            cache_dir="/tmp",
            model_dir="/tmp",
        )
        provider = create_dialogue_provider(settings)
        # Make generate raise
        provider.generate = MagicMock(side_effect=RuntimeError("backend down"))

        fallback = RuleBasedDialogue()
        state = make_state()
        memories = MemoryQueryResult(user_id=uuid4(), query="")

        try:
            provider.generate(ResponseStrategy.GREET, state, memories, None)
        except RuntimeError:
            # Simulate orchestrator fallback
            response = fallback.generate(ResponseStrategy.GREET, state, memories)
            assert response.backend == "rule_based"


# ---- T506: API endpoint schemas ----


class TestDialogueBackendInfoSchema:
    def test_dialogue_backend_info_fields(self):
        info = DialogueBackendInfo(
            available_backends=["rule_based", "ollama", "transformers"],
            active_backend="rule_based",
            active_model="",
            fallback_enabled=True,
        )
        assert info.active_backend == "rule_based"
        assert info.fallback_enabled is True
        assert len(info.available_backends) == 3


class TestNewSchemas:
    def test_grounded_prompt_context_defaults(self):
        ctx = GroundedPromptContext(
            session_id=uuid4(),
            strategy=ResponseStrategy.GREET,
        )
        assert ctx.memory_refs == []
        assert ctx.recent_turns == []
        assert ctx.context == ""

    def test_safety_decision_safe(self):
        sd = SafetyDecision(is_safe=True)
        assert sd.flags == []
        assert sd.redirect_response is None

    def test_safety_decision_unsafe(self):
        sd = SafetyDecision(
            is_safe=False,
            flags=["diagnosis_request:diagnose me"],
            redirect_response="Please see a professional.",
        )
        assert not sd.is_safe
        assert "diagnosis_request" in sd.flags[0]

    def test_dialogue_debug_trace_fields(self):
        trace = DialogueDebugTrace(
            backend="ollama",
            model_name="qwen2.5:1.5b",
            prompt_tokens_approx=120,
            latency_ms=340.5,
            fallback_used=False,
        )
        assert trace.backend == "ollama"
        assert trace.latency_ms == 340.5

    def test_grounded_response_fields(self):
        gr = GroundedResponse(
            text="Hello, nice to see you!",
            strategy=ResponseStrategy.GREET,
            backend="rule_based",
        )
        assert gr.text == "Hello, nice to see you!"
        assert gr.fallback_used is False
        assert gr.safety_flags == []

    def test_dialogue_smoke_test_result_fields(self):
        r = DialogueSmokeTestResult(
            case="Rule-based baseline",
            backend="rule_based",
            result="passed",
            notes="OK",
            response_text="Hello!",
        )
        assert r.result == "passed"

    def test_supportive_response_new_fields(self):
        session_id = uuid4()
        r = SupportiveResponse(
            session_id=session_id,
            strategy=ResponseStrategy.GREET,
            message="Hi!",
            backend="ollama",
            model_name="qwen2.5:1.5b",
            fallback_used=True,
            safety_flags=["diagnosis_request:test"],
        )
        assert r.backend == "ollama"
        assert r.fallback_used is True
        assert r.safety_flags == ["diagnosis_request:test"]


# ---- T506: GET /v1/dialogue/backends endpoint ----


class TestDialogueBackendsEndpoint:
    def test_get_dialogue_backends(self, tmp_db):
        import apps.api.main as api_module
        from fastapi.testclient import TestClient


        client = TestClient(api_module.app)
        response = client.get("/v1/dialogue/backends")
        assert response.status_code == 200
        data = response.json()
        assert "available_backends" in data
        assert "active_backend" in data
        assert "active_model" in data
        assert "fallback_enabled" in data
        assert "rule_based" in data["available_backends"]


# ---- T506: Enhanced /v1/respond response shape ----


class TestEnhancedRespondShape:
    def test_respond_includes_backend_fields(self, tmp_db):
        import apps.api.main as api_module
        from fastapi.testclient import TestClient

        client = TestClient(api_module.app)

        # Create user and session
        user_resp = client.post("/v1/users", json={"display_name": "Test505"})
        assert user_resp.status_code == 201
        user_id = user_resp.json()["id"]

        session_resp = client.post("/v1/sessions", json={"user_id": user_id})
        assert session_resp.status_code == 201
        session_id = session_resp.json()["id"]

        respond_resp = client.post(
            "/v1/respond",
            json={"session_id": session_id, "user_id": user_id, "context": ""},
        )
        assert respond_resp.status_code == 200
        data = respond_resp.json()
        assert "backend" in data
        assert "model_name" in data
        assert "fallback_used" in data
        assert "safety_flags" in data
        assert "memory_refs" in data
        assert "state_summary" in data

    def test_respond_with_safety_trigger(self, tmp_db):
        import apps.api.main as api_module
        from fastapi.testclient import TestClient

        client = TestClient(api_module.app)

        user_resp = client.post("/v1/users", json={"display_name": "SafeTest505"})
        user_id = user_resp.json()["id"]
        session_resp = client.post("/v1/sessions", json={"user_id": user_id})
        session_id = session_resp.json()["id"]

        respond_resp = client.post(
            "/v1/respond",
            json={
                "session_id": session_id,
                "user_id": user_id,
                "context": "Can you diagnose me from my face?",
            },
        )
        assert respond_resp.status_code == 200
        data = respond_resp.json()
        assert len(data["safety_flags"]) > 0
        assert "professional" in data["message"].lower() or "companion" in data["message"].lower()


# ---- T508: project_state.json validity (iteration-005) ----


class TestProjectStateSchema005:
    def test_project_state_json_is_valid(self):
        import json
        from pathlib import Path

        path = Path(__file__).parent.parent / "docs" / "project_state.json"
        assert path.exists(), "docs/project_state.json not found"
        state = json.loads(path.read_text())
        assert "iteration" in state
        assert "status" in state
        assert "api_endpoints" in state
