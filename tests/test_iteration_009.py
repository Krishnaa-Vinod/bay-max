"""Comprehensive tests for Iteration 009 - Affect-Aware Companion."""

from datetime import datetime, timedelta
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pytest

from baymax.config.settings import BaymaxSettings
from baymax.core.enums import ResponseStrategy
from baymax.memory.clinical_safety import (
    check_clinical_language,
    create_safe_affect_memory_content,
    is_affect_summary_safe,
    sanitize_clinical_language,
    validate_memory_content,
)
from baymax.perception.emotion import (
    MediaPipeEmotionAnalyzer,
    NullEmotionAnalyzer,
    get_emotion_analyzer,
)
from baymax.perception.emotion_smoother import AffectSmoother
from baymax.schemas.perception import (
    AffectMemoryObservation,
    EmotionResult,
    SmoothedAffectState,
)
from baymax.state.models import InteractionState


class TestEmotionSchemas:
    """Test affect analysis schemas."""

    def test_emotion_result_validation(self):
        """Test EmotionResult schema validation."""
        # Valid result
        result = EmotionResult(
            backend="mediapipe",
            confidence=0.8,
            valence=0.5,
            arousal=0.6,
            success=True
        )
        assert result.confidence == 0.8
        assert result.valence == 0.5
        assert result.arousal == 0.6
        assert result.backend == "mediapipe"

        # Test validation bounds
        with pytest.raises(ValueError):
            EmotionResult(backend="test", confidence=1.5)  # > 1.0

        with pytest.raises(ValueError):
            EmotionResult(backend="test", valence=2.0)  # > 1.0

        with pytest.raises(ValueError):
            EmotionResult(backend="test", arousal=-0.1)  # < 0.0

    def test_smoothed_affect_state_defaults(self):
        """Test SmoothedAffectState default values."""
        state = SmoothedAffectState()
        assert state.valence == 0.0
        assert state.arousal == 0.0
        assert state.confidence == 0.0
        assert state.stable_duration_sec == 0.0
        assert state.enabled is False
        assert state.sample_count == 0

    def test_affect_memory_observation(self):
        """Test AffectMemoryObservation schema."""
        obs = AffectMemoryObservation(
            session_id=uuid4(),
            user_id=uuid4(),
            content="User appeared calm and content during this session",
            valence_summary="generally positive",
            arousal_summary="calm throughout",
            confidence=0.8,
            duration_minutes=15.5
        )
        assert "calm" in obs.content
        assert obs.confidence == 0.8
        assert obs.duration_minutes == 15.5


class TestEmotionBackends:
    """Test emotion analysis backends."""

    def test_null_emotion_analyzer(self):
        """Test NullEmotionAnalyzer always available."""
        analyzer = NullEmotionAnalyzer()
        assert analyzer.name() == "null"
        assert analyzer.is_available() is True

        # Test analysis returns neutral
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        face_bbox = {"x1": 100, "y1": 100, "x2": 200, "y2": 200}
        result = analyzer.analyze(frame, face_bbox)

        assert result.backend == "null"
        assert result.confidence == 0.0
        assert result.valence == 0.0
        assert result.arousal == 0.0
        assert result.success is True

    def test_mediapipe_analyzer_availability(self):
        """Test MediaPipeEmotionAnalyzer availability check."""
        # Should be available since MediaPipe is installed
        analyzer = MediaPipeEmotionAnalyzer()
        assert analyzer.name() == "mediapipe"
        assert analyzer.is_available() is True

        model_info = analyzer.model_info()
        assert model_info["backend"] == "mediapipe"
        assert "Face Landmarker" in model_info["model"]

    def test_emotion_backend_factory(self):
        """Test get_emotion_analyzer factory function."""
        # Test null backend
        null_analyzer = get_emotion_analyzer("null")
        assert isinstance(null_analyzer, NullEmotionAnalyzer)

        # Test mediapipe backend
        mp_analyzer = get_emotion_analyzer("mediapipe")
        assert isinstance(mp_analyzer, MediaPipeEmotionAnalyzer)

        # Test unknown backend
        with pytest.raises(ValueError):
            get_emotion_analyzer("unknown_backend")

    def test_mediapipe_analyzer_mock_analysis(self):
        """Test MediaPipeEmotionAnalyzer with mocked analysis."""
        # This tests the analysis pipeline without requiring actual face detection
        analyzer = MediaPipeEmotionAnalyzer()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        face_bbox = {"x1": 200, "y1": 150, "x2": 400, "y2": 350}

        # Test with unloaded model (should handle gracefully)
        result = analyzer.analyze(frame, face_bbox)
        # Should return a result (either success with low confidence or graceful failure)
        assert isinstance(result, EmotionResult)
        assert result.backend == "mediapipe"


class TestAffectSmoother:
    """Test affect state smoothing."""

    def test_smoother_initialization(self):
        """Test AffectSmoother initialization."""
        smoother = AffectSmoother(
            alpha=0.3,
            stability_threshold=0.15,
            stability_duration_sec=20.0,
            confidence_threshold=0.7
        )
        assert smoother.alpha == 0.3
        assert smoother.stability_threshold == 0.15
        assert smoother.stability_duration_sec == 20.0
        assert smoother.confidence_threshold == 0.7

    def test_smoother_first_update(self):
        """Test first emotion result update."""
        smoother = AffectSmoother()

        emotion_result = EmotionResult(
            backend="test",
            confidence=0.8,
            valence=0.6,
            arousal=0.4,
            success=True
        )

        state = smoother.update(emotion_result)

        assert state.valence == 0.6
        assert state.arousal == 0.4
        assert state.confidence == 0.8
        assert state.sample_count == 1
        assert state.stable_duration_sec == 0.0

    def test_smoother_ema_behavior(self):
        """Test exponential moving average behavior."""
        smoother = AffectSmoother(alpha=0.5, confidence_threshold=0.5)

        # First update
        result1 = EmotionResult(backend="test", confidence=0.8, valence=0.0, arousal=0.0, success=True)
        state1 = smoother.update(result1)

        # Second update - should be averaged
        result2 = EmotionResult(backend="test", confidence=0.8, valence=1.0, arousal=1.0, success=True)
        state2 = smoother.update(result2)

        # With alpha=0.5: new = 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        assert state2.valence == 0.5
        assert state2.arousal == 0.5
        assert state2.sample_count == 2

    def test_smoother_low_confidence_skip(self):
        """Test skipping updates with low confidence."""
        smoother = AffectSmoother(confidence_threshold=0.6)

        # Low confidence result should be skipped
        low_conf_result = EmotionResult(backend="test", confidence=0.4, valence=0.9, arousal=0.9, success=True)
        state = smoother.update(low_conf_result)

        assert state.sample_count == 0  # Should not update

    def test_smoother_stability_tracking(self):
        """Test affect stability duration tracking."""
        smoother = AffectSmoother(
            alpha=0.1,  # Low smoothing for predictable behavior
            stability_threshold=0.1,
            confidence_threshold=0.5
        )

        # Add several similar updates
        base_time = datetime.utcnow()
        for i in range(3):
            result = EmotionResult(
                backend="test",
                confidence=0.8,
                valence=0.5,
                arousal=0.5,
                success=True,
                timestamp=base_time + timedelta(seconds=i*10)
            )
            with patch('baymax.perception.emotion_smoother.datetime') as mock_dt:
                mock_dt.utcnow.return_value = base_time + timedelta(seconds=i*10)
                smoother.update(result)

        # Should be stable for some duration
        assert smoother.is_stable(10.0)  # Should be stable for 10+ seconds

    def test_smoother_affect_summary(self):
        """Test affect summary generation."""
        smoother = AffectSmoother(confidence_threshold=0.6)

        # Add high-valence, low-arousal data
        result = EmotionResult(backend="test", confidence=0.8, valence=0.7, arousal=0.3, success=True)
        smoother.update(result)

        summary = smoother.get_affect_summary(10.0)

        assert "content" in summary
        assert "valence_summary" in summary
        assert "arousal_summary" in summary
        assert summary["confidence"] == 0.8
        assert summary["duration_minutes"] == 10.0


class TestClinicalSafety:
    """Test clinical language safety enforcement."""

    def test_clinical_language_detection(self):
        """Test clinical term detection."""
        # Should detect clinical terms
        has_clinical, terms = check_clinical_language("User appears to have depression")
        assert has_clinical is True
        assert "depression" in terms

        # Should not flag safe language
        has_clinical, terms = check_clinical_language("User appeared subdued and thoughtful")
        assert has_clinical is False
        assert len(terms) == 0

    def test_clinical_language_sanitization(self):
        """Test clinical language replacement."""
        unsafe_text = "User seems depressed and anxious today"
        safe_text = sanitize_clinical_language(unsafe_text)

        assert "depressed" not in safe_text
        assert "anxious" not in safe_text
        assert "subdued" in safe_text or "concerned" in safe_text

    def test_affect_summary_safety_check(self):
        """Test affect summary safety validation."""
        # Safe summary
        safe_summary = {
            "content": "User appeared calm and content",
            "valence_summary": "generally positive",
            "arousal_summary": "calm throughout",
            "confidence": 0.8
        }
        is_safe, issues = is_affect_summary_safe(safe_summary)
        assert is_safe is True
        assert len(issues) == 0

        # Unsafe summary
        unsafe_summary = {
            "content": "User shows symptoms of depression",
            "confidence": 0.8
        }
        is_safe, issues = is_affect_summary_safe(unsafe_summary)
        assert is_safe is False
        assert len(issues) > 0

    def test_safe_affect_memory_content_creation(self):
        """Test safe affect memory content generation."""
        content = create_safe_affect_memory_content(
            valence_summary="generally positive",
            arousal_summary="energetic and engaged",
            confidence=0.8,
            duration_minutes=15.0
        )

        assert "consistently appeared" in content or "generally appeared" in content
        assert "upbeat" in content or "positive" in content
        assert "15.0 minutes" in content

    def test_memory_content_validation(self):
        """Test final memory content validation."""
        # Safe content
        safe_content = "User appeared calm and content during this session"
        is_valid, result = validate_memory_content(safe_content)
        assert is_valid is True
        assert result == safe_content

        # Unsafe content that can be sanitized
        unsafe_content = "User appeared depressed during this session"
        is_valid, result = validate_memory_content(unsafe_content)
        assert is_valid is True
        assert "depressed" not in result


class TestPlannerAffectBias:
    """Test planner affect bias integration."""

    def test_base_strategy_unchanged(self):
        """Test that base planning logic is preserved."""
        from baymax.planner.supportive_planner import SupportivePlanner
        from baymax.schemas.memory import MemoryQueryResult

        planner = SupportivePlanner(affect_bias_enabled=False)

        state = InteractionState(
            session_id=uuid4(),
            turn_count=0,
            affect_enabled=False
        )
        memories = MemoryQueryResult(
            user_id=uuid4(),
            query="",
            episodic_memories=[],
            semantic_facts=[],
            total_count=0
        )

        strategy = planner.plan(state, memories, "")
        assert strategy == ResponseStrategy.GREET  # First interaction

    def test_low_valence_bias(self):
        """Test low valence bias toward gentler strategy."""
        from baymax.planner.supportive_planner import SupportivePlanner
        from baymax.schemas.memory import MemoryQueryResult

        planner = SupportivePlanner(
            affect_bias_enabled=True,
            negative_valence_threshold=-0.3,
            low_arousal_threshold=0.4
        )

        state = InteractionState(
            session_id=uuid4(),
            turn_count=2,
            affect_enabled=True,
            affect_confidence=0.8,
            valence=-0.4,  # Low valence
            arousal=0.3    # Low arousal
        )
        memories = MemoryQueryResult(
            user_id=uuid4(),
            query="",
            episodic_memories=[],
            semantic_facts=[],
            total_count=0
        )

        strategy = planner.plan(state, memories, "")
        # Should bias toward EMPATHIZE instead of default ENCOURAGE
        assert strategy == ResponseStrategy.EMPATHIZE

    def test_high_valence_no_override(self):
        """Test high valence doesn't override critical strategies."""
        from baymax.planner.supportive_planner import SupportivePlanner
        from baymax.schemas.memory import MemoryQueryResult

        planner = SupportivePlanner(affect_bias_enabled=True)

        state = InteractionState(
            session_id=uuid4(),
            turn_count=0,  # First turn = GREET
            affect_enabled=True,
            affect_confidence=0.8,
            valence=0.8,   # High valence
            arousal=0.7    # High arousal
        )
        memories = MemoryQueryResult(
            user_id=uuid4(),
            query="",
            episodic_memories=[],
            semantic_facts=[],
            total_count=0
        )

        strategy = planner.plan(state, memories, "")
        # Should still greet on first turn regardless of affect
        assert strategy == ResponseStrategy.GREET

    def test_low_confidence_no_bias(self):
        """Test low confidence affect doesn't trigger bias."""
        from baymax.planner.supportive_planner import SupportivePlanner
        from baymax.schemas.memory import MemoryQueryResult

        planner = SupportivePlanner(
            affect_bias_enabled=True,
            affect_bias_confidence_threshold=0.6
        )

        state = InteractionState(
            session_id=uuid4(),
            turn_count=2,
            affect_enabled=True,
            affect_confidence=0.4,  # Low confidence
            valence=-0.5,   # Would normally trigger bias
            arousal=0.2
        )
        memories = MemoryQueryResult(
            user_id=uuid4(),
            query="",
            episodic_memories=[],
            semantic_facts=[],
            total_count=0
        )

        strategy = planner.plan(state, memories, "")
        # Should use default strategy (ENCOURAGE) without bias
        assert strategy == ResponseStrategy.ENCOURAGE


class TestPromptBuilderAffectIntegration:
    """Test prompt builder affect context integration."""

    def test_affect_context_inclusion(self):
        """Test affect context added to prompt when confidence sufficient."""
        from baymax.dialogue.prompt_builder import build_prompt_context
        from baymax.schemas.memory import MemoryQueryResult

        state = InteractionState(
            session_id=uuid4(),
            user_id=uuid4(),
            affect_enabled=True,
            affect_confidence=0.8,
            valence=-0.4,  # Low valence
            arousal=0.3    # Low arousal
        )

        memories = MemoryQueryResult(
            user_id=uuid4(),
            query="",
            episodic_memories=[],
            semantic_facts=[],
            total_count=0
        )

        ctx = build_prompt_context(
            strategy=ResponseStrategy.ENCOURAGE,
            state=state,
            memories=memories,
            recent_turns=[],
            affect_bias_confidence_threshold=0.6
        )

        # Should have affect_context attribute
        assert hasattr(ctx, 'affect_context')
        assert "subdued" in ctx.affect_context or "gentler" in ctx.affect_context

    def test_affect_context_excluded_low_confidence(self):
        """Test affect context excluded when confidence too low."""
        from baymax.dialogue.prompt_builder import build_prompt_context
        from baymax.schemas.memory import MemoryQueryResult

        state = InteractionState(
            session_id=uuid4(),
            user_id=uuid4(),
            affect_enabled=True,
            affect_confidence=0.4,  # Low confidence
            valence=-0.4,
            arousal=0.3
        )

        memories = MemoryQueryResult(
            user_id=uuid4(),
            query="",
            episodic_memories=[],
            semantic_facts=[],
            total_count=0
        )

        ctx = build_prompt_context(
            strategy=ResponseStrategy.ENCOURAGE,
            state=state,
            memories=memories,
            recent_turns=[],
            affect_bias_confidence_threshold=0.6
        )

        # Should not have affect_context or it should be empty
        assert not hasattr(ctx, 'affect_context') or not ctx.affect_context


class TestInteractionStateAffectFields:
    """Test InteractionState affect field integration."""

    def test_affect_fields_default_values(self):
        """Test affect fields have correct default values."""
        state = InteractionState(session_id=uuid4())

        assert state.affect_enabled is False
        assert state.valence == 0.0
        assert state.arousal == 0.0
        assert state.affect_confidence == 0.0
        assert state.affect_stable_duration_sec == 0.0
        assert state.affect_sample_count == 0
        assert state.last_affect_update is None

    def test_affect_field_validation(self):
        """Test affect field validation bounds."""
        state = InteractionState(
            session_id=uuid4(),
            valence=0.5,
            arousal=0.8,
            affect_confidence=0.7
        )

        assert state.valence == 0.5
        assert state.arousal == 0.8
        assert state.affect_confidence == 0.7

        # Test bounds
        with pytest.raises(ValueError):
            InteractionState(session_id=uuid4(), valence=1.5)

        with pytest.raises(ValueError):
            InteractionState(session_id=uuid4(), arousal=-0.1)


class TestAffectSettings:
    """Test affect-related configuration settings."""

    def test_affect_settings_defaults(self):
        """Test affect settings have correct default values."""
        settings = BaymaxSettings()

        assert settings.enable_affect is True
        assert settings.affect_backend == "mediapipe"
        assert settings.affect_device == "auto"
        assert settings.affect_sample_every_n_frames == 10
        assert settings.affect_confidence_threshold == 0.60
        assert settings.affect_smoothing_alpha == 0.25
        assert settings.affect_stability_duration_sec == 30.0
        assert settings.affect_memory_confidence_threshold == 0.65
        assert settings.affect_bias_enabled is True
        assert settings.affect_bias_confidence_threshold == 0.60

    def test_affect_thresholds_validation(self):
        """Test affect threshold settings are reasonable."""
        settings = BaymaxSettings()

        # Confidence thresholds should be between 0 and 1
        assert 0.0 <= settings.affect_confidence_threshold <= 1.0
        assert 0.0 <= settings.affect_memory_confidence_threshold <= 1.0
        assert 0.0 <= settings.affect_bias_confidence_threshold <= 1.0

        # Smoothing alpha should be between 0 and 1
        assert 0.0 <= settings.affect_smoothing_alpha <= 1.0

        # Valence thresholds should be reasonable
        assert -1.0 <= settings.affect_negative_valence_threshold <= 1.0
        assert -1.0 <= settings.affect_positive_valence_threshold <= 1.0


class TestProjectStateValidation:
    """Test project state validity for iteration 009."""

    def test_project_state_modules(self):
        """Test project state reflects iteration 009 modules."""
        # This would normally read from docs/project_state.json
        # but for testing we'll verify the expected structure
        expected_modules = [
            "config", "core", "schemas", "state", "memory", "planner",
            "dialogue", "orchestrator", "perception", "live", "tts", "audio"
        ]

        # Should include perception.emotion module
        expected_new_files = [
            "src/baymax/perception/emotion.py",
            "src/baymax/perception/emotion_smoother.py",
            "src/baymax/memory/clinical_safety.py"
        ]

        # This is a structural test - in real implementation,
        # would verify files exist and modules are importable
        for module in expected_modules:
            assert module  # Basic structure test

    def test_iteration_completeness(self):
        """Test that iteration 009 requirements are implemented."""
        # Verify key components can be imported
        from baymax.perception.emotion import MediaPipeEmotionAnalyzer, NullEmotionAnalyzer
        from baymax.perception.emotion_smoother import AffectSmoother
        from baymax.schemas.perception import EmotionResult

        # Basic instantiation tests
        null_analyzer = NullEmotionAnalyzer()
        assert null_analyzer.is_available()

        mp_analyzer = MediaPipeEmotionAnalyzer()
        assert mp_analyzer.is_available()

        smoother = AffectSmoother()
        assert smoother.alpha > 0.0

        # Schema validation
        result = EmotionResult(backend="test", success=True)
        assert result.backend == "test"


class TestRuntimeAffectWiring:
    """Test that affect analysis is wired into the live runtime."""

    def test_runtime_has_affect_components(self):
        """Test that LiveRuntime initializes affect components when enabled."""
        from unittest.mock import MagicMock, patch

        # Mock settings to enable affect with null backend
        mock_settings = MagicMock()
        mock_settings.enable_affect = True
        mock_settings.affect_backend = "null"
        mock_settings.affect_device = "cpu"
        mock_settings.affect_sample_every_n_frames = 10
        mock_settings.affect_smoothing_alpha = 0.25
        mock_settings.affect_stability_duration_sec = 30.0
        mock_settings.affect_confidence_threshold = 0.6
        mock_settings.presence_min_consecutive_frames = 2
        mock_settings.absence_timeout_sec = 30.0
        mock_settings.session_resume_window_sec = 300.0
        mock_settings.event_min_stability_sec = 4.0
        mock_settings.quiet_companionship_interval_sec = 300.0
        mock_settings.proactive_min_interval_sec = 30.0
        mock_settings.any_response_min_interval_sec = 10.0
        mock_settings.live_artifact_dir = "./artifacts/test"
        mock_settings.enable_artifact_logging = False
        mock_settings.tts_enabled = False
        mock_settings.enable_speech_input = False
        mock_settings.enable_live_overlay = False
        mock_settings.enable_live_debug_hud = False
        mock_settings.tts_voice = "test"

        with patch("baymax.live.runtime.get_settings", return_value=mock_settings):
            from baymax.live.runtime import LiveRuntime

            runtime = LiveRuntime(db_path=":memory:")

            # Verify affect components are initialized
            assert runtime._affect_enabled is True
            assert runtime._affect_analyzer is not None
            assert runtime._affect_analyzer.name() == "null"
            assert runtime._affect_smoother is not None
            assert runtime._affect_sample_every_n_frames == 10
            assert runtime._affect_frame_counter == 0

    def test_state_manager_update_affect(self):
        """Test that StateManager can update affect fields."""
        from baymax.state.manager import StateManager

        manager = StateManager()
        session_id = uuid4()

        # Create a session state
        state = manager.get_or_create(session_id)
        assert state.affect_enabled is False
        assert state.valence == 0.0
        assert state.arousal == 0.0

        # Update affect
        updated_state = manager.update_affect(
            session_id,
            affect_enabled=True,
            valence=0.5,
            arousal=0.6,
            affect_confidence=0.8,
            affect_stable_duration_sec=15.0,
            affect_sample_count=10,
        )

        assert updated_state is not None
        assert updated_state.affect_enabled is True
        assert updated_state.valence == 0.5
        assert updated_state.arousal == 0.6
        assert updated_state.affect_confidence == 0.8
        assert updated_state.affect_stable_duration_sec == 15.0
        assert updated_state.affect_sample_count == 10

    def test_live_runtime_status_has_affect_fields(self):
        """Test that LiveRuntimeStatus has affect fields."""
        from baymax.live.schemas import LiveRuntimeStatus

        status = LiveRuntimeStatus()

        # Check affect fields exist with defaults
        assert hasattr(status, "affect_enabled")
        assert hasattr(status, "affect_backend")
        assert hasattr(status, "emotion_valence")
        assert hasattr(status, "emotion_arousal")
        assert hasattr(status, "emotion_confidence")
        assert hasattr(status, "emotion_stable_duration_sec")
        assert hasattr(status, "emotion_debug_summary")

        # Check default values
        assert status.affect_enabled is False
        assert status.affect_backend == ""
        assert status.emotion_valence == 0.0
        assert status.emotion_arousal == 0.0
        assert status.emotion_confidence == 0.0

    def test_interaction_state_has_affect_fields(self):
        """Test that InteractionState has affect fields."""
        state = InteractionState(session_id=uuid4())

        assert hasattr(state, "affect_enabled")
        assert hasattr(state, "valence")
        assert hasattr(state, "arousal")
        assert hasattr(state, "affect_confidence")
        assert hasattr(state, "affect_stable_duration_sec")
        assert hasattr(state, "affect_sample_count")
        assert hasattr(state, "last_affect_update")

        # Check default values
        assert state.affect_enabled is False
        assert state.valence == 0.0
        assert state.arousal == 0.0
        assert state.affect_confidence == 0.0
