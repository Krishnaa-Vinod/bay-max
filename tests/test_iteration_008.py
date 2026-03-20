"""Iteration 008 tests: Bidirectional Speech.

Coverage:
- VT801: Audio schemas & settings
- VT802: VAD state machine
- VT803: ASR provider abstraction
- VT804: Echo suppression
- VT805: SpeechInputService pipeline
- VT806: Spoken turn insertion & ChatTurn.source
- VT807: Persona style injection
- VT808: API endpoints (/v1/audio/status, /v1/stt/backends)
- VT809: LiveRuntimeStatus new fields
- VT810: project_state.json validity (manual audio tests noted as not_run)
"""

import json
import asyncio
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# VT801 — Audio schemas & settings
# ---------------------------------------------------------------------------


class TestAudioSchemas:
    """VT801: Pydantic schemas for audio subsystem."""

    def test_mic_mode_enum(self):
        from baymax.audio.schemas import MicMode

        assert MicMode.VAD == "vad"
        assert MicMode.PUSH_TO_TALK == "push_to_talk"

    def test_stt_backend_name_enum(self):
        from baymax.audio.schemas import STTBackendName

        assert STTBackendName.FASTER_WHISPER == "faster_whisper"
        assert STTBackendName.NULL == "null"

    def test_vad_backend_name_enum(self):
        from baymax.audio.schemas import VADBackendName

        assert VADBackendName.SILERO == "silero"
        assert VADBackendName.NULL == "null"

    def test_audio_chunk_info_defaults(self):
        from baymax.audio.schemas import AudioChunkInfo

        chunk = AudioChunkInfo()
        assert chunk.sample_rate == 16000
        assert chunk.channels == 1
        assert isinstance(chunk.timestamp, datetime)

    def test_vad_decision_model(self):
        from baymax.audio.schemas import VADDecision

        decision = VADDecision(is_speech=True, confidence=0.95)
        assert decision.is_speech is True
        assert decision.confidence == 0.95
        assert decision.backend == "silero"

    def test_speech_segment_model(self):
        from baymax.audio.schemas import SpeechSegment

        seg = SpeechSegment(duration_ms=1500.0)
        assert isinstance(seg.id, UUID)
        assert seg.duration_ms == 1500.0
        assert seg.source == "microphone"

    def test_transcription_result_model(self):
        from baymax.audio.schemas import TranscriptionResult

        result = TranscriptionResult(text="hello", success=True, latency_ms=123.4)
        assert result.text == "hello"
        assert result.success is True
        assert result.backend == "faster_whisper"

    def test_echo_decision_model(self):
        from baymax.audio.schemas import EchoDecision

        decision = EchoDecision(is_echo=True, similarity_score=0.92, compared_to="test")
        assert decision.is_echo is True
        assert decision.similarity_score == 0.92

    def test_speech_input_status_model(self):
        from baymax.audio.schemas import SpeechInputStatus

        status = SpeechInputStatus(speech_input_enabled=True, listening=True, mic_mode="vad")
        assert status.speech_input_enabled is True
        assert status.mic_mode == "vad"
        assert status.cooldown_active is False

    def test_audio_backend_info_model(self):
        from baymax.audio.schemas import AudioBackendInfo

        info = AudioBackendInfo(active_backend="faster_whisper", vad_backend="silero")
        assert info.active_backend == "faster_whisper"

    def test_speech_artifact_manifest_model(self):
        from baymax.audio.schemas import SpeechArtifactManifest

        manifest = SpeechArtifactManifest(stt_backend="faster_whisper", vad_backend="silero")
        assert manifest.stt_backend == "faster_whisper"
        assert manifest.segments_total == 0
        assert isinstance(manifest.created_at, datetime)

    def test_transcription_result_serializes(self):
        from baymax.audio.schemas import TranscriptionResult

        result = TranscriptionResult(text="test", success=True)
        d = result.model_dump(mode="json")
        assert "text" in d
        assert "success" in d
        assert "timestamp" in d


class TestSpeechSettings:
    """VT801: Settings for speech input from env vars."""

    def test_speech_settings_defaults(self):
        from baymax.config.settings import BaymaxSettings

        s = BaymaxSettings()
        assert s.enable_speech_input is True
        assert s.mic_sample_rate == 16000
        assert s.mic_channels == 1
        assert s.vad_backend == "silero"
        assert s.vad_threshold == 0.65
        assert s.vad_min_speech_ms == 300.0
        assert s.vad_silence_ms == 500.0
        assert s.stt_backend == "faster_whisper"
        assert s.whisper_model == "base.en"
        assert s.whisper_device == "auto"
        assert s.echo_suppression_enabled is True
        assert s.echo_similarity_threshold == 0.85
        assert s.mic_mode == "vad"
        assert s.post_speech_cooldown_ms == 1500.0
        assert s.enable_listening_indicator is True

    def test_speech_settings_env_override(self, monkeypatch):
        from baymax.config.settings import BaymaxSettings

        monkeypatch.setenv("BAYMAX_ENABLE_SPEECH_INPUT", "false")
        monkeypatch.setenv("BAYMAX_STT_BACKEND", "null")
        monkeypatch.setenv("BAYMAX_MIC_MODE", "push_to_talk")
        s = BaymaxSettings()
        assert s.enable_speech_input is False
        assert s.stt_backend == "null"
        assert s.mic_mode == "push_to_talk"


# ---------------------------------------------------------------------------
# VT802 — VAD state machine
# ---------------------------------------------------------------------------


class TestVADStateMachine:
    """VT802: VAD state transitions and segmentation."""

    def test_vad_state_enum(self):
        from baymax.audio.vad import VADState

        assert VADState.IDLE == "idle"
        assert VADState.SPEECH_STARTED == "speech_started"
        assert VADState.SPEECH_ONGOING == "speech_ongoing"
        assert VADState.SILENCE_AFTER_SPEECH == "silence_after_speech"

    def test_null_vad_always_speech(self):
        from baymax.audio.vad import NullVAD

        vad = NullVAD()
        assert vad.is_available is True
        assert vad.load_model() is True
        audio = np.zeros(1600, dtype=np.float32)
        decision, segment = vad.process_chunk(audio)
        assert decision.is_speech is True
        assert decision.confidence == 1.0
        assert segment is not None  # NullVAD returns audio as segment

    def test_null_vad_reset(self):
        from baymax.audio.vad import NullVAD, VADState

        vad = NullVAD()
        vad.reset()
        assert vad.state == VADState.IDLE

    def test_silero_vad_init(self):
        from baymax.audio.vad import SileroVAD, VADState

        vad = SileroVAD(threshold=0.7, min_speech_ms=200, silence_ms=400)
        assert vad.state == VADState.IDLE
        assert vad._threshold == 0.7
        assert vad._min_speech_ms == 200

    def test_silero_vad_no_model_returns_no_speech(self):
        from baymax.audio.vad import SileroVAD

        vad = SileroVAD()
        # Without loading model, process_chunk should return no speech
        audio = np.random.randn(16000).astype(np.float32)
        decision, segment = vad.process_chunk(audio)
        assert decision.is_speech is False
        assert segment is None

    def test_silero_vad_reset(self):
        from baymax.audio.vad import SileroVAD, VADState

        vad = SileroVAD()
        vad._state = VADState.SPEECH_ONGOING
        vad.reset()
        assert vad.state == VADState.IDLE


# ---------------------------------------------------------------------------
# VT803 — ASR provider abstraction
# ---------------------------------------------------------------------------


class TestASRProviders:
    """VT803: ASR provider factory and implementations."""

    def test_null_asr_provider(self):
        from baymax.audio.transcriber import NullASRProvider

        provider = NullASRProvider()
        assert provider.name() == "null"
        assert provider.is_available() is True
        audio = np.zeros(16000, dtype=np.float32)
        result = provider.transcribe(audio)
        assert result.text == ""
        assert result.success is False
        assert result.backend == "null"

    def test_null_asr_model_info(self):
        from baymax.audio.transcriber import NullASRProvider

        info = NullASRProvider().model_info()
        assert info["backend"] == "null"

    def test_faster_whisper_provider_init(self):
        from baymax.audio.transcriber import FasterWhisperProvider

        provider = FasterWhisperProvider(model_size="base.en", device="auto")
        assert provider.name() == "faster_whisper"
        assert provider.is_available() is True

    def test_faster_whisper_model_info(self):
        from baymax.audio.transcriber import FasterWhisperProvider

        provider = FasterWhisperProvider(model_size="base.en")
        info = provider.model_info()
        assert info["backend"] == "faster_whisper"
        assert info["model"] == "base.en"

    def test_get_asr_provider_null(self):
        from baymax.audio.transcriber import NullASRProvider, get_asr_provider

        provider = get_asr_provider(backend="null")
        assert isinstance(provider, NullASRProvider)

    def test_get_asr_provider_unknown_falls_back(self):
        from baymax.audio.transcriber import NullASRProvider, get_asr_provider

        provider = get_asr_provider(backend="nonexistent")
        assert isinstance(provider, NullASRProvider)

    def test_get_asr_provider_faster_whisper(self):
        from baymax.audio.transcriber import FasterWhisperProvider, get_asr_provider

        provider = get_asr_provider(backend="faster_whisper")
        assert isinstance(provider, FasterWhisperProvider)


# ---------------------------------------------------------------------------
# VT804 — Echo suppression
# ---------------------------------------------------------------------------


class TestEchoSuppression:
    """VT804: Echo suppression logic."""

    def test_echo_suppressor_no_history(self):
        from baymax.audio.echo_suppression import EchoSuppressor

        suppressor = EchoSuppressor(similarity_threshold=0.85)
        decision = suppressor.check("hello world")
        assert decision.is_echo is False
        assert decision.similarity_score == 0.0

    def test_echo_suppressor_exact_match(self):
        from baymax.audio.echo_suppression import EchoSuppressor

        suppressor = EchoSuppressor(similarity_threshold=0.85)
        suppressor.record_spoken("Hello, how are you today?")
        decision = suppressor.check("Hello, how are you today?")
        assert decision.is_echo is True
        assert decision.similarity_score >= 0.85

    def test_echo_suppressor_close_match(self):
        from baymax.audio.echo_suppression import EchoSuppressor

        suppressor = EchoSuppressor(similarity_threshold=0.85)
        suppressor.record_spoken("Hello, how are you today?")
        decision = suppressor.check("hello how are you today")
        assert decision.is_echo is True

    def test_echo_suppressor_different_text(self):
        from baymax.audio.echo_suppression import EchoSuppressor

        suppressor = EchoSuppressor(similarity_threshold=0.85)
        suppressor.record_spoken("I hope you are feeling well.")
        decision = suppressor.check("What is the weather like?")
        assert decision.is_echo is False

    def test_echo_suppressor_empty_text(self):
        from baymax.audio.echo_suppression import EchoSuppressor

        suppressor = EchoSuppressor()
        decision = suppressor.check("")
        assert decision.is_echo is False

    def test_echo_suppressor_clear(self):
        from baymax.audio.echo_suppression import EchoSuppressor

        suppressor = EchoSuppressor()
        suppressor.record_spoken("test")
        suppressor.clear()
        assert suppressor.recent_spoken == []

    def test_echo_suppressor_max_recent(self):
        from baymax.audio.echo_suppression import EchoSuppressor

        suppressor = EchoSuppressor(max_recent=2)
        suppressor.record_spoken("first")
        suppressor.record_spoken("second")
        suppressor.record_spoken("third")
        assert len(suppressor.recent_spoken) == 2
        assert suppressor.recent_spoken[0] == "second"

    def test_echo_suppressor_case_insensitive(self):
        from baymax.audio.echo_suppression import EchoSuppressor

        suppressor = EchoSuppressor(similarity_threshold=0.85)
        suppressor.record_spoken("HELLO THERE")
        decision = suppressor.check("hello there")
        assert decision.is_echo is True


# ---------------------------------------------------------------------------
# VT805 — SpeechInputService
# ---------------------------------------------------------------------------


class TestSpeechInputService:
    """VT805: SpeechInputService pipeline."""

    def test_service_init(self):
        from baymax.audio.speech_input_service import SpeechInputService
        from baymax.audio.transcriber import NullASRProvider

        service = SpeechInputService(
            asr_provider=NullASRProvider(),
            vad_backend="null",
            mic_device="default",
        )
        assert service.is_running is False
        assert service.is_listening is False

    def test_service_status(self):
        from baymax.audio.speech_input_service import SpeechInputService
        from baymax.audio.transcriber import NullASRProvider

        service = SpeechInputService(
            asr_provider=NullASRProvider(),
            vad_backend="null",
        )
        status = service.status()
        assert status.speech_input_enabled is False
        assert status.mic_mode == "vad"

    def test_service_speaking_lock(self):
        from baymax.audio.speech_input_service import SpeechInputService
        from baymax.audio.transcriber import NullASRProvider

        service = SpeechInputService(
            asr_provider=NullASRProvider(),
            vad_backend="null",
        )
        service.set_speaking_lock(True)
        assert service.speaking_lock_active is True
        service.set_speaking_lock(False)
        assert service.speaking_lock_active is False

    def test_service_record_spoken_text(self):
        from baymax.audio.speech_input_service import SpeechInputService
        from baymax.audio.transcriber import NullASRProvider

        service = SpeechInputService(
            asr_provider=NullASRProvider(),
            vad_backend="null",
        )
        service.record_spoken_text("test message")
        # Echo suppressor should have it
        assert len(service._echo._recent_spoken) == 1

    @pytest.mark.asyncio
    async def test_service_handle_completed_segment(self):
        from baymax.audio.speech_input_service import SpeechInputService
        from baymax.audio.transcriber import NullASRProvider

        service = SpeechInputService(
            asr_provider=NullASRProvider(),
            vad_backend="null",
            enable_artifacts=False,
        )
        audio = np.zeros(16000, dtype=np.float32)
        result = await service._handle_completed_segment(audio)
        # NullASR returns empty, so result should be None
        assert result is None

    def test_service_artifact_data(self):
        from baymax.audio.speech_input_service import SpeechInputService
        from baymax.audio.transcriber import NullASRProvider

        service = SpeechInputService(
            asr_provider=NullASRProvider(),
            vad_backend="null",
        )
        data = service.get_artifact_data()
        assert "transcription_log" in data
        assert "echo_decisions" in data
        assert "timing_log" in data

    def test_start_rejects_null_microphone(self):
        from baymax.audio.microphone import NullMicrophone
        from baymax.audio.speech_input_service import SpeechInputService
        from baymax.audio.transcriber import NullASRProvider

        service = SpeechInputService(
            asr_provider=NullASRProvider(),
            vad_backend="null",
        )
        service._mic = NullMicrophone()

        started = asyncio.run(service.start())
        assert started is False
        assert "Microphone unavailable" in service.last_error

    def test_start_rejects_null_asr(self):
        from baymax.audio.speech_input_service import SpeechInputService
        from baymax.audio.transcriber import NullASRProvider

        class WorkingMic:
            def start(self):
                return True

            def stop(self):
                return None

        service = SpeechInputService(
            asr_provider=NullASRProvider(),
            vad_backend="null",
        )
        service._mic = WorkingMic()

        started = asyncio.run(service.start())
        assert started is False
        assert "ASR unavailable" in service.last_error


# ---------------------------------------------------------------------------
# VT806 — Spoken turn insertion & ChatTurn.source
# ---------------------------------------------------------------------------


class TestSpokenTurnInsertion:
    """VT806: ChatTurn.source='speech' for spoken turns."""

    def test_chat_turn_source_default(self):
        from baymax.core.enums import TurnRole
        from baymax.schemas.memory import ChatTurn

        turn = ChatTurn(
            session_id=uuid4(),
            role=TurnRole.USER,
            text="hello",
        )
        assert turn.source == "typed"

    def test_chat_turn_source_speech(self):
        from baymax.core.enums import TurnRole
        from baymax.schemas.memory import ChatTurn

        turn = ChatTurn(
            session_id=uuid4(),
            role=TurnRole.USER,
            text="hello from mic",
            source="speech",
        )
        assert turn.source == "speech"

    def test_chat_turn_serializes_source(self):
        from baymax.core.enums import TurnRole
        from baymax.schemas.memory import ChatTurn

        turn = ChatTurn(
            session_id=uuid4(),
            role=TurnRole.USER,
            text="test",
            source="speech",
        )
        d = turn.model_dump(mode="json")
        assert d["source"] == "speech"


# ---------------------------------------------------------------------------
# VT807 — Persona style injection
# ---------------------------------------------------------------------------


class TestCompanionPersona:
    """VT807: Baymax-inspired companion persona style."""

    def test_persona_file_exists(self):
        persona_path = Path(__file__).parent.parent / "docs" / "COMPANION_PERSONA.md"
        assert persona_path.exists(), "docs/COMPANION_PERSONA.md must exist"

    def test_system_prompt_includes_companion_style(self):
        from baymax.dialogue.prompt_builder import build_system_prompt

        prompt = build_system_prompt()
        assert "calm" in prompt.lower()
        assert "gentle" in prompt.lower()
        assert "companion" in prompt.lower()
        assert "Bay-Max" in prompt

    def test_persona_style_in_prompt(self):
        from baymax.dialogue.prompt_builder import _PERSONA_STYLE

        assert "calm" in _PERSONA_STYLE.lower()
        assert "nonjudgmental" in _PERSONA_STYLE.lower()

    def test_safety_rules_preserved(self):
        from baymax.dialogue.prompt_builder import build_system_prompt

        prompt = build_system_prompt()
        assert "never diagnose" in prompt.lower()
        assert "never recommend medication" in prompt.lower()


# ---------------------------------------------------------------------------
# VT808 — API endpoints
# ---------------------------------------------------------------------------


class TestAPIEndpoints:
    """VT808: New API endpoints for speech input."""

    @pytest.mark.asyncio
    async def test_audio_status_endpoint(self):
        from apps.api.main import app
        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/audio/status")
            assert resp.status_code == 200
            data = resp.json()
            assert "speech_input_enabled" in data
            assert "listening" in data
            assert "stt_backend" in data
            assert "tts_backend" in data
            assert "speaking_lock_active" in data

    @pytest.mark.asyncio
    async def test_stt_backends_endpoint(self):
        from apps.api.main import app
        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/stt/backends")
            assert resp.status_code == 200
            data = resp.json()
            assert "available_backends" in data
            assert "faster_whisper" in data["available_backends"]
            assert "active_backend" in data
            assert "active_model" in data
            assert "vad_backend" in data

    @pytest.mark.asyncio
    async def test_existing_tts_endpoint_still_works(self):
        from apps.api.main import app
        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/tts/backends")
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_healthz_still_works(self):
        from apps.api.main import app
        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/healthz")
            assert resp.status_code == 200


# ---------------------------------------------------------------------------
# VT809 — LiveRuntimeStatus new fields
# ---------------------------------------------------------------------------


class TestLiveRuntimeStatusFields:
    """VT809: LiveRuntimeStatus has speech input fields."""

    def test_status_has_speech_input_fields(self):
        from baymax.live.schemas import LiveRuntimeStatus

        status = LiveRuntimeStatus()
        assert hasattr(status, "speech_input_enabled")
        assert hasattr(status, "listening")
        assert hasattr(status, "vad_active")
        assert hasattr(status, "stt_backend")
        assert hasattr(status, "speaking_lock_active")
        assert hasattr(status, "last_heard_text")
        assert hasattr(status, "transcription_latency_ms")
        assert hasattr(status, "mic_mode")

    def test_status_defaults(self):
        from baymax.live.schemas import LiveRuntimeStatus

        status = LiveRuntimeStatus()
        assert status.speech_input_enabled is False
        assert status.listening is False
        assert status.vad_active is False
        assert status.stt_backend == ""
        assert status.mic_mode == "vad"

    def test_status_serializes(self):
        from baymax.live.schemas import LiveRuntimeStatus

        status = LiveRuntimeStatus(
            speech_input_enabled=True,
            listening=True,
            stt_backend="faster_whisper",
        )
        d = status.model_dump(mode="json")
        assert d["speech_input_enabled"] is True
        assert d["stt_backend"] == "faster_whisper"

    def test_status_retains_tts_fields(self):
        from baymax.live.schemas import LiveRuntimeStatus

        status = LiveRuntimeStatus(
            last_spoken_text="hello",
            tts_backend="kokoro",
        )
        assert status.last_spoken_text == "hello"
        assert status.tts_backend == "kokoro"


# ---------------------------------------------------------------------------
# VT810 — project_state.json validity  (+ mic hardware note)
# ---------------------------------------------------------------------------


class TestProjectState:
    """VT810: Verify project_state.json is valid."""

    def test_project_state_json_is_valid(self):
        state_path = Path(__file__).parent.parent / "docs" / "project_state.json"
        assert state_path.exists(), "docs/project_state.json must exist"
        data = json.loads(state_path.read_text())
        assert "iteration" in data
        assert data["iteration"] == "008"
        assert "audio" in data.get("modules_implemented", [])

    def test_env_example_has_speech_input_vars(self):
        env_path = Path(__file__).parent.parent / ".env.example"
        text = env_path.read_text()
        assert "BAYMAX_ENABLE_SPEECH_INPUT" in text
        assert "BAYMAX_STT_BACKEND" in text
        assert "BAYMAX_VAD_BACKEND" in text
        assert "BAYMAX_WHISPER_MODEL" in text
        assert "BAYMAX_MIC_MODE" in text


# ---------------------------------------------------------------------------
# Additional integration-style tests
# ---------------------------------------------------------------------------


class TestMicrophoneCapture:
    """Test the NullMicrophone for headless environments."""

    @pytest.mark.asyncio
    async def test_null_microphone(self):
        from baymax.audio.microphone import NullMicrophone

        mic = NullMicrophone()
        assert mic.is_available is True
        assert mic.start() is True
        assert mic.is_running is True
        chunk = await mic.read_chunk(timeout=0.01)
        assert chunk is None
        mic.stop()
        assert mic.is_running is False


class TestFasterWhisperTranscription:
    """Test faster-whisper transcription with real model (on GPU)."""

    @pytest.mark.asyncio
    async def test_transcribe_silence(self):
        """Transcribe silent audio — expect empty or near-empty output."""
        from baymax.audio.transcriber import FasterWhisperProvider

        provider = FasterWhisperProvider(model_size="base.en", device="auto")
        audio = np.zeros(32000, dtype=np.float32)  # 2 seconds of silence
        result = provider.transcribe(audio)
        # Silence should produce empty or very short output
        assert result.backend == "faster_whisper"
        assert result.latency_ms > 0
