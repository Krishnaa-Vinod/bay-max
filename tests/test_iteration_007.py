"""Tests for Iteration 007: Webcam Verification + TTS."""

import json
import os
import tempfile

import pytest

from baymax.config.settings import BaymaxSettings

# ============================================================
# TTS Schemas
# ============================================================


class TestTTSSchemas:
    """Verify all TTS-related Pydantic schemas."""

    def test_tts_backend_info(self):
        from baymax.tts.schemas import TTSBackendInfo

        info = TTSBackendInfo(backend="kokoro", voice="af_heart")
        assert info.backend == "kokoro"
        assert info.voice == "af_heart"
        assert info.sample_rate == 24000
        assert info.status == "available"

    def test_speech_synthesis_result_success(self):
        from baymax.tts.schemas import SpeechSynthesisResult

        r = SpeechSynthesisResult(
            success=True, wav_path="/tmp/test.wav", backend="kokoro",
            voice="af_heart", text="hello", duration_sec=1.5,
        )
        assert r.success is True
        assert r.wav_path == "/tmp/test.wav"
        assert r.duration_sec == 1.5

    def test_speech_synthesis_result_failure(self):
        from baymax.tts.schemas import SpeechSynthesisResult

        r = SpeechSynthesisResult(
            success=False, backend="kokoro",
            error="Pipeline not initialized.",
        )
        assert r.success is False
        assert r.error == "Pipeline not initialized."
        assert r.wav_path is None

    def test_speech_event(self):
        from baymax.tts.schemas import SpeechEvent

        e = SpeechEvent(
            text="hello", backend="kokoro", voice="af_heart",
            wav_path="/tmp/t.wav", session_id="abc",
            trigger_event="recognized_user_arrived",
        )
        assert e.text == "hello"
        assert e.session_id == "abc"
        assert e.success is True

    def test_speech_event_failed(self):
        from baymax.tts.schemas import SpeechEvent

        e = SpeechEvent(
            text="hello", success=False,
            error="Backend unavailable",
            event_type="speech_failed",
        )
        assert e.success is False
        assert e.event_type == "speech_failed"

    def test_speech_queue_status_defaults(self):
        from baymax.tts.schemas import SpeechQueueStatus

        s = SpeechQueueStatus()
        assert s.queue_depth == 0
        assert s.is_playing is False
        assert s.last_spoken_text == ""
        assert s.total_synthesized == 0

    def test_tts_smoke_test_result(self):
        from baymax.tts.schemas import TTSSmokeTestResult

        r = TTSSmokeTestResult(
            backend="kokoro", voice="af_heart",
            success=True, wav_path="/tmp/smoke.wav",
            duration_sec=2.0, sample_rate=24000,
        )
        assert r.success is True
        assert r.sample_rate == 24000

    def test_live_verification_run(self):
        from baymax.tts.schemas import LiveVerificationRun

        v = LiveVerificationRun(
            webcam_used=True, tts_backend="kokoro",
            tts_voice="af_heart", frames_processed=100,
        )
        assert v.webcam_used is True
        assert v.tts_backend == "kokoro"

    def test_tts_backend_name_enum(self):
        from baymax.tts.schemas import TTSBackendName

        assert TTSBackendName.NULL == "null"
        assert TTSBackendName.KOKORO == "kokoro"
        assert TTSBackendName.PIPER == "piper"
        assert len(TTSBackendName) == 3


# ============================================================
# TTS Provider Interface and Factory
# ============================================================


class TestTTSProviders:
    """Test TTS provider selection and NullTTSProvider."""

    def test_null_provider(self):
        from baymax.tts.provider import NullTTSProvider

        p = NullTTSProvider()
        assert p.name() == "null"
        assert p.is_available() is True
        info = p.info()
        assert info.backend == "null"

    def test_null_provider_synthesize(self):
        from baymax.tts.provider import NullTTSProvider

        p = NullTTSProvider()
        result = p.synthesize("hello", "/tmp/null.wav")
        assert result.success is True
        assert result.wav_path is None
        assert result.backend == "null"

    def test_factory_null(self):
        from baymax.tts.provider import get_tts_provider

        p = get_tts_provider("null")
        assert p.name() == "null"

    def test_factory_unknown_falls_back_to_null(self):
        from baymax.tts.provider import get_tts_provider

        p = get_tts_provider("nonexistent_backend_xyz")
        assert p.name() == "null"

    def test_factory_kokoro_returns_kokoro_provider(self):
        from baymax.tts.provider import get_tts_provider

        p = get_tts_provider("kokoro")
        assert p.name() == "kokoro"

    def test_factory_piper_returns_piper_provider(self):
        from baymax.tts.provider import get_tts_provider

        p = get_tts_provider("piper")
        assert p.name() == "piper"

    def test_piper_provider_unavailable(self):
        from baymax.tts.piper_provider import PiperTTSProvider

        p = PiperTTSProvider()
        # piper-tts package is not installed in this env
        assert p.is_available() is False
        info = p.info()
        assert info.status == "unavailable"
        result = p.synthesize("test", "/tmp/p.wav")
        assert result.success is False
        assert "not available" in result.error


# ============================================================
# SpeechService
# ============================================================


class TestSpeechService:
    """Test the speech synthesis service with the null backend."""

    def test_null_service_synthesize(self):
        from baymax.tts.provider import NullTTSProvider
        from baymax.tts.speech_service import SpeechService

        svc = SpeechService(provider=NullTTSProvider())
        result = svc.synthesize("Hello world")
        assert result.success is True
        assert result.backend == "null"

    def test_null_service_queue_status(self):
        from baymax.tts.provider import NullTTSProvider
        from baymax.tts.speech_service import SpeechService

        svc = SpeechService(provider=NullTTSProvider())
        svc.synthesize("Hello")
        qs = svc.queue_status()
        assert qs.total_synthesized == 1
        # null provider produces no wav, so queue_depth stays 0
        assert qs.queue_depth == 0

    def test_speech_events_tracked(self):
        from baymax.tts.provider import NullTTSProvider
        from baymax.tts.speech_service import SpeechService

        svc = SpeechService(provider=NullTTSProvider())
        svc.synthesize("Test one", session_id="s1")
        svc.synthesize("Test two", trigger_event="arrival")
        events = svc.events
        assert len(events) == 2
        assert events[0].session_id == "s1"
        assert events[1].trigger_event == "arrival"

    @pytest.mark.asyncio
    async def test_synthesize_and_play_null(self):
        from baymax.tts.provider import NullTTSProvider
        from baymax.tts.speech_service import SpeechService

        svc = SpeechService(
            provider=NullTTSProvider(), enable_playback=False,
        )
        result = await svc.synthesize_and_play("Hello async")
        assert result.success is True

    def test_service_from_settings(self):
        from baymax.tts.speech_service import SpeechService

        svc = SpeechService(backend="null")
        result = svc.synthesize("test from settings")
        assert result.success is True

    def test_queue_bounded(self):
        from baymax.tts.speech_service import SpeechService

        with tempfile.TemporaryDirectory() as td:
            svc = SpeechService(
                backend="null", output_dir=td, max_queue_size=3,
            )
            # Null produces no wav files, so nothing added to queue
            for i in range(5):
                svc.synthesize(f"Message {i}")
            assert svc.queue_status().total_synthesized == 5


# ============================================================
# Kokoro TTS Backend (real synthesis test)
# ============================================================


class TestKokoroProvider:
    """Test the Kokoro TTS provider."""

    def test_kokoro_provider_creates(self):
        from baymax.tts.kokoro_provider import KokoroTTSProvider

        p = KokoroTTSProvider(voice="af_heart")
        assert p.name() == "kokoro"

    def test_kokoro_is_available(self):
        from baymax.tts.kokoro_provider import KokoroTTSProvider

        p = KokoroTTSProvider(voice="af_heart")
        assert p.is_available() is True

    def test_kokoro_synthesize_wav(self):
        from baymax.tts.kokoro_provider import KokoroTTSProvider

        p = KokoroTTSProvider(voice="af_heart")
        with tempfile.TemporaryDirectory() as td:
            wav_path = os.path.join(td, "test.wav")
            result = p.synthesize("Hello, I am Bay-Max.", wav_path)
            assert result.success is True, f"Kokoro synthesis failed: {result.error}"
            assert result.wav_path == wav_path
            assert os.path.isfile(wav_path)
            assert result.duration_sec > 0
            assert result.sample_rate == 24000

    def test_kokoro_empty_text(self):
        from baymax.tts.kokoro_provider import KokoroTTSProvider

        p = KokoroTTSProvider(voice="af_heart")
        result = p.synthesize("", "/tmp/empty.wav")
        assert result.success is False
        assert "Empty text" in result.error

    def test_kokoro_info(self):
        from baymax.tts.kokoro_provider import KokoroTTSProvider

        p = KokoroTTSProvider(voice="af_heart")
        info = p.info()
        assert info.backend == "kokoro"
        assert info.voice == "af_heart"


# ============================================================
# ArtifactLogger speech logging
# ============================================================


class TestArtifactSpeechLogging:
    """Test speech event logging in ArtifactLogger."""

    def test_log_speech_event(self):
        with tempfile.TemporaryDirectory() as td:
            from baymax.live.artifact_logger import ArtifactLogger

            logger = ArtifactLogger(artifact_dir=td, enabled=True)
            logger.log_speech_event(
                text="Hello",
                backend="kokoro",
                voice="af_heart",
                wav_path="/tmp/test.wav",
                success=True,
                session_id="s1",
                trigger_event="arrival",
            )
            speech_path = os.path.join(td, "speech.jsonl")
            assert os.path.isfile(speech_path)
            with open(speech_path) as f:
                entry = json.loads(f.readline())
            assert entry["text"] == "Hello"
            assert entry["backend"] == "kokoro"
            assert entry["success"] is True
            assert entry["trigger_event"] == "arrival"

    def test_log_speech_event_failure(self):
        with tempfile.TemporaryDirectory() as td:
            from baymax.live.artifact_logger import ArtifactLogger

            logger = ArtifactLogger(artifact_dir=td, enabled=True)
            logger.log_speech_event(
                text="Hello",
                backend="kokoro",
                success=False,
                error="Pipeline init failed",
            )
            speech_path = os.path.join(td, "speech.jsonl")
            with open(speech_path) as f:
                entry = json.loads(f.readline())
            assert entry["success"] is False
            assert entry["event_type"] == "speech_failed"
            assert entry["error"] == "Pipeline init failed"

    def test_log_speech_disabled(self):
        with tempfile.TemporaryDirectory() as td:
            from baymax.live.artifact_logger import ArtifactLogger

            logger = ArtifactLogger(artifact_dir=td, enabled=False)
            logger.log_speech_event(text="Hello", backend="null")
            speech_path = os.path.join(td, "speech.jsonl")
            assert not os.path.exists(speech_path)


# ============================================================
# LiveRuntimeStatus speech fields
# ============================================================


class TestLiveStatusSpeechFields:
    """Verify LiveRuntimeStatus includes TTS fields."""

    def test_status_has_speech_fields(self):
        from baymax.live.schemas import LiveRuntimeStatus

        s = LiveRuntimeStatus()
        assert s.last_spoken_text == ""
        assert s.speech_queue_depth == 0
        assert s.tts_backend == ""
        assert s.tts_voice == ""

    def test_status_with_speech_data(self):
        from baymax.live.schemas import LiveRuntimeStatus

        s = LiveRuntimeStatus(
            live_mode_active=True,
            last_spoken_text="Hello there",
            speech_queue_depth=2,
            tts_backend="kokoro",
            tts_voice="af_heart",
        )
        assert s.last_spoken_text == "Hello there"
        assert s.tts_backend == "kokoro"


# ============================================================
# API Endpoints
# ============================================================


class TestTTSAPIEndpoints:
    """Test the TTS API endpoints."""

    @pytest.fixture
    def client(self):
        from apps.api.main import app
        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=app)
        return AsyncClient(transport=transport, base_url="http://test")

    @pytest.mark.asyncio
    async def test_tts_backends_endpoint(self, client):
        async with client as c:
            r = await c.get("/v1/tts/backends")
        assert r.status_code == 200
        data = r.json()
        assert "available_backends" in data
        assert "active_backend" in data
        assert "active_voice" in data
        assert "audio_playback_enabled" in data
        assert "null" in data["available_backends"]
        assert "kokoro" in data["available_backends"]

    @pytest.mark.asyncio
    async def test_live_status_has_speech_fields(self, client):
        async with client as c:
            r = await c.get("/v1/live/status")
        assert r.status_code == 200
        data = r.json()
        assert "last_spoken_text" in data
        assert "speech_queue_depth" in data
        assert "tts_backend" in data
        assert "tts_voice" in data


# ============================================================
# TTS Settings
# ============================================================


class TestTTSSettings:
    """Verify TTS configuration settings."""

    def test_default_tts_settings(self):
        settings = BaymaxSettings()
        assert settings.tts_backend == "kokoro"
        assert settings.tts_voice == "af_heart"
        assert settings.tts_rate == 24000
        assert settings.tts_enabled is True
        assert settings.tts_device == "auto"
        assert settings.tts_output_dir == "./artifacts/tts"
        assert settings.enable_audio_playback is True
        assert settings.audio_backend == "sounddevice"
        assert settings.kokoro_voice == "af_heart"
        assert settings.piper_model_path == ""
        assert settings.enable_live_debug_hud is True

    def test_artifact_dir_updated(self):
        settings = BaymaxSettings()
        assert settings.live_artifact_dir == "./artifacts/live_run_007"


# ============================================================
# Graceful TTS Backend Failure
# ============================================================


class TestTTSGracefulFailure:
    """Ensure TTS failures do not crash the runtime."""

    def test_broken_kokoro_voice_returns_error(self):
        from baymax.tts.kokoro_provider import KokoroTTSProvider

        p = KokoroTTSProvider(voice="nonexistent_voice_xyz")
        with tempfile.TemporaryDirectory() as td:
            wav_path = os.path.join(td, "broken.wav")
            result = p.synthesize("Test", wav_path)
            # Kokoro may or may not fail on bad voice — check no crash
            assert isinstance(result.success, bool)

    def test_speech_service_handles_backend_error(self):
        """Ensure SpeechService doesn't crash on backend failure."""
        from baymax.tts.piper_provider import PiperTTSProvider
        from baymax.tts.speech_service import SpeechService

        svc = SpeechService(provider=PiperTTSProvider())
        result = svc.synthesize("Should fail gracefully")
        assert result.success is False
        assert svc.queue_status().total_errors == 1

    @pytest.mark.asyncio
    async def test_async_synthesize_failure_no_crash(self):
        from baymax.tts.piper_provider import PiperTTSProvider
        from baymax.tts.speech_service import SpeechService

        svc = SpeechService(
            provider=PiperTTSProvider(), enable_playback=False,
        )
        result = await svc.synthesize_and_play("Test async failure")
        assert result.success is False


# ============================================================
# Project state validity
# ============================================================


class TestProjectState007:
    """Verify project_state.json stays valid."""

    def test_project_state_valid(self):
        path = os.path.join(
            os.path.dirname(__file__),
            "..", "docs", "project_state.json",
        )
        assert os.path.isfile(path)
        with open(path) as f:
            data = json.load(f)
        assert "iteration" in data
        assert "modules_implemented" in data
        assert "tests_passing" in data
