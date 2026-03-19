"""SpeechInputService: orchestrates mic -> VAD -> ASR -> echo check pipeline."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from datetime import UTC, datetime

import numpy as np

from baymax.audio.echo_suppression import EchoSuppressor
from baymax.audio.microphone import MicrophoneCapture, NullMicrophone
from baymax.audio.schemas import (
    MicMode,
    SpeechInputStatus,
    TranscriptionResult,
)
from baymax.audio.transcriber import ASRProvider, get_asr_provider
from baymax.audio.vad import NullVAD, SileroVAD

logger = logging.getLogger(__name__)


class SpeechInputService:
    """Manages the full speech input pipeline.

    Microphone -> VAD segmentation -> ASR transcription -> Echo suppression.

    The service runs as an async task that can be started/stopped alongside
    the live runtime. When a valid transcription is produced, it is placed
    into an output queue for the calling code to consume.
    """

    def __init__(
        self,
        asr_provider: ASRProvider | None = None,
        sample_rate: int = 16000,
        channels: int = 1,
        mic_device: str = "default",
        vad_backend: str = "silero",
        vad_threshold: float = 0.65,
        vad_min_speech_ms: float = 300.0,
        vad_silence_ms: float = 500.0,
        stt_backend: str = "faster_whisper",
        stt_model: str = "base.en",
        stt_device: str = "auto",
        echo_threshold: float = 0.85,
        mic_mode: str = "vad",
        post_speech_cooldown_ms: float = 1500.0,
        artifact_dir: str = "./artifacts/audio",
        enable_artifacts: bool = True,
    ) -> None:
        self._sample_rate = sample_rate
        self._mic_mode = MicMode(mic_mode)
        self._post_speech_cooldown_ms = post_speech_cooldown_ms
        self._artifact_dir = artifact_dir
        self._enable_artifacts = enable_artifacts

        # Microphone
        self._mic: MicrophoneCapture | NullMicrophone
        try:
            self._mic = MicrophoneCapture(
                sample_rate=sample_rate,
                channels=channels,
                device=mic_device if mic_device != "default" else None,
            )
        except Exception:
            logger.warning("MicrophoneCapture init failed, using NullMicrophone")
            self._mic = NullMicrophone(sample_rate=sample_rate)

        # VAD
        if vad_backend == "silero":
            self._vad = SileroVAD(
                threshold=vad_threshold,
                min_speech_ms=vad_min_speech_ms,
                silence_ms=vad_silence_ms,
                sample_rate=sample_rate,
            )
        else:
            self._vad = NullVAD()

        # ASR
        self._asr = asr_provider or get_asr_provider(
            backend=stt_backend,
            model=stt_model,
            device=stt_device,
        )

        # Echo suppression
        self._echo = EchoSuppressor(similarity_threshold=echo_threshold)

        # Speaking lock state
        self._speaking_lock = False
        self._cooldown_until: float = 0.0

        # Output queue
        self._output_queue: asyncio.Queue[TranscriptionResult] = asyncio.Queue(maxsize=20)

        # Artifact tracking
        self._transcription_log: list[dict] = []
        self._echo_decisions: list[dict] = []
        self._timing_log: list[dict] = []

        # State
        self._running = False
        self._listening = False
        self._last_heard_text = ""
        self._last_transcription_latency = 0.0
        self._last_error = ""

        # Push-to-talk state
        self._ptt_recording = False
        self._ptt_audio: list[np.ndarray] = []

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_listening(self) -> bool:
        return self._listening and not self._speaking_lock

    @property
    def speaking_lock_active(self) -> bool:
        return self._speaking_lock

    @property
    def last_heard_text(self) -> str:
        return self._last_heard_text

    @property
    def output_queue(self) -> asyncio.Queue[TranscriptionResult]:
        return self._output_queue

    @property
    def last_error(self) -> str:
        return self._last_error

    def set_speaking_lock(self, locked: bool) -> None:
        """Set or release the speaking lock (called by TTS/SpeechService)."""
        self._speaking_lock = locked
        if not locked:
            cooldown_sec = self._post_speech_cooldown_ms / 1000.0
            self._cooldown_until = time.time() + cooldown_sec
            logger.debug("Speaking lock released, cooldown +%.1fs", cooldown_sec)
        else:
            logger.debug("Speaking lock engaged")

    def record_spoken_text(self, text: str) -> None:
        """Record text spoken by Bay-Max for echo suppression."""
        self._echo.record_spoken(text)

    def status(self) -> SpeechInputStatus:
        """Return current status of the speech input subsystem."""
        now = time.time()
        cooldown_active = now < self._cooldown_until
        cooldown_remaining = max(0, (self._cooldown_until - now) * 1000)

        return SpeechInputStatus(
            speech_input_enabled=self._running,
            listening=self._listening and not self._speaking_lock,
            vad_active=self._vad.state.value != "idle" if hasattr(self._vad, "state") else False,
            stt_backend=self._asr.name(),
            speaking_lock_active=self._speaking_lock,
            last_heard_text=self._last_heard_text,
            transcription_latency_ms=self._last_transcription_latency,
            mic_mode=self._mic_mode.value,
            cooldown_active=cooldown_active,
            cooldown_remaining_ms=round(cooldown_remaining, 1),
        )

    async def start(self) -> bool:
        """Start the speech input pipeline. Returns True on success."""
        if self._running:
            return True

        self._last_error = ""

        # Load VAD model
        if isinstance(self._vad, SileroVAD):
            if not self._vad.load_model():
                self._last_error = "VAD model failed to load"
                logger.warning("VAD model load failed, speech input will not work")
                return False

        # Start microphone
        if not self._mic.start():
            self._last_error = "Microphone start failed (device unavailable or permission denied)"
            logger.warning("Microphone start failed, speech input will not work")
            return False

        if self._enable_artifacts:
            os.makedirs(self._artifact_dir, exist_ok=True)

        self._running = True
        self._listening = True
        logger.info("SpeechInputService started (mode=%s)", self._mic_mode.value)
        return True

    async def stop(self) -> None:
        """Stop the speech input pipeline."""
        self._running = False
        self._listening = False
        self._mic.stop()
        self._vad.reset()
        logger.info("SpeechInputService stopped")

    async def run_loop(self) -> None:
        """Main async loop: read mic -> VAD -> transcribe -> output queue.

        Should be run as an asyncio task within the live runtime.
        """
        while self._running:
            # Skip processing if speaking lock is active
            if self._speaking_lock:
                await asyncio.sleep(0.05)
                continue

            # Skip if in cooldown
            if time.time() < self._cooldown_until:
                await asyncio.sleep(0.05)
                continue

            self._listening = True

            if self._mic_mode == MicMode.VAD:
                await self._process_vad_chunk()
            else:
                # Push-to-talk handled externally
                await asyncio.sleep(0.05)

    async def _process_vad_chunk(self) -> None:
        """Read one mic chunk, process through VAD, transcribe if segment complete."""
        chunk = await self._mic.read_chunk(timeout=0.05)
        if chunk is None:
            return

        decision, completed_segment = self._vad.process_chunk(chunk)

        if completed_segment is not None:
            await self._handle_completed_segment(completed_segment)

    async def handle_ptt_audio(self, audio: np.ndarray) -> TranscriptionResult | None:
        """Handle push-to-talk audio: transcribe directly without VAD."""
        return await self._handle_completed_segment(audio)

    async def _handle_completed_segment(self, audio: np.ndarray) -> TranscriptionResult | None:
        """Transcribe a completed speech segment and check for echo."""

        # Save segment WAV if artifacts enabled
        wav_path = None
        if self._enable_artifacts:
            wav_path = self._save_segment_wav(audio)

        # Transcribe
        result = self._asr.transcribe(audio, sample_rate=self._sample_rate)
        result.wav_path = wav_path
        self._last_transcription_latency = result.latency_ms

        # Log timing
        timing_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "segment_duration_ms": round(len(audio) / self._sample_rate * 1000, 1),
            "transcription_latency_ms": result.latency_ms,
            "text": result.text,
            "success": result.success,
        }
        self._timing_log.append(timing_entry)

        if not result.success or not result.text.strip():
            self._transcription_log.append({
                "timestamp": datetime.now(UTC).isoformat(),
                "text": "",
                "success": False,
                "error": result.error,
            })
            return None

        # Echo check
        echo_decision = self._echo.check(result.text)
        self._echo_decisions.append(echo_decision.model_dump(mode="json"))

        if echo_decision.is_echo:
            logger.info("Echo rejected: '%s'", result.text[:60])
            self._transcription_log.append({
                "timestamp": datetime.now(UTC).isoformat(),
                "text": result.text,
                "success": True,
                "echo_rejected": True,
                "echo_similarity": echo_decision.similarity_score,
            })
            return None

        # Valid transcription
        self._last_heard_text = result.text
        self._transcription_log.append({
            "timestamp": datetime.now(UTC).isoformat(),
            "text": result.text,
            "success": True,
            "echo_rejected": False,
            "latency_ms": result.latency_ms,
        })

        try:
            self._output_queue.put_nowait(result)
        except asyncio.QueueFull:
            logger.warning("Output queue full, dropping transcription")

        return result

    def _save_segment_wav(self, audio: np.ndarray) -> str | None:
        """Save a speech segment as a WAV file for artifacts."""
        try:
            import soundfile as sf

            filename = f"segment_{uuid.uuid4().hex[:12]}.wav"
            path = os.path.join(self._artifact_dir, filename)
            sf.write(path, audio, self._sample_rate)
            return path
        except Exception as exc:
            logger.debug("Failed to save segment WAV: %s", exc)
            return None

    def get_artifact_data(self) -> dict:
        """Return artifact data for verification logging."""
        return {
            "transcription_log": list(self._transcription_log),
            "echo_decisions": list(self._echo_decisions),
            "timing_log": list(self._timing_log),
        }
