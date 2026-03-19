"""Speech synthesis service with queued playback."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from collections import deque
from datetime import UTC, datetime

from baymax.tts.provider import TTSProvider, get_tts_provider
from baymax.tts.schemas import (
    SpeechEvent,
    SpeechQueueStatus,
    SpeechSynthesisResult,
)

logger = logging.getLogger(__name__)


class SpeechService:
    """Synthesizes text and manages a bounded playback queue.

    The service:
    1. Accepts text strings and synthesizes them to WAV via a TTSProvider.
    2. Optionally plays audio through speakers using sounddevice.
    3. Maintains a bounded queue so responses never pile up unboundedly.
    4. Tracks all speech events for artifact logging.
    """

    def __init__(
        self,
        provider: TTSProvider | None = None,
        backend: str = "null",
        voice: str = "",
        device: str = "auto",
        model_dir: str = "",
        sample_rate: int = 24000,
        output_dir: str = "./artifacts/tts",
        enable_playback: bool = True,
        audio_backend: str = "sounddevice",
        max_queue_size: int = 5,
    ) -> None:
        self._provider = provider or get_tts_provider(
            backend=backend,
            voice=voice,
            device=device,
            model_dir=model_dir,
            sample_rate=sample_rate,
        )
        self._output_dir = output_dir
        self._enable_playback = enable_playback
        self._audio_backend = audio_backend
        self._sample_rate = sample_rate
        self._max_queue_size = max_queue_size

        self._queue: deque[SpeechSynthesisResult] = deque(
            maxlen=max_queue_size
        )
        self._is_playing = False
        self._last_spoken_text = ""
        self._last_spoken_at: datetime | None = None
        self._total_synthesized = 0
        self._total_played = 0
        self._total_errors = 0
        self._events: list[SpeechEvent] = []
        self._playback_lock = asyncio.Lock()

    @property
    def provider(self) -> TTSProvider:
        return self._provider

    @property
    def events(self) -> list[SpeechEvent]:
        return list(self._events)

    def queue_status(self) -> SpeechQueueStatus:
        return SpeechQueueStatus(
            queue_depth=len(self._queue),
            is_playing=self._is_playing,
            last_spoken_text=self._last_spoken_text,
            last_spoken_at=self._last_spoken_at,
            total_synthesized=self._total_synthesized,
            total_played=self._total_played,
            total_errors=self._total_errors,
        )

    def synthesize(
        self,
        text: str,
        session_id: str | None = None,
        trigger_event: str | None = None,
    ) -> SpeechSynthesisResult:
        """Synthesize text to WAV. Returns the result immediately."""
        os.makedirs(self._output_dir, exist_ok=True)
        filename = f"speech_{uuid.uuid4().hex[:12]}.wav"
        output_path = os.path.join(self._output_dir, filename)

        result = self._provider.synthesize(text, output_path)
        result.synthesis_ok = result.success
        self._total_synthesized += 1

        event = SpeechEvent(
            text=text,
            backend=self._provider.name(),
            voice=result.voice,
            wav_path=result.wav_path,
            success=result.success,
            error=result.error,
            session_id=session_id,
            trigger_event=trigger_event,
        )

        if not result.success:
            self._total_errors += 1
            event.event_type = "speech_failed"

        self._events.append(event)

        if result.success and result.wav_path:
            self._queue.append(result)

        return result

    async def synthesize_and_play(
        self,
        text: str,
        session_id: str | None = None,
        trigger_event: str | None = None,
    ) -> SpeechSynthesisResult:
        """Synthesize and optionally play. Non-blocking via background."""
        result = self.synthesize(
            text, session_id=session_id, trigger_event=trigger_event
        )
        if result.success and result.wav_path and self._enable_playback:
            await self._play_audio(result)
        elif result.success and result.wav_path and not self._enable_playback:
            result.playback_ok = False
            result.playback_error = "Playback disabled by configuration"
        return result

    async def _play_audio(self, result: SpeechSynthesisResult) -> None:
        """Play a WAV file through speakers if possible."""
        async with self._playback_lock:
            self._is_playing = True
            play_event = self._events[-1] if self._events else None
            if play_event:
                play_event.playback_started_at = datetime.now(UTC)
            try:
                played = await self._do_playback(result.wav_path or "")
                if played:
                    result.playback_ok = True
                    result.playback_error = None
                    self._total_played += 1
                    self._last_spoken_text = result.text
                    self._last_spoken_at = datetime.now(UTC)
                    if play_event:
                        play_event.playback_finished_at = datetime.now(
                            UTC
                        )
                        play_event.event_type = "speech_played"
                else:
                    result.playback_ok = False
                    if not result.playback_error:
                        result.playback_error = "Playback unavailable"
            except Exception as exc:
                logger.warning("Audio playback failed: %s", exc)
                self._total_errors += 1
                result.playback_ok = False
                result.playback_error = str(exc)
            finally:
                self._is_playing = False

    async def _do_playback(self, wav_path: str) -> bool:
        """Attempt playback via sounddevice. Returns True if played."""
        if not wav_path or not os.path.isfile(wav_path):
            return False
        try:
            import sounddevice as sd
            import soundfile as sf

            data, samplerate = sf.read(wav_path)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda: sd.play(data, samplerate)
            )
            await loop.run_in_executor(None, sd.wait)
            return True
        except ImportError:
            logger.info(
                "sounddevice not available — WAV saved but not played: %s",
                wav_path,
            )
            self._last_spoken_text = self._events[-1].text if self._events else ""
            self._last_spoken_at = datetime.now(UTC)
            if self._events:
                self._events[-1].error = "sounddevice not available"
            return False
        except OSError as exc:
            logger.warning("Audio device error: %s — WAV saved: %s", exc, wav_path)
            self._last_spoken_text = self._events[-1].text if self._events else ""
            self._last_spoken_at = datetime.now(UTC)
            if self._events:
                self._events[-1].error = str(exc)
            return False

    async def drain_queue(self) -> int:
        """Play all queued items. Returns count played."""
        played = 0
        while self._queue:
            result = self._queue.popleft()
            await self._play_audio(result)
            played += 1
        return played
