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
        output_device: str | int | None = None,
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
        self._output_device = self._normalize_output_device(output_device)
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

    @staticmethod
    def _normalize_output_device(output_device: str | int | None) -> int | None:
        if output_device is None:
            return None
        if isinstance(output_device, int):
            return output_device
        text = str(output_device).strip().lower()
        if text in {"", "default", "auto", "none"}:
            return None
        try:
            return int(text)
        except ValueError:
            return None

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
            import numpy as np
            import sounddevice as sd
            import soundfile as sf

            data, samplerate = sf.read(wav_path, dtype="float32")
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(
                    None,
                    lambda: sd.play(data, samplerate, device=self._output_device),
                )
                await loop.run_in_executor(None, sd.wait)
            except Exception as exc:
                if "Invalid sample rate" not in str(exc):
                    raise

                out_device = self._output_device
                if out_device is None:
                    default_devices = sd.default.device
                    if isinstance(default_devices, (list, tuple)) and len(default_devices) >= 2:
                        out_device = default_devices[1]

                output_info = sd.query_devices(out_device, "output")
                target_rate = int(round(float(output_info.get("default_samplerate", samplerate))))
                if target_rate <= 0 or target_rate == samplerate:
                    raise

                logger.info(
                    "Resampling playback from %dHz to %dHz for output device compatibility",
                    samplerate,
                    target_rate,
                )

                src_len = int(data.shape[0]) if getattr(data, "ndim", 1) > 0 else 0
                if src_len <= 1:
                    raise

                dst_len = max(1, int(round(src_len * target_rate / samplerate)))
                src_x = np.linspace(0.0, 1.0, src_len, endpoint=False)
                dst_x = np.linspace(0.0, 1.0, dst_len, endpoint=False)

                if data.ndim == 1:
                    resampled = np.interp(dst_x, src_x, data).astype(np.float32)
                else:
                    channels = [
                        np.interp(dst_x, src_x, data[:, ch])
                        for ch in range(data.shape[1])
                    ]
                    resampled = np.stack(channels, axis=1).astype(np.float32)

                await loop.run_in_executor(
                    None,
                    lambda: sd.play(resampled, target_rate, device=out_device),
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
