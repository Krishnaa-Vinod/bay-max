"""Pydantic v2 schemas for the TTS module."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class TTSBackendName(StrEnum):
    """Available TTS backend identifiers."""

    NULL = "null"
    KOKORO = "kokoro"
    PIPER = "piper"


class TTSBackendInfo(BaseModel):
    """Information about a TTS backend."""

    backend: str
    status: str = "available"
    voice: str = ""
    sample_rate: int = 24000
    notes: str = ""


class SpeechSynthesisResult(BaseModel):
    """Result from a TTS synthesis call."""

    success: bool
    wav_path: str | None = None
    backend: str = ""
    voice: str = ""
    sample_rate: int = 24000
    duration_sec: float = 0.0
    text: str = ""
    error: str | None = None
    synthesized_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )


class SpeechEvent(BaseModel):
    """A logged speech event for artifact tracking."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
    event_type: str = "speech_synthesized"
    text: str = ""
    backend: str = ""
    voice: str = ""
    wav_path: str | None = None
    playback_started_at: datetime | None = None
    playback_finished_at: datetime | None = None
    success: bool = True
    error: str | None = None
    session_id: str | None = None
    trigger_event: str | None = None


class SpeechQueueStatus(BaseModel):
    """Current state of the speech playback queue."""

    queue_depth: int = 0
    is_playing: bool = False
    last_spoken_text: str = ""
    last_spoken_at: datetime | None = None
    total_synthesized: int = 0
    total_played: int = 0
    total_errors: int = 0


class TTSSmokeTestResult(BaseModel):
    """Result from a TTS smoke test."""

    backend: str
    voice: str
    success: bool
    wav_path: str | None = None
    duration_sec: float = 0.0
    sample_rate: int = 0
    error: str | None = None
    tested_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )


class LiveVerificationRun(BaseModel):
    """Metadata for a local verification run."""

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
    ended_at: datetime | None = None
    webcam_used: bool = False
    tts_backend: str = ""
    tts_voice: str = ""
    audio_playback_tested: bool = False
    frames_processed: int = 0
    events_detected: int = 0
    responses_generated: int = 0
    speech_synthesized: int = 0
    speech_played: int = 0
    artifacts_dir: str = ""
    notes: str = ""
