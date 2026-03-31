"""Pydantic schemas for audio / speech-input subsystem."""

from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MicMode(StrEnum):
    """Microphone input mode."""

    VAD = "vad"
    PUSH_TO_TALK = "push_to_talk"


class STTBackendName(StrEnum):
    """Supported speech-to-text backend names."""

    FASTER_WHISPER = "faster_whisper"
    OPENAI_WHISPER = "openai_whisper"
    NULL = "null"


class VADBackendName(StrEnum):
    """Supported VAD backend names."""

    SILERO = "silero"
    NULL = "null"


class AudioChunkInfo(BaseModel):
    """Metadata about an incoming audio chunk from the microphone."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    sample_rate: int = 16000
    channels: int = 1
    duration_ms: float = 0.0
    samples: int = 0


class VADDecision(BaseModel):
    """Decision from the voice activity detector."""

    is_speech: bool = False
    confidence: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    backend: str = "silero"


class SpeechSegment(BaseModel):
    """A completed speech segment ready for transcription."""

    id: UUID = Field(default_factory=uuid4)
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: datetime = Field(default_factory=datetime.utcnow)
    duration_ms: float = 0.0
    sample_rate: int = 16000
    wav_path: str | None = None
    source: str = "microphone"


class TranscriptionResult(BaseModel):
    """Result from speech-to-text transcription."""

    segment_id: UUID = Field(default_factory=uuid4)
    text: str = ""
    language: str = "en"
    confidence: float = 0.0
    backend: str = "faster_whisper"
    model: str = "base.en"
    latency_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    success: bool = True
    error: str | None = None
    wav_path: str | None = None
    # Optional provider-level confidence signals.
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    # Runtime-computed quality decision.
    quality_score: float = 0.0
    quality_reason: str = ""


class EchoDecision(BaseModel):
    """Decision on whether transcribed text is Bay-Max echo."""

    is_echo: bool = False
    similarity_score: float = 0.0
    compared_to: str = ""
    threshold: float = 0.85
    reason: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SpeechInputStatus(BaseModel):
    """Runtime status of the speech input subsystem."""

    speech_input_enabled: bool = False
    listening: bool = False
    vad_active: bool = False
    stt_backend: str = ""
    tts_backend: str = ""
    speaking_lock_active: bool = False
    last_heard_text: str = ""
    last_spoken_text: str = ""
    transcription_latency_ms: float = 0.0
    mic_mode: str = "vad"
    cooldown_active: bool = False
    cooldown_remaining_ms: float = 0.0


class AudioBackendInfo(BaseModel):
    """Information about available audio backends."""

    available_backends: list[str] = Field(default_factory=list)
    active_backend: str = ""
    active_model: str = ""
    vad_backend: str = ""


class SpeechArtifactManifest(BaseModel):
    """Manifest for speech verification artifacts."""

    run_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    stt_backend: str = ""
    stt_model: str = ""
    vad_backend: str = ""
    tts_backend: str = ""
    transcripts: list[dict] = Field(default_factory=list)
    echo_decisions: list[dict] = Field(default_factory=list)
    timing: list[dict] = Field(default_factory=list)
    segments_total: int = 0
    segments_transcribed: int = 0
    segments_echo_rejected: int = 0
    segments_accepted: int = 0
    notes: list[str] = Field(default_factory=list)
