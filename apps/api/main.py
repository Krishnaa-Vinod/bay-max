"""Bay-Max FastAPI application."""

import io
from contextlib import asynccontextmanager
from uuid import UUID

import numpy as np

# Import live UI routers
from apps.api import live_http, live_ws
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

from baymax.config.settings import get_settings
from baymax.core.enums import TurnRole
from baymax.live.schemas import LiveRuntimeStatus
from baymax.orchestrator.service import Orchestrator
from baymax.schemas.memory import (
    ChatTurn,
    ConsolidationResult,
    MemoryCorrectionRequest,
    MemoryCorrectionResult,
    MemoryHit,
    MemoryQuery,
    MemoryQueryResult,
    MemorySummaryResponse,
)
from baymax.schemas.perception import FrameAnalysisResult
from baymax.schemas.response import (
    DialogueBackendInfo,
    RespondRequest,
    SupportiveResponse,
)
from baymax.schemas.session import Session, SessionCreate
from baymax.schemas.user import FaceEnrollment, UserProfile, UserProfileCreate
from baymax.state.models import InteractionState

orchestrator = Orchestrator()

# Live runtime reference (set externally when live mode is active)
_live_runtime = None


def get_active_orchestrator() -> Orchestrator:
    """Return the orchestrator currently backing API requests.

    When local live runtime is active, API and live telemetry must share the same
    orchestrator instance and DB state.
    """
    if _live_runtime is not None:
        return _live_runtime._orch
    return orchestrator


def set_live_runtime(runtime):
    """Set the live runtime reference for status queries."""
    global _live_runtime
    _live_runtime = runtime
    # Also set in the live modules
    live_ws.set_live_runtime(runtime)
    live_http.set_live_runtime(runtime)
    if runtime is not None:
        live_http.set_orchestrator(runtime._orch)
    else:
        live_http.set_orchestrator(orchestrator)


@asynccontextmanager
async def lifespan(app: FastAPI):
    active_orchestrator = get_active_orchestrator()
    await active_orchestrator.initialize()
    # Keep live HTTP orchestrator aligned with the active runtime when present.
    if _live_runtime is not None:
        live_http.set_orchestrator(_live_runtime._orch)
    else:
        live_http.set_orchestrator(orchestrator)
    yield
    await active_orchestrator.shutdown()


app = FastAPI(
    title="Bay-Max API",
    description="Memory-first empathetic companion agent",
    version="0.10.0",
    lifespan=lifespan,
)

# Add CORS middleware for the companion UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include live UI routers
app.include_router(live_ws.router)
app.include_router(live_http.router)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Basic health check."""
    return {"status": "ok"}


# --- Users ---


@app.post("/v1/users", response_model=UserProfile, status_code=201)
async def create_user(request: UserProfileCreate) -> UserProfile:
    """Create a new user profile."""
    return await get_active_orchestrator().create_user(request)


@app.get("/v1/users", response_model=list[UserProfile])
async def list_users() -> list[UserProfile]:
    """List all active users."""
    return await get_active_orchestrator().list_users()


@app.get("/v1/users/{user_id}", response_model=UserProfile)
async def get_user(user_id: UUID) -> UserProfile:
    """Fetch a user profile."""
    user = await get_active_orchestrator().get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


# --- Face Enrollment ---


@app.post("/v1/users/{user_id}/enroll/face", response_model=FaceEnrollment, status_code=201)
async def enroll_face(user_id: UUID, file: UploadFile = File(...)) -> FaceEnrollment:
    """Enroll a face from an uploaded image.

    Expects a single image file (JPEG/PNG). The image must contain exactly one face.
    """
    orch = get_active_orchestrator()
    user = await orch.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    contents = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        frame = np.array(pil_image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    return await orch.enroll_face(user_id, frame)


# --- Sessions ---


@app.post("/v1/sessions", response_model=Session, status_code=201)
async def create_session(request: SessionCreate) -> Session:
    """Create a new interaction session."""
    return await get_active_orchestrator().create_session(user_id=request.user_id)


@app.get("/v1/sessions/{session_id}/state", response_model=InteractionState)
async def get_session_state(session_id: UUID) -> InteractionState:
    """Get the current interaction state for a session."""
    state = await get_active_orchestrator().get_session_state(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return state


# --- Frame Analysis ---


@app.post("/v1/sessions/{session_id}/frames", response_model=FrameAnalysisResult)
async def analyze_frame(session_id: UUID, file: UploadFile = File(...)) -> FrameAnalysisResult:
    """Submit a frame for face detection and recognition."""
    orch = get_active_orchestrator()
    session = await orch.get_session_state(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    contents = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        frame = np.array(pil_image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    return await orch.analyze_frame(session_id, frame)


# --- Memory ---


@app.post("/v1/memory/query", response_model=MemoryQueryResult)
async def query_memories(request: MemoryQuery) -> MemoryQueryResult:
    """Retrieve memories for a user."""
    return await get_active_orchestrator().query_memories(
        user_id=request.user_id,
        query=request.query,
        limit=request.limit,
    )


# --- Response ---


# --- Iteration 005: Dialogue Backends ---


@app.get("/v1/dialogue/backends", response_model=DialogueBackendInfo)
async def get_dialogue_backends() -> DialogueBackendInfo:
    """Return configured and available dialogue backends."""
    return get_active_orchestrator().get_dialogue_backends()


@app.post("/v1/respond", response_model=SupportiveResponse)
async def respond(request: RespondRequest) -> SupportiveResponse:
    """Generate a supportive personalized response."""
    return await get_active_orchestrator().respond(
        session_id=request.session_id,
        user_id=request.user_id,
        context=request.context,
    )


# --- Iteration 004: Chat Turns ---


class ChatTurnRequest(BaseModel):
    """Request to add a typed conversation turn."""

    role: TurnRole
    text: str
    user_id: UUID | None = None


@app.post(
    "/v1/sessions/{session_id}/turns",
    response_model=ChatTurn,
    status_code=201,
)
async def add_turn(session_id: UUID, request: ChatTurnRequest) -> ChatTurn:
    """Add a typed conversation turn to a session."""
    orch = get_active_orchestrator()
    session = await orch.get_session_state(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return await orch.add_turn(
        session_id=session_id,
        role=request.role,
        text=request.text,
        user_id=request.user_id,
    )


@app.get(
    "/v1/sessions/{session_id}/turns",
    response_model=list[ChatTurn],
)
async def get_turns(session_id: UUID) -> list[ChatTurn]:
    """Get all chat turns for a session."""
    return await get_active_orchestrator().get_turns(session_id)


# --- Iteration 004: Consolidation ---


class ConsolidateRequest(BaseModel):
    """Request to consolidate a session."""

    user_id: UUID


@app.post(
    "/v1/sessions/{session_id}/consolidate",
    response_model=ConsolidationResult,
)
async def consolidate_session(
    session_id: UUID, request: ConsolidateRequest
) -> ConsolidationResult:
    """Consolidate a session into episodic memories and semantic facts."""
    return await get_active_orchestrator().consolidate(
        session_id=session_id,
        user_id=request.user_id,
    )


# --- Iteration 004: Memory Summary ---


@app.get(
    "/v1/memory/summary/{user_id}",
    response_model=MemorySummaryResponse,
)
async def get_memory_summary(user_id: UUID) -> MemorySummaryResponse:
    """Get a summary of all memories for a user."""
    orch = get_active_orchestrator()
    user = await orch.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return await orch.get_memory_summary(user_id)


# --- Iteration 004: Semantic Memory Query ---


class SemanticQueryRequest(BaseModel):
    """Request for semantic memory search."""

    user_id: UUID
    query: str
    top_k: int = Field(default=5, ge=1, le=50)


@app.post(
    "/v1/memory/search",
    response_model=list[MemoryHit],
)
async def query_memory_semantic(request: SemanticQueryRequest) -> list[MemoryHit]:
    """Search memories using semantic similarity."""
    return await get_active_orchestrator().query_memory_semantic(
        user_id=request.user_id,
        query=request.query,
        top_k=request.top_k,
    )


# --- Iteration 004: Memory Correction ---


@app.post(
    "/v1/memory/correct",
    response_model=MemoryCorrectionResult,
)
async def correct_memory(request: MemoryCorrectionRequest) -> MemoryCorrectionResult:
    """Apply a correction to a semantic fact (confirm, reject, or update)."""
    try:
        return await get_active_orchestrator().correct_memory(request)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# --- Iteration 006: Live Runtime ---


@app.get("/v1/live/status", response_model=LiveRuntimeStatus)
async def get_live_status() -> LiveRuntimeStatus:
    """Return current live runtime status."""
    if _live_runtime is None:
        return LiveRuntimeStatus(live_mode_active=False)
    return _live_runtime.status


class LiveControlRequest(BaseModel):
    """Request to control live mode."""

    action: str = Field(description="Action: start | stop")
    source: str | None = Field(default=None, description="Override source path for replay")


@app.post("/v1/live/control")
async def live_control(request: LiveControlRequest) -> dict[str, str]:
    """Control live mode (start/stop)."""
    if request.action == "stop":
        if _live_runtime is not None:
            _live_runtime.stop()
            return {"status": "ok", "message": "Stop requested"}
        return {"status": "ok", "message": "Live mode not active"}
    elif request.action == "start":
        return {"status": "error", "message": "Live mode must be started via CLI runner"}
    return {"status": "error", "message": f"Unknown action: {request.action}"}


# --- Iteration 007: TTS / Spoken Output ---


class TTSBackendsResponse(BaseModel):
    """Response for TTS backends endpoint."""

    available_backends: list[str]
    active_backend: str
    active_voice: str
    audio_playback_enabled: bool
    sample_rate: int
    tts_enabled: bool


@app.get("/v1/tts/backends", response_model=TTSBackendsResponse)
async def get_tts_backends() -> TTSBackendsResponse:
    """Return available and configured TTS backends."""
    settings = get_settings()
    available = ["null", "kokoro", "piper"]
    return TTSBackendsResponse(
        available_backends=available,
        active_backend=settings.tts_backend,
        active_voice=settings.tts_voice,
        audio_playback_enabled=settings.enable_audio_playback,
        sample_rate=settings.tts_rate,
        tts_enabled=settings.tts_enabled,
    )


# --- Iteration 008: Speech Input / Bidirectional Speech ---


class AudioStatusResponse(BaseModel):
    """Response for audio status endpoint."""

    speech_input_enabled: bool = False
    listening: bool = False
    vad_active: bool = False
    stt_backend: str = ""
    tts_backend: str = ""
    speaking_lock_active: bool = False
    last_heard_text: str = ""
    last_spoken_text: str = ""
    transcription_latency_ms: float = 0.0
    # Iteration 009 — affect analysis state
    affect_enabled: bool = False
    affect_backend: str = ""
    emotion_valence: float = 0.0
    emotion_arousal: float = 0.0
    emotion_confidence: float = 0.0
    emotion_stable_duration_sec: float = 0.0
    emotion_debug_summary: str = ""


@app.get("/v1/audio/status", response_model=AudioStatusResponse)
async def get_audio_status() -> AudioStatusResponse:
    """Return current speech-input and speech-output runtime status."""
    settings = get_settings()
    response = AudioStatusResponse(
        speech_input_enabled=settings.enable_speech_input,
        tts_backend=settings.tts_backend,
        affect_enabled=settings.enable_affect,
        affect_backend=settings.affect_backend,
    )
    if _live_runtime is not None:
        status = _live_runtime.status
        response.listening = status.listening
        response.vad_active = status.vad_active
        response.stt_backend = status.stt_backend
        response.speaking_lock_active = status.speaking_lock_active
        response.last_heard_text = status.last_heard_text
        response.last_spoken_text = status.last_spoken_text
        response.transcription_latency_ms = status.transcription_latency_ms
        # Affect fields from live runtime status (iteration 009)
        response.emotion_valence = status.emotion_valence
        response.emotion_arousal = status.emotion_arousal
        response.emotion_confidence = status.emotion_confidence
        response.emotion_stable_duration_sec = status.emotion_stable_duration_sec
        response.emotion_debug_summary = status.emotion_debug_summary
    return response


class STTBackendsResponse(BaseModel):
    """Response for STT backends endpoint."""

    available_backends: list[str]
    active_backend: str
    active_model: str
    vad_backend: str


@app.get("/v1/stt/backends", response_model=STTBackendsResponse)
async def get_stt_backends() -> STTBackendsResponse:
    """Return configured and available speech-to-text backends."""
    settings = get_settings()
    available = ["faster_whisper", "null"]
    return STTBackendsResponse(
        available_backends=available,
        active_backend=settings.stt_backend,
        active_model=settings.whisper_model,
        vad_backend=settings.vad_backend,
    )


# --- Iteration 009: Affect Analysis / Facial Emotion Recognition ---


class EmotionBackendsResponse(BaseModel):
    """Response for emotion backends endpoint."""

    available_backends: list[str]
    active_backend: str
    active_model_or_runtime: str
    enabled: bool
    sample_every_n_frames: int


@app.get("/v1/emotion/backends", response_model=EmotionBackendsResponse)
async def get_emotion_backends() -> EmotionBackendsResponse:
    """Return configured and available affect analysis backends."""
    settings = get_settings()
    available = ["null", "mediapipe"]

    # Determine active model/runtime description
    if settings.affect_backend == "mediapipe":
        active_model = "MediaPipe Face Landmarker with blendshapes"
    elif settings.affect_backend == "null":
        active_model = "NullEmotionAnalyzer (testing/fallback)"
    else:
        active_model = f"{settings.affect_backend} (unknown)"

    return EmotionBackendsResponse(
        available_backends=available,
        active_backend=settings.affect_backend,
        active_model_or_runtime=active_model,
        enabled=settings.enable_affect,
        sample_every_n_frames=settings.affect_sample_every_n_frames,
    )
