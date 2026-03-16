"""Bay-Max FastAPI application."""

import io
from contextlib import asynccontextmanager
from uuid import UUID

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from baymax.orchestrator.service import Orchestrator
from baymax.schemas.memory import MemoryQuery, MemoryQueryResult
from baymax.schemas.perception import FrameAnalysisResult
from baymax.schemas.response import RespondRequest, SupportiveResponse
from baymax.schemas.session import Session, SessionCreate
from baymax.schemas.user import FaceEnrollment, UserProfile, UserProfileCreate
from baymax.state.models import InteractionState

orchestrator = Orchestrator()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await orchestrator.initialize()
    yield
    await orchestrator.shutdown()


app = FastAPI(
    title="Bay-Max API",
    description="Memory-first empathetic companion agent",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Basic health check."""
    return {"status": "ok"}


# --- Users ---


@app.post("/v1/users", response_model=UserProfile, status_code=201)
async def create_user(request: UserProfileCreate) -> UserProfile:
    """Create a new user profile."""
    return await orchestrator.create_user(request)


@app.get("/v1/users", response_model=list[UserProfile])
async def list_users() -> list[UserProfile]:
    """List all active users."""
    return await orchestrator.list_users()


@app.get("/v1/users/{user_id}", response_model=UserProfile)
async def get_user(user_id: UUID) -> UserProfile:
    """Fetch a user profile."""
    user = await orchestrator.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


# --- Face Enrollment ---


@app.post("/v1/users/{user_id}/enroll/face", response_model=FaceEnrollment, status_code=201)
async def enroll_face(user_id: UUID, file: UploadFile = File(...)) -> FaceEnrollment:
    """Enroll a face from an uploaded image.

    Expects a single image file (JPEG/PNG). The image must contain exactly one face.
    """
    user = await orchestrator.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    # Read and decode the uploaded image
    contents = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        frame = np.array(pil_image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    return await orchestrator.enroll_face(user_id, frame)


# --- Sessions ---


@app.post("/v1/sessions", response_model=Session, status_code=201)
async def create_session(request: SessionCreate) -> Session:
    """Create a new interaction session."""
    return await orchestrator.create_session(user_id=request.user_id)


@app.get("/v1/sessions/{session_id}/state", response_model=InteractionState)
async def get_session_state(session_id: UUID) -> InteractionState:
    """Get the current interaction state for a session."""
    state = await orchestrator.get_session_state(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return state


# --- Frame Analysis ---


@app.post("/v1/sessions/{session_id}/frames", response_model=FrameAnalysisResult)
async def analyze_frame(session_id: UUID, file: UploadFile = File(...)) -> FrameAnalysisResult:
    """Submit a frame for face detection and recognition.

    Expects a single image file (JPEG/PNG). Returns detection results,
    recognized/unknown faces, updated state, and any observations written.
    """
    session = await orchestrator.get_session_state(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    contents = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        frame = np.array(pil_image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    return await orchestrator.analyze_frame(session_id, frame)


# --- Memory ---


@app.post("/v1/memory/query", response_model=MemoryQueryResult)
async def query_memories(request: MemoryQuery) -> MemoryQueryResult:
    """Retrieve memories for a user."""
    return await orchestrator.query_memories(
        user_id=request.user_id,
        query=request.query,
        limit=request.limit,
    )


# --- Response ---


@app.post("/v1/respond", response_model=SupportiveResponse)
async def respond(request: RespondRequest) -> SupportiveResponse:
    """Generate a supportive personalized response."""
    return await orchestrator.respond(
        session_id=request.session_id,
        user_id=request.user_id,
        context=request.context,
    )
