"""Bay-Max FastAPI application."""

from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import FastAPI, HTTPException

from baymax.orchestrator.service import Orchestrator
from baymax.schemas.memory import MemoryQuery, MemoryQueryResult
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
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Basic health check."""
    return {"status": "ok"}


@app.post("/v1/users", response_model=UserProfile, status_code=201)
async def create_user(request: UserProfileCreate) -> UserProfile:
    """Create a new user profile."""
    return await orchestrator.create_user(request)


@app.get("/v1/users/{user_id}", response_model=UserProfile)
async def get_user(user_id: UUID) -> UserProfile:
    """Fetch a user profile."""
    user = await orchestrator.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.post("/v1/users/{user_id}/enroll/face", response_model=FaceEnrollment, status_code=201)
async def enroll_face(user_id: UUID) -> FaceEnrollment:
    """Store placeholder face enrollment metadata."""
    user = await orchestrator.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return await orchestrator.enroll_face(user_id)


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


@app.post("/v1/memory/query", response_model=MemoryQueryResult)
async def query_memories(request: MemoryQuery) -> MemoryQueryResult:
    """Retrieve memories for a user."""
    return await orchestrator.query_memories(
        user_id=request.user_id,
        query=request.query,
        limit=request.limit,
    )


@app.post("/v1/respond", response_model=SupportiveResponse)
async def respond(request: RespondRequest) -> SupportiveResponse:
    """Generate a supportive personalized response."""
    return await orchestrator.respond(
        session_id=request.session_id,
        user_id=request.user_id,
        context=request.context,
    )
