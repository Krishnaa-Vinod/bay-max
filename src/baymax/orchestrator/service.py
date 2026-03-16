"""Orchestrator service coordinating the end-to-end flow."""

from uuid import UUID

from baymax.dialogue.rule_based import RuleBasedDialogue
from baymax.memory.retrieve import retrieve_memories
from baymax.memory.store_sqlite import SQLiteMetadataStore
from baymax.planner.supportive_planner import SupportivePlanner
from baymax.schemas.memory import MemoryQueryResult
from baymax.schemas.response import SupportiveResponse
from baymax.schemas.session import Session
from baymax.schemas.user import FaceEnrollment, UserProfile, UserProfileCreate
from baymax.state.manager import StateManager
from baymax.state.models import InteractionState


class Orchestrator:
    """Coordinates the flow from frame ingest to supportive response."""

    def __init__(self, db_path: str = "baymax.db") -> None:
        self.store = SQLiteMetadataStore(db_path=db_path)
        self.state_manager = StateManager()
        self.planner = SupportivePlanner()
        self.dialogue = RuleBasedDialogue()

    async def initialize(self) -> None:
        """Initialize the orchestrator and its dependencies."""
        await self.store.initialize()

    async def create_user(self, request: UserProfileCreate) -> UserProfile:
        """Create a new user profile."""
        user = UserProfile(display_name=request.display_name, notes=request.notes)
        return await self.store.create_user(user)

    async def get_user(self, user_id: UUID) -> UserProfile | None:
        """Fetch a user profile by ID."""
        return await self.store.get_user(user_id)

    async def enroll_face(self, user_id: UUID) -> FaceEnrollment:
        """Create a placeholder face enrollment for a user."""
        enrollment = FaceEnrollment(user_id=user_id, status="pending")
        return await self.store.create_face_enrollment(enrollment)

    async def create_session(self, user_id: UUID | None = None) -> Session:
        """Create a new interaction session."""
        session = Session(user_id=user_id)
        stored = await self.store.create_session(session)
        self.state_manager.get_or_create(stored.id, user_id=user_id)
        return stored

    async def get_session_state(self, session_id: UUID) -> InteractionState | None:
        """Get the current interaction state for a session."""
        session = await self.store.get_session(session_id)
        if session is None:
            return None
        return self.state_manager.get_or_create(session_id, user_id=session.user_id)

    async def query_memories(
        self,
        user_id: UUID,
        query: str = "",
        limit: int = 10,
    ) -> MemoryQueryResult:
        """Query memories for a user."""
        return await retrieve_memories(
            store=self.store,
            user_id=user_id,
            query=query,
            limit=limit,
        )

    async def respond(
        self,
        session_id: UUID,
        user_id: UUID | None = None,
        context: str = "",
    ) -> SupportiveResponse:
        """Generate a supportive response for the current session."""
        state = self.state_manager.get_or_create(session_id, user_id=user_id)

        # Retrieve memories if we have a user
        memories = MemoryQueryResult(
            user_id=user_id or UUID("00000000-0000-0000-0000-000000000000"),
            query=context,
        )
        if user_id:
            memories = await retrieve_memories(
                store=self.store,
                user_id=user_id,
                query=context,
                limit=5,
            )

        # Plan strategy
        strategy = self.planner.plan(state, memories)

        # Generate response
        response = self.dialogue.generate(strategy, state, memories)

        # Update state
        self.state_manager.increment_turn(session_id)

        return response

    async def shutdown(self) -> None:
        """Shut down the orchestrator."""
        await self.store.close()
