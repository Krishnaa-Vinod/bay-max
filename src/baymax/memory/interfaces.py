"""Memory module interfaces."""

from abc import ABC, abstractmethod
from uuid import UUID

from baymax.schemas.memory import (
    ChatTurn,
    EpisodicMemory,
    MemoryQuery,
    MemoryQueryResult,
    SemanticFact,
    SessionSummary,
)
from baymax.schemas.perception import (
    BodyStateObservation,
    FaceEmbeddingRecord,
    RecognitionObservation,
)
from baymax.schemas.session import Session
from baymax.schemas.user import FaceEnrollment, UserProfile


class MetadataStore(ABC):
    """Interface for persistent metadata storage (users, sessions, memories)."""

    @abstractmethod
    async def create_user(self, user: UserProfile) -> UserProfile:
        ...

    @abstractmethod
    async def get_user(self, user_id: UUID) -> UserProfile | None:
        ...

    @abstractmethod
    async def list_users(self) -> list[UserProfile]:
        ...

    @abstractmethod
    async def create_face_enrollment(self, enrollment: FaceEnrollment) -> FaceEnrollment:
        ...

    @abstractmethod
    async def store_face_embedding(self, record: FaceEmbeddingRecord) -> FaceEmbeddingRecord:
        ...

    @abstractmethod
    async def get_all_face_embeddings(self) -> list[FaceEmbeddingRecord]:
        ...

    @abstractmethod
    async def get_face_embeddings_for_user(self, user_id: UUID) -> list[FaceEmbeddingRecord]:
        ...

    @abstractmethod
    async def create_session(self, session: Session) -> Session:
        ...

    @abstractmethod
    async def get_session(self, session_id: UUID) -> Session | None:
        ...

    @abstractmethod
    async def store_episodic_memory(self, memory: EpisodicMemory) -> EpisodicMemory:
        ...

    @abstractmethod
    async def store_semantic_fact(self, fact: SemanticFact) -> SemanticFact:
        ...

    @abstractmethod
    async def get_semantic_fact(self, fact_id: UUID) -> SemanticFact | None:
        ...

    @abstractmethod
    async def update_semantic_fact(self, fact: SemanticFact) -> SemanticFact:
        ...

    @abstractmethod
    async def store_observation(
        self, obs: RecognitionObservation | BodyStateObservation
    ) -> RecognitionObservation | BodyStateObservation:
        ...

    @abstractmethod
    async def query_memories(self, query: MemoryQuery) -> MemoryQueryResult:
        ...

    # --- Iteration 004: chat turns ---

    @abstractmethod
    async def store_chat_turn(self, turn: ChatTurn) -> ChatTurn:
        ...

    @abstractmethod
    async def get_chat_turns(self, session_id: UUID) -> list[ChatTurn]:
        ...

    # --- Iteration 004: session summaries ---

    @abstractmethod
    async def store_session_summary(self, summary: SessionSummary) -> SessionSummary:
        ...

    @abstractmethod
    async def get_session_summaries(self, user_id: UUID) -> list[SessionSummary]:
        ...

    # --- Iteration 004: observations for consolidation ---

    @abstractmethod
    async def get_observations_for_session(
        self, session_id: UUID
    ) -> list[RecognitionObservation | BodyStateObservation]:
        ...

    @abstractmethod
    async def count_episodic_memories(self, user_id: UUID) -> int:
        ...

    @abstractmethod
    async def count_semantic_facts(self, user_id: UUID) -> int:
        ...

    @abstractmethod
    async def close(self) -> None:
        ...
