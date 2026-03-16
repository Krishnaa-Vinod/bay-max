"""Memory module interfaces."""

from abc import ABC, abstractmethod
from uuid import UUID

from baymax.schemas.memory import (
    EpisodicMemory,
    MemoryQuery,
    MemoryQueryResult,
    SemanticFact,
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
    async def create_face_enrollment(self, enrollment: FaceEnrollment) -> FaceEnrollment:
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
    async def query_memories(self, query: MemoryQuery) -> MemoryQueryResult:
        ...

    @abstractmethod
    async def close(self) -> None:
        ...
