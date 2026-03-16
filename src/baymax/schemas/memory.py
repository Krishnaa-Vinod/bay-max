"""Memory-related Pydantic schemas."""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from baymax.core.enums import MemoryStatus, MemoryType


class EpisodicMemory(BaseModel):
    """A memory of a specific interaction episode."""

    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    session_id: UUID
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    content: str = Field(..., description="Natural-language summary of the episode.")
    salience: float = Field(default=0.5, ge=0.0, le=1.0)
    emotion_context: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    status: MemoryStatus = MemoryStatus.ACTIVE
    memory_type: MemoryType = MemoryType.EPISODIC


class SemanticFact(BaseModel):
    """A stable fact about a user, derived from observations over time."""

    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    content: str = Field(
        ...,
        description="The semantic fact, e.g. 'User prefers morning greetings.'",
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    first_observed: datetime = Field(default_factory=datetime.utcnow)
    last_confirmed: datetime = Field(default_factory=datetime.utcnow)
    evidence_refs: list[str] = Field(default_factory=list)
    status: MemoryStatus = MemoryStatus.ACTIVE
    memory_type: MemoryType = MemoryType.SEMANTIC


class InterventionMemory(BaseModel):
    """A record of a supportive intervention the system performed."""

    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    session_id: UUID
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    strategy: str = Field(..., description="The response strategy used.")
    content: str = Field(..., description="What the system said or did.")
    user_reaction: str | None = None
    salience: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_refs: list[str] = Field(default_factory=list)
    status: MemoryStatus = MemoryStatus.ACTIVE
    memory_type: MemoryType = MemoryType.INTERVENTION


class RetrievalLog(BaseModel):
    """Log of a memory retrieval query and its results."""

    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    session_id: UUID | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    query: str
    results_count: int = 0
    top_memory_ids: list[str] = Field(default_factory=list)
    retrieval_method: str = "sqlite"
    latency_ms: float | None = None


class MemoryQuery(BaseModel):
    """Request schema for querying memories."""

    user_id: UUID
    query: str = Field(default="", description="Optional text query for semantic search.")
    memory_types: list[MemoryType] = Field(
        default_factory=lambda: [MemoryType.EPISODIC, MemoryType.SEMANTIC]
    )
    limit: int = Field(default=10, ge=1, le=100)


class MemoryQueryResult(BaseModel):
    """Response schema for memory queries."""

    user_id: UUID
    query: str
    episodic_memories: list[EpisodicMemory] = Field(default_factory=list)
    semantic_facts: list[SemanticFact] = Field(default_factory=list)
    intervention_memories: list[InterventionMemory] = Field(default_factory=list)
    total_count: int = 0


class ProjectState(BaseModel):
    """Machine-readable project state for agent continuity."""

    iteration: str
    branch_name: str
    status: str
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    modules_implemented: list[str] = Field(default_factory=list)
    modules_stubbed: list[str] = Field(default_factory=list)
    tests_passing: bool = False
    api_endpoints: list[str] = Field(default_factory=list)
    known_issues: list[str] = Field(default_factory=list)
    next_tasks: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
