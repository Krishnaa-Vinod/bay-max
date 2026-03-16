"""Memory-related Pydantic schemas."""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from baymax.core.enums import (
    CorrectionAction,
    FactStatus,
    MemoryStatus,
    MemoryType,
    TurnRole,
)


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


# --- Iteration 004: new schemas ---


class ChatTurn(BaseModel):
    """A single typed conversation turn."""

    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    user_id: UUID | None = None
    role: TurnRole
    text: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SessionSummary(BaseModel):
    """Summary produced by consolidating a session."""

    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    user_id: UUID
    summary_text: str
    turn_count: int = 0
    observation_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    episodic_ids: list[str] = Field(default_factory=list)
    semantic_ids: list[str] = Field(default_factory=list)


class MemoryEmbeddingRecord(BaseModel):
    """Record linking a memory to its vector embedding."""

    memory_id: UUID
    memory_type: MemoryType
    user_id: UUID
    content: str
    embedding: list[float] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MemoryHit(BaseModel):
    """A single result from semantic memory search."""

    memory_id: UUID
    memory_type: MemoryType
    content: str
    score: float
    user_id: UUID


class ConsolidationResult(BaseModel):
    """Result of running session consolidation."""

    session_id: UUID
    user_id: UUID
    summary: SessionSummary | None = None
    episodic_memories_created: int = 0
    semantic_facts_created: int = 0
    errors: list[str] = Field(default_factory=list)


class MemoryCorrectionRequest(BaseModel):
    """Request to correct a semantic fact."""

    fact_id: UUID
    action: CorrectionAction
    updated_content: str | None = None


class MemoryCorrectionResult(BaseModel):
    """Result of a memory correction."""

    fact_id: UUID
    action: CorrectionAction
    previous_content: str
    new_content: str | None = None
    new_status: FactStatus


class MemorySummaryResponse(BaseModel):
    """Summary of all memories for a user."""

    user_id: UUID
    episodic_count: int = 0
    semantic_count: int = 0
    intervention_count: int = 0
    session_summaries: list[SessionSummary] = Field(default_factory=list)
    recent_episodic: list[EpisodicMemory] = Field(default_factory=list)
    confirmed_facts: list[SemanticFact] = Field(default_factory=list)


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
