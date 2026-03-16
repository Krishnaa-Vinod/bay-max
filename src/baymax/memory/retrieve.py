"""Memory retrieval utilities."""

from uuid import UUID

from baymax.memory.interfaces import MetadataStore
from baymax.schemas.memory import MemoryQuery, MemoryQueryResult, MemoryType


async def retrieve_memories(
    store: MetadataStore,
    user_id: UUID,
    query: str = "",
    memory_types: list[MemoryType] | None = None,
    limit: int = 10,
) -> MemoryQueryResult:
    """Retrieve memories for a user from the metadata store."""
    if memory_types is None:
        memory_types = [MemoryType.EPISODIC, MemoryType.SEMANTIC]

    mq = MemoryQuery(
        user_id=user_id,
        query=query,
        memory_types=memory_types,
        limit=limit,
    )
    return await store.query_memories(mq)
