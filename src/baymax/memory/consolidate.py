"""Memory consolidation stubs for iteration-001."""

from baymax.schemas.memory import EpisodicMemory, SemanticFact


async def consolidate_episodic_to_semantic(
    memories: list[EpisodicMemory],
) -> list[SemanticFact]:
    """Stub: In future iterations, this will analyze episodic memories
    and extract stable semantic facts.

    For iteration-001, returns an empty list.
    """
    return []


async def merge_duplicate_facts(facts: list[SemanticFact]) -> list[SemanticFact]:
    """Stub: In future iterations, this will merge duplicate or overlapping
    semantic facts.

    For iteration-001, returns the input unchanged.
    """
    return facts
