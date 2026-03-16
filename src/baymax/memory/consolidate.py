"""Memory consolidation: session turns + observations -> episodic memories + semantic facts."""

import logging
from uuid import UUID

from baymax.memory.interfaces import MetadataStore
from baymax.schemas.memory import (
    ChatTurn,
    ConsolidationResult,
    EpisodicMemory,
    SemanticFact,
    SessionSummary,
)
from baymax.schemas.perception import BodyStateObservation, RecognitionObservation

logger = logging.getLogger(__name__)


def _summarize_turns(turns: list[ChatTurn]) -> str:
    """Build a natural-language summary from typed turns."""
    if not turns:
        return ""
    parts = []
    for t in turns:
        parts.append(f"{t.role.value}: {t.text}")
    return " | ".join(parts)


def _summarize_observations(
    observations: list[RecognitionObservation | BodyStateObservation],
) -> str:
    """Build a summary from observation contents."""
    if not observations:
        return ""
    unique = []
    seen: set[str] = set()
    for obs in observations:
        if obs.content not in seen:
            unique.append(obs.content)
            seen.add(obs.content)
    return " ".join(unique)


def _extract_episodic(
    session_id: UUID,
    user_id: UUID,
    turns: list[ChatTurn],
    observations: list[RecognitionObservation | BodyStateObservation],
) -> list[EpisodicMemory]:
    """Create episodic memories from a session's content."""
    memories: list[EpisodicMemory] = []

    # Create an episodic memory from conversation turns if any exist
    if turns:
        turn_summary = _summarize_turns(turns)
        memories.append(EpisodicMemory(
            user_id=user_id,
            session_id=session_id,
            content=f"Conversation: {turn_summary}",
            salience=0.6,
            evidence_refs=[f"session:{session_id}"],
        ))

    # Create episodic memories from notable observations
    for obs in observations:
        if obs.event_type in (
            "first_recognition",
            "engagement_change",
            "posture_change",
            "pose_first_seen",
        ):
            memories.append(EpisodicMemory(
                user_id=user_id,
                session_id=session_id,
                content=obs.content,
                salience=0.5,
                evidence_refs=[f"observation:{obs.id}"],
            ))

    return memories


def _extract_semantic_candidates(
    user_id: UUID,
    turns: list[ChatTurn],
) -> list[SemanticFact]:
    """Extract candidate semantic facts from session content.

    Uses simple heuristics: looks for preference-like statements in user turns.
    """
    facts: list[SemanticFact] = []

    preference_signals = [
        "i like", "i prefer", "i enjoy", "my favorite", "i love",
        "i always", "i usually", "i tend to",
    ]
    for turn in turns:
        if turn.role.value != "user":
            continue
        text_lower = turn.text.lower()
        for signal in preference_signals:
            if signal in text_lower:
                facts.append(SemanticFact(
                    user_id=user_id,
                    content=f"User stated: {turn.text}",
                    confidence=0.4,
                    evidence_refs=[f"turn:{turn.id}"],
                ))
                break

    return facts


async def consolidate_session(
    store: MetadataStore,
    session_id: UUID,
    user_id: UUID,
) -> ConsolidationResult:
    """Consolidate a session into episodic memories and candidate semantic facts.

    Steps:
    1. Fetch chat turns and observations for the session
    2. Build a session summary
    3. Extract episodic memories
    4. Extract candidate semantic facts
    5. Store everything and return results
    """
    errors: list[str] = []

    turns = await store.get_chat_turns(session_id)
    observations = await store.get_observations_for_session(session_id)

    if not turns and not observations:
        return ConsolidationResult(
            session_id=session_id,
            user_id=user_id,
            errors=["No turns or observations found for session"],
        )

    # Build summary
    turn_summary = _summarize_turns(turns)
    obs_summary = _summarize_observations(observations)
    summary_text = ""
    if turn_summary:
        summary_text += f"Conversation: {turn_summary}"
    if obs_summary:
        if summary_text:
            summary_text += " | "
        summary_text += f"Observations: {obs_summary}"

    session_summary = SessionSummary(
        session_id=session_id,
        user_id=user_id,
        summary_text=summary_text,
        turn_count=len(turns),
        observation_count=len(observations),
    )

    # Extract memories
    episodic_list = _extract_episodic(session_id, user_id, turns, observations)
    semantic_list = _extract_semantic_candidates(user_id, turns)

    # Store everything
    episodic_ids: list[str] = []
    semantic_ids: list[str] = []

    for mem in episodic_list:
        try:
            stored = await store.store_episodic_memory(mem)
            episodic_ids.append(str(stored.id))
        except Exception as e:
            errors.append(f"Failed to store episodic memory: {e}")
            logger.warning("Failed to store episodic memory", exc_info=True)

    for fact in semantic_list:
        try:
            stored = await store.store_semantic_fact(fact)
            semantic_ids.append(str(stored.id))
        except Exception as e:
            errors.append(f"Failed to store semantic fact: {e}")
            logger.warning("Failed to store semantic fact", exc_info=True)

    session_summary.episodic_ids = episodic_ids
    session_summary.semantic_ids = semantic_ids

    try:
        await store.store_session_summary(session_summary)
    except Exception as e:
        errors.append(f"Failed to store session summary: {e}")

    return ConsolidationResult(
        session_id=session_id,
        user_id=user_id,
        summary=session_summary,
        episodic_memories_created=len(episodic_ids),
        semantic_facts_created=len(semantic_ids),
        errors=errors,
    )


async def consolidate_episodic_to_semantic(
    memories: list[EpisodicMemory],
) -> list[SemanticFact]:
    """Analyze episodic memories and extract stable semantic facts."""
    if not memories:
        return []

    by_user: dict[UUID, list[EpisodicMemory]] = {}
    for m in memories:
        by_user.setdefault(m.user_id, []).append(m)

    facts: list[SemanticFact] = []
    for uid, user_memories in by_user.items():
        if len(user_memories) >= 3:
            facts.append(SemanticFact(
                user_id=uid,
                content=f"User has had {len(user_memories)} recorded interactions.",
                confidence=0.5,
                evidence_refs=[str(m.id) for m in user_memories[-3:]],
            ))
    return facts


async def merge_duplicate_facts(facts: list[SemanticFact]) -> list[SemanticFact]:
    """Merge duplicate or overlapping semantic facts.

    Deduplicates by exact content match, keeping highest confidence.
    """
    seen: dict[str, SemanticFact] = {}
    for fact in facts:
        key = fact.content.lower().strip()
        if key in seen:
            existing = seen[key]
            if fact.confidence > existing.confidence:
                seen[key] = fact
        else:
            seen[key] = fact
    return list(seen.values())
