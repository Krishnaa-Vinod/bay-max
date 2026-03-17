"""Memory consolidation: session turns + observations -> episodic memories + semantic facts."""

import logging
from datetime import datetime
from uuid import UUID

from baymax.memory.clinical_safety import (
    create_safe_affect_memory_content,
    is_affect_summary_safe,
    validate_memory_content,
)
from baymax.memory.interfaces import MetadataStore
from baymax.perception.emotion_smoother import AffectSmoother
from baymax.schemas.memory import (
    ChatTurn,
    ConsolidationResult,
    EpisodicMemory,
    SemanticFact,
    SessionSummary,
)
from baymax.schemas.perception import (
    AffectMemoryObservation,
    BodyStateObservation,
    RecognitionObservation,
)
from baymax.state.models import InteractionState

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


def _create_affect_observation(
    session_id: UUID,
    user_id: UUID,
    affect_smoother: AffectSmoother,
    session_duration_minutes: float,
    confidence_threshold: float = 0.65,
    stability_threshold_sec: float = 30.0,
) -> AffectMemoryObservation | None:
    """Create affect memory observation from smoother state if stable and confident.

    Args:
        session_id: Session UUID
        user_id: User UUID
        affect_smoother: AffectSmoother with session state
        session_duration_minutes: Duration of the session in minutes
        confidence_threshold: Minimum confidence to create observation
        stability_threshold_sec: Minimum stability duration required

    Returns:
        AffectMemoryObservation if conditions are met, None otherwise
    """
    if not affect_smoother.has_sufficient_confidence(confidence_threshold):
        logger.debug("Affect confidence too low for memory consolidation")
        return None

    if not affect_smoother.is_stable(stability_threshold_sec):
        logger.debug("Affect state not stable enough for memory consolidation")
        return None

    # Get affect summary
    summary = affect_smoother.get_affect_summary(session_duration_minutes)

    # Check clinical safety
    is_safe, issues = is_affect_summary_safe(summary)
    if not is_safe:
        logger.warning("Affect summary failed safety check: %s", issues)
        return None

    # Create safe content
    safe_content = create_safe_affect_memory_content(
        summary["valence_summary"],
        summary["arousal_summary"],
        summary["confidence"],
        session_duration_minutes
    )

    # Validate final content
    is_valid, validated_content = validate_memory_content(safe_content)
    if not is_valid:
        logger.error("Failed to create safe affect memory content: %s", validated_content)
        return None

    return AffectMemoryObservation(
        session_id=session_id,
        user_id=user_id,
        content=validated_content,
        valence_summary=summary["valence_summary"],
        arousal_summary=summary["arousal_summary"],
        confidence=summary["confidence"],
        duration_minutes=session_duration_minutes,
        stability_evidence=summary["stability_evidence"]
    )


async def consolidate_session(
    store: MetadataStore,
    session_id: UUID,
    user_id: UUID,
    affect_smoother: AffectSmoother | None = None,
    session_start_time: datetime | None = None,
) -> ConsolidationResult:
    """Consolidate a session into episodic memories and candidate semantic facts.

    Args:
        store: Metadata store interface
        session_id: Session UUID to consolidate
        user_id: User UUID
        affect_smoother: Optional affect smoother with session state
        session_start_time: Session start time for duration calculation

    Steps:
    1. Fetch chat turns and observations for the session
    2. Build a session summary
    3. Extract episodic memories
    4. Extract candidate semantic facts
    5. Create affect observation if stable and confident
    6. Store everything and return results
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

    # Calculate session duration
    session_duration_minutes = 0.0
    if session_start_time and turns:
        latest_turn_time = max(t.timestamp for t in turns)
        duration_delta = latest_turn_time - session_start_time
        session_duration_minutes = duration_delta.total_seconds() / 60.0

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

    # Create affect observation if conditions are met
    affect_observation = None
    if affect_smoother and session_duration_minutes >= 5.0:  # Minimum 5 minutes for affect summary
        affect_observation = _create_affect_observation(
            session_id=session_id,
            user_id=user_id,
            affect_smoother=affect_smoother,
            session_duration_minutes=session_duration_minutes
        )
        if affect_observation:
            logger.info("Created affect observation for session %s", session_id)

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

    # Store affect observation if created
    if affect_observation:
        try:
            # Convert affect observation to episodic memory for storage
            affect_memory = EpisodicMemory(
                user_id=user_id,
                session_id=session_id,
                content=affect_observation.content,
                salience=0.4,  # Lower salience for affect observations
                evidence_refs=[f"affect_analysis:session:{session_id}"],
            )
            stored_affect = await store.store_episodic_memory(affect_memory)
            episodic_ids.append(str(stored_affect.id))
            logger.debug("Stored affect observation as episodic memory %s", stored_affect.id)
        except Exception as e:
            errors.append(f"Failed to store affect observation: {e}")
            logger.warning("Failed to store affect observation", exc_info=True)

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
