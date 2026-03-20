"""Grounded prompt builder for memory-aware Bay-Max dialogue."""

from __future__ import annotations

import re

from baymax.core.enums import ResponseStrategy, TurnIntent
from baymax.schemas.memory import ChatTurn, MemoryQueryResult
from baymax.schemas.response import GroundedPromptContext
from baymax.state.models import InteractionState

_ANSWER_STRATEGIES = {
    ResponseStrategy.ANSWER,
    ResponseStrategy.WEB_ANSWER,
    ResponseStrategy.CLARIFY,
}

_COMPANION_STRATEGY_INSTRUCTIONS: dict[ResponseStrategy, str] = {
    ResponseStrategy.GREET: (
        "The user has just arrived or started a new session. "
        "Greet them warmly and briefly."
    ),
    ResponseStrategy.CHECK_IN: (
        "The user appears disengaged. "
        "Offer a low-pressure check-in and let them lead."
    ),
    ResponseStrategy.ENCOURAGE: (
        "The user is engaged. "
        "Respond with brief encouragement without overdoing tone."
    ),
    ResponseStrategy.EMPATHIZE: (
        "The user is sharing a difficult feeling. "
        "Validate emotions first and avoid generic factual pivots."
    ),
    ResponseStrategy.RECALL: (
        "The user asked to recall prior context. "
        "Use one concrete memory detail if available."
    ),
    ResponseStrategy.SUGGEST: (
        "Offer one practical, optional suggestion in a calm tone."
    ),
    ResponseStrategy.FAREWELL: (
        "Close warmly and briefly."
    ),
    ResponseStrategy.PROACTIVE_CHECK_IN: (
        "Offer a short, gentle optional check-in. "
        "Do not pressure the user."
    ),
}

_SAFETY_RULES = [
    "You are a supportive companion, not a clinician.",
    "Never diagnose a disease, condition, or illness.",
    "Never recommend medication, treatment, or dosage.",
    "If asked for medical diagnosis/treatment advice, redirect to qualified care.",
    "Never fabricate memories or facts.",
]

_PERSONA_STYLE = (
    "Calm, literal, warm, concise. Usually 1-3 sentences. "
    "Avoid forced cheerfulness and avoid rambling."
)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def _memory_relevance(memory: str, context: str) -> float:
    mem_tokens = _tokenize(memory)
    ctx_tokens = _tokenize(context)
    if not mem_tokens or not ctx_tokens:
        return 0.0
    overlap = len(mem_tokens & ctx_tokens)
    return overlap / float(max(len(ctx_tokens), 1))


def select_relevant_memories(
    memory_refs: list[str],
    context: str,
    turn_intent: TurnIntent | None,
    threshold: float = 0.18,
    top_k: int = 5,
) -> list[str]:
    """Return memory refs only when they are relevant to the current turn."""
    if not memory_refs:
        return []

    if turn_intent in (TurnIntent.RECALL, TurnIntent.FOLLOW_UP):
        return memory_refs[:top_k]

    if turn_intent == TurnIntent.EMOTIONAL_SHARE:
        scored = sorted(
            ((m, _memory_relevance(m, context)) for m in memory_refs),
            key=lambda item: item[1],
            reverse=True,
        )
        return [m for m, s in scored if s >= threshold][:top_k]

    scored = sorted(
        ((m, _memory_relevance(m, context)) for m in memory_refs),
        key=lambda item: item[1],
        reverse=True,
    )
    return [m for m, s in scored if s >= threshold][:top_k]


def build_system_prompt(
    strategy: ResponseStrategy = ResponseStrategy.ENCOURAGE,
    limited_mode: bool = False,
) -> str:
    """Return a strategy-aware system prompt."""
    rules = "\n".join(f"- {r}" for r in _SAFETY_RULES)

    if strategy in _ANSWER_STRATEGIES:
        limited_clause = ""
        if limited_mode:
            limited_clause = (
                " You are in a limited fallback backend. Be truthful about limits and "
                "do not claim high-confidence world knowledge without sources."
            )
        return (
            "You are Bay-Max. Answer the user's exact question first. "
            "Stay on-topic, concise, and coherent with recent turn context. "
            "Do not force a memory mention. Do not add supportive fluff unless it naturally fits. "
            "Do not ask unnecessary follow-up questions."
            f"{limited_clause}\n\n"
            "Safety rules:\n"
            f"{rules}"
        )

    return (
        "You are Bay-Max, a warm and attentive supportive companion.\n"
        f"Companion style: {_PERSONA_STYLE}\n\n"
        "Safety rules:\n"
        f"{rules}"
    )


def build_grounded_user_prompt(ctx: GroundedPromptContext) -> str:
    """Build the user-turn prompt from grounded context."""
    parts: list[str] = []

    if ctx.user_display_name:
        parts.append(f"User name: {ctx.user_display_name}")

    if ctx.context:
        parts.append(f"Latest user turn: {ctx.context}")

    if ctx.recent_turns:
        turn_lines: list[str] = []
        for t in ctx.recent_turns[-8:]:
            role = t.get("role", "user")
            text = t.get("text", "")
            turn_lines.append(f"  {role}: {text}")
        parts.append("Recent conversation:\n" + "\n".join(turn_lines))

    if ctx.strategy in _ANSWER_STRATEGIES:
        if ctx.memory_refs:
            mem_lines = "\n".join(f"  - {m}" for m in ctx.memory_refs[: ctx.top_k_memories])
            parts.append(
                "Relevant optional memory context (use only if directly helpful):\n"
                f"{mem_lines}"
            )

        if ctx.strategy == ResponseStrategy.CLARIFY:
            parts.append(
                "Respond with a short clarification request because the transcript/input "
                "is uncertain or unclear."
            )
        elif ctx.strategy == ResponseStrategy.WEB_ANSWER:
            parts.append(
                "Answer first with current information from provided web context when present. "
                "Keep concise."
            )
        else:
            parts.append("Answer the user's exact question directly and briefly.")

        return "\n\n".join(parts)

    if ctx.memory_refs:
        mem_lines = "\n".join(f"  - {m}" for m in ctx.memory_refs[: ctx.top_k_memories])
        parts.append(f"Memories about this user:\n{mem_lines}")

    if ctx.affect_context:
        parts.append(f"Observation note (internal): {ctx.affect_context}")

    strategy_instr = _COMPANION_STRATEGY_INSTRUCTIONS.get(
        ctx.strategy,
        _COMPANION_STRATEGY_INSTRUCTIONS[ResponseStrategy.ENCOURAGE],
    )
    parts.append(f"Response guidance: {strategy_instr}")
    parts.append("Write one concise response in Bay-Max voice.")

    return "\n\n".join(parts)


def build_messages(ctx: GroundedPromptContext) -> list[dict]:
    """Return an OpenAI-style messages list for chat completion APIs."""
    limited_mode = bool(ctx.backend_limited_mode)
    return [
        {"role": "system", "content": build_system_prompt(ctx.strategy, limited_mode=limited_mode)},
        {"role": "user", "content": build_grounded_user_prompt(ctx)},
    ]


def build_prompt_context(
    strategy: ResponseStrategy,
    state: InteractionState,
    memories: MemoryQueryResult,
    recent_turns: list[ChatTurn],
    context: str = "",
    top_k_memories: int = 5,
    affect_bias_confidence_threshold: float = 0.6,
    turn_intent: TurnIntent | None = None,
    memory_relevance_threshold: float = 0.18,
    backend_limited_mode: bool = False,
) -> GroundedPromptContext:
    """Assemble a GroundedPromptContext from orchestrator data."""
    all_memory_refs: list[str] = []
    for m in memories.episodic_memories:
        if m.content not in all_memory_refs:
            all_memory_refs.append(m.content)
    for f in memories.semantic_facts:
        if f.content not in all_memory_refs:
            all_memory_refs.append(f.content)

    selected_memory_refs = select_relevant_memories(
        all_memory_refs,
        context=context,
        turn_intent=(turn_intent or (TurnIntent.RECALL if strategy == ResponseStrategy.RECALL else None)),
        threshold=memory_relevance_threshold,
        top_k=top_k_memories,
    )

    state_summary: dict[str, str] = {
        "engagement": state.engagement_level.value,
        "posture": state.posture.value,
        "turn_count": str(state.turn_count),
    }
    if state.user_display_name:
        state_summary["user"] = state.user_display_name

    turn_dicts = [
        {
            "role": t.role.value,
            "text": t.text,
            "timestamp": t.timestamp.isoformat(),
        }
        for t in recent_turns
    ]

    affect_context = ""
    if (
        state.affect_enabled
        and state.affect_confidence >= affect_bias_confidence_threshold
        and strategy not in _ANSWER_STRATEGIES
    ):
        if state.valence <= -0.3 and state.arousal <= 0.4:
            affect_context = "User appears subdued; use gentler validating tone"
        elif state.valence <= -0.3 and state.arousal > 0.55:
            affect_context = "User appears tense; prioritize calming empathy"

    ctx = GroundedPromptContext(
        user_id=state.user_id,
        user_display_name=state.user_display_name,
        session_id=state.session_id,
        strategy=strategy,
        state_summary=state_summary,
        memory_refs=selected_memory_refs,
        recent_turns=turn_dicts,
        safety_rules=_SAFETY_RULES,
        context=context,
        top_k_memories=top_k_memories,
        affect_context=affect_context,
        turn_intent=turn_intent.value if turn_intent else "",
        backend_limited_mode=backend_limited_mode,
    )
    return ctx
