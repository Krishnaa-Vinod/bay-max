"""Grounded prompt builder for memory-aware LLM dialogue.

Uses the plan-then-verbalize pattern: the planner chooses a strategy, the
prompt builder packages observed facts, retrieved memories, recent conversation
turns, and explicit safety constraints into a structured prompt, and the LLM
verbalizes within those constraints.
"""

from baymax.core.enums import ResponseStrategy
from baymax.schemas.memory import ChatTurn, MemoryQueryResult
from baymax.schemas.response import GroundedPromptContext
from baymax.state.models import InteractionState

_STRATEGY_INSTRUCTIONS: dict[ResponseStrategy, str] = {
    ResponseStrategy.GREET: (
        "The user has just arrived or started a new session. "
        "Greet them warmly and personally if you know their name. "
        "Ask an open, gentle question about how they are doing."
    ),
    ResponseStrategy.CHECK_IN: (
        "The user appears disengaged or may be stepping away. "
        "Gently acknowledge this and let them know you are here whenever they are ready. "
        "Do not pressure them."
    ),
    ResponseStrategy.ENCOURAGE: (
        "The user is engaged and present. "
        "Offer genuine acknowledgment of their presence and encourage them. "
        "Keep it warm but brief."
    ),
    ResponseStrategy.EMPATHIZE: (
        "The user appears to be in a difficult emotional state. "
        "Acknowledge their feelings without minimising them. "
        "Do not offer solutions unless asked. Simply be present."
    ),
    ResponseStrategy.RECALL: (
        "You have retrieved relevant memories about this user. "
        "Weave in one or two specific remembered details naturally — "
        "but only facts you actually have. Do not fabricate details you do not have."
    ),
    ResponseStrategy.SUGGEST: (
        "The user appears tired or under strain. "
        "Gently suggest a simple self-care action such as a short break, "
        "a glass of water, or a moment of calm. Keep it inviting, not prescriptive."
    ),
    ResponseStrategy.FAREWELL: (
        "The session is ending or the user is leaving. "
        "Say a warm, caring goodbye. Remind them you will be here next time."
    ),
}

_SAFETY_RULES = [
    "You are a supportive companion, not a clinician.",
    "You must never diagnose a disease, condition, or illness.",
    "You must never recommend medication, treatment, or dosage.",
    "You must never assert medical certainty beyond what you observe.",
    "If the user asks for medical advice or diagnosis, gently redirect them "
    "to a qualified healthcare professional.",
    "Keep responses warm, supportive, concise, and grounded in what you actually know.",
    "Never fabricate memories or facts you do not have.",
    "Speak in first person as Bay-Max.",
]


def build_system_prompt() -> str:
    """Return the fixed system-level persona and safety instruction."""
    rules = "\n".join(f"- {r}" for r in _SAFETY_RULES)
    return (
        "You are Bay-Max, a warm and attentive supportive companion. "
        "You remember what users share with you across sessions and use those memories "
        "to personalise your responses.\n\n"
        "Safety rules you must always follow:\n"
        f"{rules}\n\n"
        "Respond with a single concise supportive message (1-3 sentences). "
        "Do not add headers, bullet points, or meta-commentary."
    )


def build_grounded_user_prompt(ctx: GroundedPromptContext) -> str:
    """Build the user-turn prompt from the grounded context package."""
    parts: list[str] = []

    # User identity
    if ctx.user_display_name:
        parts.append(f"User name: {ctx.user_display_name}")

    # Current state
    if ctx.state_summary:
        state_str = ", ".join(f"{k}={v}" for k, v in ctx.state_summary.items())
        parts.append(f"Current session state: {state_str}")

    # Retrieved memories
    if ctx.memory_refs:
        mem_lines = "\n".join(f"  - {m}" for m in ctx.memory_refs[: ctx.top_k_memories])
        parts.append(f"Memories about this user:\n{mem_lines}")
    else:
        parts.append(
            "No memories retrieved for this user yet. "
            "Do not invent past interactions."
        )

    # Recent conversation turns
    if ctx.recent_turns:
        turn_lines: list[str] = []
        for t in ctx.recent_turns[-8:]:
            role = t.get("role", "user")
            text = t.get("text", "")
            turn_lines.append(f"  {role}: {text}")
        parts.append("Recent conversation:\n" + "\n".join(turn_lines))

    # Optional free-form context
    if ctx.context:
        parts.append(f"Additional context: {ctx.context}")

    # Strategy instruction
    strategy_instr = _STRATEGY_INSTRUCTIONS.get(
        ctx.strategy,
        _STRATEGY_INSTRUCTIONS[ResponseStrategy.ENCOURAGE],
    )
    parts.append(f"Response guidance: {strategy_instr}")

    parts.append(
        "Now write a single supportive response following the guidance above. "
        "Be natural and warm. Do not repeat the instructions."
    )

    return "\n\n".join(parts)


def build_messages(ctx: GroundedPromptContext) -> list[dict]:
    """Return an OpenAI-style messages list for chat completion APIs."""
    return [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_grounded_user_prompt(ctx)},
    ]


def build_prompt_context(
    strategy: ResponseStrategy,
    state: InteractionState,
    memories: MemoryQueryResult,
    recent_turns: list[ChatTurn],
    context: str = "",
    top_k_memories: int = 5,
) -> GroundedPromptContext:
    """Assemble a GroundedPromptContext from orchestrator data.

    Args:
        strategy: The planned response strategy.
        state: Current interaction state.
        memories: Retrieved memories.
        recent_turns: The most recent chat turns from SQLite.
        context: Optional free-form context string from the caller.
        top_k_memories: Number of memories to include.

    Returns:
        A fully-populated GroundedPromptContext ready for prompt building.
    """
    memory_refs: list[str] = []
    for m in memories.episodic_memories:
        if m.content not in memory_refs:
            memory_refs.append(m.content)
    for f in memories.semantic_facts:
        if f.content not in memory_refs:
            memory_refs.append(f.content)

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

    return GroundedPromptContext(
        user_id=state.user_id,
        user_display_name=state.user_display_name,
        session_id=state.session_id,
        strategy=strategy,
        state_summary=state_summary,
        memory_refs=memory_refs[:top_k_memories],
        recent_turns=turn_dicts,
        safety_rules=_SAFETY_RULES,
        context=context,
        top_k_memories=top_k_memories,
    )
