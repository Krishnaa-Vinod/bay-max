"""Rule-based dialogue provider — baseline fallback.

This provider is intentionally lightweight but should remain context-aware,
memory-aware, and emotionally safe when LLM backends are unavailable.
"""

import re

from baymax.core.enums import ResponseStrategy
from baymax.dialogue.interfaces import DialogueProvider
from baymax.schemas.memory import MemoryQueryResult
from baymax.schemas.response import GroundedPromptContext, SupportiveResponse
from baymax.state.models import InteractionState

_TEMPLATES: dict[ResponseStrategy, list[str]] = {
    ResponseStrategy.ANSWER: [
        "I am in limited local mode right now, so I may not answer broad factual questions reliably.",
    ],
    ResponseStrategy.WEB_ANSWER: [
        "I can answer from the provided web context, but I am running in limited local mode.",
    ],
    ResponseStrategy.CLARIFY: [
        "I did not catch that clearly. Could you repeat it?",
    ],
    ResponseStrategy.PROACTIVE_CHECK_IN: [
        "You seem a little down. Do you want to talk about it?",
    ],
    ResponseStrategy.GREET: [
        "Hello! It's good to see you. How are you doing today?",
        "Hi there! Welcome. I'm here if you'd like to chat.",
    ],
    ResponseStrategy.CHECK_IN: [
        "I noticed you might be stepping away. I'm here whenever you're ready.",
        "Take your time. I'll be here when you'd like to continue.",
    ],
    ResponseStrategy.ENCOURAGE: [
        "You're doing well. Keep it up!",
        "I appreciate you being here. That takes effort, and it matters.",
    ],
    ResponseStrategy.EMPATHIZE: [
        "It sounds like things might be tough right now. That's completely understandable.",
        "I can see this might be a difficult moment. I'm here with you.",
    ],
    ResponseStrategy.RECALL: [
        "I remember from our previous conversation that {memory}. How has that been going?",
        "Last time we talked about {memory}. Would you like to continue that topic?",
    ],
    ResponseStrategy.SUGGEST: [
        "It might help to take a short break. Sometimes a little rest makes a big difference.",
        "Would you like to try a brief relaxation exercise? It can help when things feel heavy.",
    ],
    ResponseStrategy.FAREWELL: [
        "Take care! I'm always here when you need to talk.",
        "Goodbye for now. Remember, it's okay to come back anytime.",
    ],
}


class RuleBasedDialogue(DialogueProvider):
    """Simple rule-based dialogue provider using templates."""

    def __init__(self) -> None:
        self._turn_counter: int = 0

    _NEGATIVE_CUES = (
        "feeling low",
        "i feel low",
        "sad",
        "down",
        "depressed",
        "hopeless",
        "stressed",
        "stress",
        "anxious",
        "overwhelmed",
        "burned out",
        "tired",
        "lonely",
    )

    _RECALL_CUES = (
        "remember",
        "recall",
        "memory",
        "memories",
        "what do you remember",
        "what did we discuss",
        "last time",
        "previous",
    )

    _GREET_CUES = (
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
    )

    @property
    def backend_name(self) -> str:
        return "rule_based"

    @property
    def model_name(self) -> str:
        return ""

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        strategy: ResponseStrategy,
        state: InteractionState,
        memories: MemoryQueryResult,
        prompt_context: GroundedPromptContext | None = None,
    ) -> SupportiveResponse:
        latest_user_text = self._extract_latest_user_text(prompt_context)
        inferred = self._infer_intent_and_tone(latest_user_text)

        effective_strategy = strategy
        if inferred == "recall":
            effective_strategy = ResponseStrategy.RECALL
        elif inferred in {"negative", "stress"} and strategy in {
            ResponseStrategy.ENCOURAGE,
            ResponseStrategy.CHECK_IN,
            ResponseStrategy.SUGGEST,
        }:
            # Prevent cheerful invalidation for clearly negative user text.
            effective_strategy = ResponseStrategy.EMPATHIZE
        elif inferred == "greet" and strategy == ResponseStrategy.ENCOURAGE:
            effective_strategy = ResponseStrategy.GREET

        templates = _TEMPLATES.get(effective_strategy, _TEMPLATES[ResponseStrategy.ENCOURAGE])
        template = templates[self._turn_counter % len(templates)]
        self._turn_counter += 1

        memory_ref = self._select_memory_ref(memories, prompt_context)

        # Build a short, calm Baymax-inspired response for high-signal user text.
        if effective_strategy in {ResponseStrategy.ANSWER, ResponseStrategy.WEB_ANSWER}:
            message = self._build_limited_answer(latest_user_text)
        elif effective_strategy == ResponseStrategy.CLARIFY:
            message = template
        elif inferred in {"negative", "stress"}:
            message = self._build_supportive_negative(message_seed=template, user_text=latest_user_text, memory_ref=memory_ref)
        elif inferred == "recall":
            message = self._build_recall(memory_ref)
        else:
            message = template
            if "{memory}" in template:
                message = template.format(memory=memory_ref or "our previous conversation")

        return SupportiveResponse(
            session_id=state.session_id,
            user_id=state.user_id,
            strategy=effective_strategy,
            message=message,
            backend="rule_based",
            model_name="",
            limited_mode=True,
            limited_mode_reason="LLM unavailable; using deterministic templates",
        )

    def _extract_latest_user_text(self, prompt_context: GroundedPromptContext | None) -> str:
        if prompt_context is None:
            return ""

        # Prefer explicit latest USER turn when available.
        for turn in reversed(prompt_context.recent_turns):
            if str(turn.get("role", "")).lower() == "user":
                text = str(turn.get("text", "")).strip()
                if text:
                    return text

        context = (prompt_context.context or "").strip()
        if not context:
            return ""

        # Context often arrives as "The user typed: ..." or "The user said: ...".
        prefixes = (
            "the user typed:",
            "the user said:",
            "user typed:",
            "user said:",
        )
        lowered = context.lower()
        for prefix in prefixes:
            if lowered.startswith(prefix):
                return context[len(prefix):].strip()
        return context

    def _infer_intent_and_tone(self, user_text: str) -> str:
        if not user_text:
            return "neutral"

        lowered = re.sub(r"\s+", " ", user_text.lower()).strip()
        if any(cue in lowered for cue in self._RECALL_CUES):
            return "recall"
        if any(cue in lowered for cue in self._NEGATIVE_CUES):
            if "stress" in lowered or "stressed" in lowered or "overwhelmed" in lowered:
                return "stress"
            return "negative"
        if any(lowered.startswith(cue) or f" {cue}" in lowered for cue in self._GREET_CUES):
            return "greet"
        return "neutral"

    def _select_memory_ref(
        self,
        memories: MemoryQueryResult,
        prompt_context: GroundedPromptContext | None,
    ) -> str | None:
        if prompt_context and prompt_context.memory_refs:
            return prompt_context.memory_refs[0]
        if memories.episodic_memories:
            return memories.episodic_memories[0].content
        if memories.semantic_facts:
            return memories.semantic_facts[0].content
        return None

    def _build_supportive_negative(
        self,
        message_seed: str,
        user_text: str,
        memory_ref: str | None,
    ) -> str:
        if "stress" in user_text.lower() or "stressed" in user_text.lower() or "overwhelmed" in user_text.lower():
            base = "That sounds stressful. Thank you for telling me."
            follow = "We can take this one small step at a time."
        else:
            base = "I hear you. Feeling low can be heavy."
            follow = "You do not need to carry it alone right now."

        if memory_ref:
            return f"{base} {follow} I remember: {memory_ref}."

        # Keep wording calm and literal in Baymax-inspired style.
        if "difficult" in message_seed.lower() or "tough" in message_seed.lower():
            return f"{base} {message_seed}"
        return f"{base} {follow}"

    def _build_recall(self, memory_ref: str | None) -> str:
        if memory_ref:
            return (
                f"I remember this: {memory_ref}. "
                "Would you like to continue from there?"
            )
        return (
            "I do not have a clear memory to cite yet. "
            "If you share a detail now, I can keep it for later recall."
        )

    def _build_limited_answer(self, user_text: str) -> str:
        lower = user_text.lower()
        if "capital of france" in lower:
            return "Paris. I am in limited local mode right now, so I will keep answers concise."
        if "center a div" in lower and "css" in lower:
            return "Use flexbox: set the parent to display:flex; justify-content:center; align-items:center."
        return (
            "I am in limited local mode right now, so I cannot reliably answer broad factual questions. "
            "If you want, ask me to search the web and I will cite sources."
        )
