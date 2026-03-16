"""Rule-based dialogue provider for iteration-001."""

from baymax.core.enums import ResponseStrategy
from baymax.dialogue.interfaces import DialogueProvider
from baymax.schemas.memory import MemoryQueryResult
from baymax.schemas.response import SupportiveResponse
from baymax.state.models import InteractionState

_TEMPLATES: dict[ResponseStrategy, list[str]] = {
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

    def generate(
        self,
        strategy: ResponseStrategy,
        state: InteractionState,
        memories: MemoryQueryResult,
    ) -> SupportiveResponse:
        templates = _TEMPLATES.get(strategy, _TEMPLATES[ResponseStrategy.ENCOURAGE])
        template = templates[self._turn_counter % len(templates)]
        self._turn_counter += 1

        # Fill in memory reference if available
        message = template
        if "{memory}" in template and memories.total_count > 0:
            if memories.episodic_memories:
                memory_ref = memories.episodic_memories[0].content
            elif memories.semantic_facts:
                memory_ref = memories.semantic_facts[0].content
            else:
                memory_ref = "our previous conversation"
            message = template.format(memory=memory_ref)

        return SupportiveResponse(
            session_id=state.session_id,
            user_id=state.user_id,
            strategy=strategy,
            message=message,
        )
