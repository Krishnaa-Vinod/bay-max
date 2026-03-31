"""Memory relevance gating tests."""

from baymax.core.enums import TurnIntent
from baymax.dialogue.prompt_builder import select_relevant_memories


def test_irrelevant_memory_not_forced_into_direct_answer() -> None:
    memories = [
        "User likes mountain biking.",
        "User has a dog named Bolt.",
    ]
    selected = select_relevant_memories(
        memories,
        context="What is the capital of France?",
        turn_intent=TurnIntent.DIRECT_QUESTION,
        threshold=0.2,
    )
    assert selected == []


def test_recall_intent_can_include_memories() -> None:
    memories = ["User likes mountain biking."]
    selected = select_relevant_memories(
        memories,
        context="What do you remember about me?",
        turn_intent=TurnIntent.RECALL,
    )
    assert selected == memories
