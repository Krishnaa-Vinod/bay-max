"""Speech/text intent routing tests for iteration 012."""

from baymax.core.enums import TurnIntent
from baymax.dialogue.intent_router import classify_turn_intent


def test_router_classifies_direct_question() -> None:
    intent = classify_turn_intent("How do I center a div in CSS?")
    assert intent in {TurnIntent.DIRECT_QUESTION, TurnIntent.TASK_REQUEST}


def test_router_classifies_follow_up_from_context() -> None:
    recent = [
        {"role": "assistant", "text": "Paris is the capital of France."},
        {"role": "user", "text": "Okay"},
    ]
    intent = classify_turn_intent("why?", recent_turns=recent)
    assert intent == TurnIntent.FOLLOW_UP


def test_router_classifies_emotional_share() -> None:
    intent = classify_turn_intent("I feel anxious and stressed today")
    assert intent == TurnIntent.EMOTIONAL_SHARE


def test_router_classifies_unclear_short_text() -> None:
    intent = classify_turn_intent("uh")
    assert intent == TurnIntent.UNCLEAR
