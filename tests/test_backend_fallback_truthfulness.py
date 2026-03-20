"""Fallback truthfulness tests for limited dialogue mode."""

from uuid import uuid4

from baymax.core.enums import EmotionLabel, EngagementLevel, PostureLabel, ResponseStrategy
from baymax.dialogue.rule_based import RuleBasedDialogue
from baymax.schemas.memory import MemoryQueryResult
from baymax.state.models import InteractionState


def _state() -> InteractionState:
    return InteractionState(
        session_id=uuid4(),
        engagement_level=EngagementLevel.MEDIUM,
        dominant_emotion=EmotionLabel.NEUTRAL,
        posture=PostureLabel.UPRIGHT,
    )


def test_rule_based_marks_limited_mode() -> None:
    rb = RuleBasedDialogue()
    response = rb.generate(ResponseStrategy.ANSWER, _state(), MemoryQueryResult(user_id=uuid4(), query=""))
    assert response.limited_mode is True
    assert response.limited_mode_reason


def test_rule_based_does_not_pretend_broad_knowledge() -> None:
    rb = RuleBasedDialogue()
    response = rb.generate(ResponseStrategy.ANSWER, _state(), MemoryQueryResult(user_id=uuid4(), query=""))
    assert "limited" in response.message.lower()
