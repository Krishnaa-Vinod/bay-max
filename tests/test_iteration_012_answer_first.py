"""Iteration 012 answer-first behavior tests."""

from uuid import uuid4

from baymax.core.enums import EmotionLabel, EngagementLevel, PostureLabel, ResponseStrategy, TurnIntent
from baymax.planner.supportive_planner import SupportivePlanner
from baymax.schemas.memory import MemoryQueryResult
from baymax.state.models import InteractionState


def _state() -> InteractionState:
    return InteractionState(
        session_id=uuid4(),
        engagement_level=EngagementLevel.MEDIUM,
        dominant_emotion=EmotionLabel.NEUTRAL,
        posture=PostureLabel.UPRIGHT,
    )


def test_direct_question_routes_to_answer() -> None:
    planner = SupportivePlanner()
    strategy = planner.plan(
        _state(),
        MemoryQueryResult(user_id=uuid4(), query=""),
        context="What is the capital of France?",
        turn_intent=TurnIntent.DIRECT_QUESTION,
    )
    assert strategy == ResponseStrategy.ANSWER


def test_follow_up_routes_to_answer() -> None:
    planner = SupportivePlanner()
    strategy = planner.plan(
        _state(),
        MemoryQueryResult(user_id=uuid4(), query=""),
        context="why?",
        turn_intent=TurnIntent.FOLLOW_UP,
    )
    assert strategy == ResponseStrategy.ANSWER


def test_emotion_routes_to_empathize() -> None:
    planner = SupportivePlanner()
    strategy = planner.plan(
        _state(),
        MemoryQueryResult(user_id=uuid4(), query=""),
        context="I feel sad today.",
        turn_intent=TurnIntent.EMOTIONAL_SHARE,
    )
    assert strategy == ResponseStrategy.EMPATHIZE
