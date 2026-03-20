"""Deterministic intent routing for user turns.

This module keeps first-pass intent mapping predictable and cheap.
"""

from __future__ import annotations

import re

from baymax.core.enums import TurnIntent

_RECALL_CUES = (
    "remember",
    "recall",
    "memory",
    "memories",
    "what do you remember",
    "last time",
    "previously",
)

_EMOTION_CUES = (
    "i feel sad",
    "i am sad",
    "feeling sad",
    "sad today",
    "i feel anxious",
    "i am anxious",
    "anxious",
    "stressed",
    "stress",
    "frustrated",
    "overwhelmed",
    "burned out",
)

_GREETING_CUES = (
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
)

_ACK_CUES = (
    "ok",
    "okay",
    "got it",
    "thanks",
    "thank you",
    "cool",
    "sure",
)

_FOLLOW_UP_CUES = (
    "why",
    "how",
    "what do you mean",
    "what about that",
    "and then",
    "can you explain",
)

_TASK_CUES = (
    "show me",
    "give me",
    "help me",
    "write",
    "build",
    "fix",
    "debug",
    "steps",
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _is_follow_up(text: str, recent_turns: list[dict]) -> bool:
    if not recent_turns:
        return False
    for cue in _FOLLOW_UP_CUES:
        if text == cue or text.startswith(f"{cue}?") or text.startswith(f"{cue} "):
            return True
    return text in {"and?", "then?", "why?", "how?"}


def classify_turn_intent(context: str, recent_turns: list[dict] | None = None) -> TurnIntent:
    """Return a deterministic turn intent for the latest user text."""
    recent_turns = recent_turns or []
    text = _normalize(context)

    if not text:
        return TurnIntent.UNCLEAR

    if len(text) <= 2:
        return TurnIntent.UNCLEAR

    if any(cue in text for cue in _RECALL_CUES):
        return TurnIntent.RECALL

    if any(cue in text for cue in _EMOTION_CUES):
        return TurnIntent.EMOTIONAL_SHARE

    if _is_follow_up(text, recent_turns):
        return TurnIntent.FOLLOW_UP

    if any(text == cue or text.startswith(f"{cue} ") for cue in _GREETING_CUES):
        return TurnIntent.GREETING

    if text in _ACK_CUES:
        return TurnIntent.ACKNOWLEDGMENT

    if "?" in text:
        if any(cue in text for cue in _TASK_CUES):
            return TurnIntent.TASK_REQUEST
        return TurnIntent.DIRECT_QUESTION

    if any(cue in text for cue in _TASK_CUES):
        return TurnIntent.TASK_REQUEST

    # Short utterances without signal are usually unclear.
    if len(text.split()) <= 2:
        return TurnIntent.UNCLEAR

    if any(token in text for token in ("weather", "music", "movie", "weekend")):
        return TurnIntent.SMALLTALK

    return TurnIntent.SMALLTALK
