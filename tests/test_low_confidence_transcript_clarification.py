"""Low-confidence transcript quality gate tests."""

from types import SimpleNamespace

from baymax.live.runtime import LiveRuntime


def test_rejects_short_low_confidence_transcript() -> None:
    runtime = LiveRuntime(db_path=":memory:")
    result = SimpleNamespace(text="uh", confidence=0.1)
    accepted, score, reason = runtime._assess_transcript_quality(result)
    assert accepted is False
    assert score < 0.55
    assert reason


def test_accepts_clear_transcript() -> None:
    runtime = LiveRuntime(db_path=":memory:")
    result = SimpleNamespace(text="How do I center a div in CSS?", confidence=0.9)
    accepted, score, _ = runtime._assess_transcript_quality(result)
    assert accepted is True
    assert score >= 0.55
