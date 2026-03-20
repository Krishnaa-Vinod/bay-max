"""Tests for Iteration 011 voice-first web assistant changes."""

import asyncio
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from baymax.live.schemas import LiveRuntimeStatus, VoiceMode
from baymax.orchestrator.service import Orchestrator
from baymax.tools.web_fetch import summarize_sources


def test_live_status_exposes_mode_and_web_tool_fields() -> None:
    status = LiveRuntimeStatus()
    assert status.voice_mode == VoiceMode.LOCAL_CHAINED
    assert status.speech_loop_state == "idle"
    assert status.web_tools_enabled is False
    assert status.web_tools_available is False


def test_web_tool_gate_detects_live_info_queries() -> None:
    orch = Orchestrator(db_path=":memory:")
    assert orch._should_use_web_tools("Search the latest Nvidia stock price") is True
    assert orch._should_use_web_tools("how are you") is False


def test_summarize_sources_compacts_items() -> None:
    text = summarize_sources([
        {"title": "Example", "url": "https://example.com", "snippet": "A summary."},
        {"title": "Another", "url": "https://example.org", "snippet": "Second summary."},
    ])
    assert "Example" in text
    assert "https://example.com" in text


def test_text_input_returns_tool_usage_and_sources() -> None:
    from apps.api import live_http

    user_id = uuid4()
    session_id = uuid4()

    runtime = MagicMock()
    runtime.supervisor.current_session_id = session_id
    runtime.status.current_user_id = user_id
    runtime._status.last_tts_result = ""
    runtime._status.last_tts_error = ""
    runtime._status.last_tts_wav_path = ""
    runtime._status.last_memory_refs = []
    runtime._status.last_tools_used = []
    runtime._status.last_web_sources = []
    runtime._status.turn_count = 0
    runtime.speech_service = None

    response_obj = SimpleNamespace(
        message="hello\n\nSources:\n- Example: https://example.com",
        memory_refs=["fact-a"],
        strategy=SimpleNamespace(value="greet"),
        backend="ollama",
        model_name="qwen2.5:1.5b",
        fallback_used=False,
        tool_usage=["search_web", "summarize_sources"],
        source_refs=[{"title": "Example", "url": "https://example.com", "snippet": "..."}],
    )

    orch = MagicMock()
    orch.add_turn = AsyncMock(return_value=None)
    orch.respond = AsyncMock(return_value=response_obj)

    with patch.object(live_http, "_live_runtime", runtime), patch.object(live_http, "_orchestrator", orch):
        payload = asyncio.run(live_http.submit_text_input(live_http.TextInputRequest(text="hi")))

    assert payload.success is True
    assert payload.modality == "text"
    assert payload.tool_usage == ["search_web", "summarize_sources"]
    assert payload.source_refs[0]["url"] == "https://example.com"


def test_mic_toggle_uses_runtime_safe_lifecycle() -> None:
    from apps.api import live_http

    speech_input = MagicMock()
    speech_input.status.return_value = SimpleNamespace(listening=True, mic_mode="vad")

    runtime = MagicMock()
    runtime.speech_input_service = speech_input
    runtime.start_microphone = AsyncMock(return_value=(True, ""))
    runtime.stop_microphone = AsyncMock(return_value=None)
    runtime.status = SimpleNamespace(speech_loop_state="listening", speech_input_enabled=True)

    with patch.object(live_http, "_live_runtime", runtime):
        result = asyncio.run(live_http.toggle_microphone(live_http.MicToggleRequest(action="start")))

    assert result.success is True
    runtime.start_microphone.assert_awaited_once()
