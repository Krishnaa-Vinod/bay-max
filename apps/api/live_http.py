"""HTTP endpoints for live UI support."""

import logging
from datetime import datetime
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from baymax.core.enums import TurnRole

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/live", tags=["live"])

# References set externally
_live_runtime = None
_orchestrator = None
_last_annotated_frame: bytes | None = None
_last_frame_timestamp: datetime | None = None


def set_live_runtime(runtime):
    """Set the live runtime reference."""
    global _live_runtime
    _live_runtime = runtime


def set_orchestrator(orchestrator):
    """Set the orchestrator reference."""
    global _orchestrator
    _orchestrator = orchestrator


def update_annotated_frame(frame: np.ndarray):
    """Update the cached annotated frame."""
    global _last_annotated_frame, _last_frame_timestamp
    try:
        # Convert RGB to BGR for OpenCV encoding
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Encode as JPEG
        success, encoded = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if success:
            _last_annotated_frame = encoded.tobytes()
            _last_frame_timestamp = datetime.utcnow()
    except Exception as e:
        logger.warning("Failed to encode annotated frame: %s", e)


class UIBootstrapState(BaseModel):
    """Bootstrap state for the companion UI."""

    live_mode_active: bool = False
    source: str = ""
    voice_mode: str = "local_chained_voice"
    voice_mode_requested: str = "local_chained"
    voice_fallback_reason: str = ""
    speech_loop_state: str = "idle"
    session_id: str | None = None
    session_state: str = "idle"
    user_id: str | None = None
    user_name: str | None = None

    # Backend info
    dialogue_backend: str = ""
    dialogue_model: str = ""
    dialogue_requested_backend: str = ""
    dialogue_requested_model: str = ""
    dialogue_fallback_warning: str = ""
    tts_backend: str = ""
    tts_voice: str = ""
    tts_voice_preset: str = ""
    tts_last_result: str = ""
    tts_last_error: str = ""
    tts_last_wav_path: str = ""
    tts_last_playback_ok: bool | None = None
    stt_backend: str = ""
    affect_backend: str = ""

    # Feature flags
    speech_input_enabled: bool = False
    speech_disabled_reason: str = ""
    proactive_mode: str = "affect_only"
    web_tools_enabled: bool = False
    web_tools_available: bool = False
    web_tools_disabled_reason: str = ""
    affect_enabled: bool = False
    tts_enabled: bool = False
    transcript_quality_score: float = 0.0
    transcript_quality_reason: str = ""
    emotion_valence: float = 0.0
    emotion_arousal: float = 0.0
    emotion_confidence: float = 0.0
    emotion_debug_summary: str = ""

    # Recognition/session diagnostics
    face_detected: bool = False
    recognition_state: str = "no_face"
    face_match_threshold: float = 0.0
    session_binding: str = "anonymous"

    # Stats
    frame_count: int = 0
    analysis_count: int = 0
    turn_count: int = 0
    uptime_sec: float = 0.0
    session_start_to_ready_ms: float = 0.0
    end_of_speech_to_first_audio_ms: float = 0.0
    interrupt_to_audio_stop_ms: float = 0.0

    # Last known state
    last_response: str = ""
    last_event: str = ""

    # Timestamps
    server_time: str = ""


class TextInputRequest(BaseModel):
    """Request to submit text input."""

    text: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = None


class TextInputResponse(BaseModel):
    """Response from text input submission."""

    success: bool
    session_id: str | None = None
    response_text: str = ""
    memory_refs: list[dict] = Field(default_factory=list)
    strategy: str = ""
    backend: str = ""
    model_name: str = ""
    fallback_used: bool = False
    tts_result: str = ""
    tts_error: str = ""
    tts_wav_path: str = ""
    modality: str = "text"
    tool_usage: list[str] = Field(default_factory=list)
    source_refs: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


class MicToggleRequest(BaseModel):
    """Request to toggle microphone state."""

    action: str = Field(..., description="Action: 'start', 'stop', or 'toggle'")


class MicToggleResponse(BaseModel):
    """Response from mic toggle."""

    success: bool
    listening: bool = False
    mic_mode: str = ""
    disabled_reason: str = ""
    speech_loop_state: str = "idle"
    error: str | None = None


class MemoryStats(BaseModel):
    """Memory stats for UI panel."""

    total_memories: int = 0
    total_facts: int = 0
    total_sessions: int = 0


class MemoryRecentResponse(BaseModel):
    """Recent memory payload for UI panel."""

    memories: list[dict[str, Any]] = Field(default_factory=list)
    facts: list[dict[str, Any]] = Field(default_factory=list)
    stats: MemoryStats = Field(default_factory=MemoryStats)
    empty_reason: str = ""
    recognition_reason: str = ""
    memory_unavailable_hint: str = ""
    error: str | None = None


def _resolve_speech_disabled_reason() -> str:
    """Return precise reason when speech input is unavailable."""
    if _live_runtime is None:
        return "Live runtime not active"

    status = _live_runtime.status
    if status.speech_input_enabled:
        return ""

    if status.speech_disabled_reason:
        return status.speech_disabled_reason

    return "Speech input unavailable"


def _resolve_memory_unavailable_reason() -> tuple[str, str]:
    """Return explicit recognition reason and memory availability hint."""
    hint = "Memory unavailable until an enrolled user is recognized."
    if _live_runtime is None:
        return ("live runtime not active", hint)

    status = _live_runtime.status
    state = status.recognition_state

    if state == "no_face":
        return ("no face detected", hint)
    if state == "face_seen_unknown":
        return ("face detected, unknown user", hint)
    if state == "known_low_confidence":
        conf = status.recognition_confidence
        if conf is not None:
            return (
                f"recognized below threshold ({conf:.2f} < {status.face_match_threshold:.2f})",
                hint,
            )
        return ("recognized below threshold", hint)
    if state == "known_attached":
        return ("recognized enrolled user", "")
    return ("no active user", hint)


def _display_text(response: Any) -> str:
    display = getattr(response, "display_text", None)
    if isinstance(display, str) and display.strip():
        return display
    message = getattr(response, "message", "")
    return message if isinstance(message, str) else str(message)


def _spoken_text(response: Any) -> str:
    spoken = getattr(response, "spoken_text", None)
    if isinstance(spoken, str) and spoken.strip():
        return spoken
    return _display_text(response)


@router.get("/frame/latest")
async def get_latest_frame():
    """Return the latest annotated frame as JPEG.

    Returns a real annotated frame from the live runtime, or a placeholder
    if no frame is available.
    """
    global _last_annotated_frame

    if _last_annotated_frame is not None:
        return Response(
            content=_last_annotated_frame,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "X-Frame-Timestamp": (
                    _last_frame_timestamp.isoformat() if _last_frame_timestamp else ""
                ),
            },
        )

    # Generate a placeholder frame
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        placeholder,
        "No frame available",
        (180, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (128, 128, 128),
        2,
    )
    cv2.putText(
        placeholder,
        "Start live runtime to see video",
        (120, 280),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (100, 100, 100),
        1,
    )

    success, encoded = cv2.imencode(".jpg", placeholder, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if success:
        return Response(
            content=encoded.tobytes(),
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "X-Frame-Status": "placeholder",
            },
        )

    raise HTTPException(status_code=500, detail="Failed to generate frame")


@router.get("/ui-state", response_model=UIBootstrapState)
async def get_ui_state() -> UIBootstrapState:
    """Return bootstrap state for the UI.

    This endpoint provides all the information the UI needs to initialize
    when it loads after the runtime has already started.
    """
    state = UIBootstrapState(server_time=datetime.utcnow().isoformat() + "Z")

    if _live_runtime is not None:
        status = _live_runtime.status
        state.live_mode_active = status.live_mode_active
        state.source = status.source
        state.voice_mode = status.voice_mode.value
        state.voice_mode_requested = status.voice_mode_requested
        state.voice_fallback_reason = status.voice_fallback_reason
        state.speech_loop_state = status.speech_loop_state
        state.session_state = status.session_status.value
        state.user_id = str(status.current_user_id) if status.current_user_id else None
        state.user_name = status.current_user_display_name

        # Backend info
        state.dialogue_backend = status.dialogue_backend
        state.dialogue_model = status.dialogue_model
        state.dialogue_requested_backend = status.dialogue_requested_backend
        state.dialogue_requested_model = status.dialogue_requested_model
        state.dialogue_fallback_warning = status.dialogue_fallback_warning

        state.tts_backend = status.tts_backend
        state.tts_voice = status.tts_voice
        state.tts_voice_preset = status.tts_voice_preset
        state.tts_last_result = status.last_tts_result
        state.tts_last_error = status.last_tts_error
        state.tts_last_wav_path = status.last_tts_wav_path
        state.tts_last_playback_ok = status.last_tts_playback_ok
        state.stt_backend = status.stt_backend
        state.affect_backend = status.affect_backend

        # Feature flags
        state.speech_input_enabled = status.speech_input_enabled
        state.speech_disabled_reason = status.speech_disabled_reason
        state.proactive_mode = status.proactive_mode
        state.web_tools_enabled = status.web_tools_enabled
        state.web_tools_available = status.web_tools_available
        state.web_tools_disabled_reason = status.web_tools_disabled_reason
        state.affect_enabled = status.affect_enabled
        state.tts_enabled = bool(status.tts_backend and status.tts_backend != "null")
        state.transcript_quality_score = status.transcript_quality_score
        state.transcript_quality_reason = status.transcript_quality_reason
        state.emotion_valence = status.emotion_valence
        state.emotion_arousal = status.emotion_arousal
        state.emotion_confidence = status.emotion_confidence
        state.emotion_debug_summary = status.emotion_debug_summary

        # Recognition/session diagnostics
        state.face_detected = status.face_detected
        state.recognition_state = status.recognition_state
        state.face_match_threshold = status.face_match_threshold
        state.session_binding = status.session_binding

        # Stats
        state.frame_count = status.frame_count
        state.analysis_count = status.analysis_count
        state.turn_count = status.turn_count
        state.uptime_sec = status.uptime_sec
        state.session_start_to_ready_ms = status.session_start_to_ready_ms
        state.end_of_speech_to_first_audio_ms = status.end_of_speech_to_first_audio_ms
        state.interrupt_to_audio_stop_ms = status.interrupt_to_audio_stop_ms

        # Last known state
        state.last_response = status.last_response_text or ""
        state.last_event = status.last_event or ""

        # Session ID
        session_id = _live_runtime.supervisor.current_session_id
        state.session_id = str(session_id) if session_id else None

    elif _orchestrator is not None:
        state.dialogue_backend = _orchestrator.dialogue.backend_name
        state.dialogue_model = _orchestrator.dialogue.model_name
        state.dialogue_requested_backend = _orchestrator.dialogue_requested_backend
        state.dialogue_requested_model = _orchestrator.dialogue_requested_model
        state.dialogue_fallback_warning = _orchestrator.dialogue_fallback_warning

    return state


@router.post("/text-input", response_model=TextInputResponse)
async def submit_text_input(request: TextInputRequest) -> TextInputResponse:
    """Submit typed text as a user turn in the current session.

    This allows the UI to send text that gets processed through the
    full dialogue pipeline, just like speech input.
    """
    if _live_runtime is None:
        return TextInputResponse(
            success=False,
            error="Live runtime not active",
        )

    if _orchestrator is None:
        return TextInputResponse(
            success=False,
            error="Orchestrator not available",
        )

    # Get current session
    session_id = _live_runtime.supervisor.current_session_id
    user_id = _live_runtime.status.current_user_id

    if session_id is None:
        try:
            created = await _orchestrator.create_session(user_id=user_id)
            _live_runtime.supervisor.attach_active_session(created.id, user_id)
            session_id = created.id
            logger.info("Created session %s from text input", session_id)
        except Exception as e:
            logger.error("Failed to auto-create session for text input: %s", e)
            return TextInputResponse(
                success=False,
                error=f"No active session and failed to create one: {e}",
            )

    try:
        # Add user turn
        await _orchestrator.add_turn(
            session_id=session_id,
            role=TurnRole.USER,
            text=request.text,
            user_id=user_id,
            source="ui_text",
        )
        _live_runtime._status.turn_count += 1
        _live_runtime.scheduler.note_user_turn()

        # Generate response
        response = await _orchestrator.respond(
            session_id=session_id,
            user_id=user_id,
            context=request.text,
            modality="text",
        )

        # Store system response
        await _orchestrator.add_turn(
            session_id=session_id,
            role=TurnRole.SYSTEM,
            text=_display_text(response),
            user_id=user_id,
            source="ui_text_reply",
        )
        _live_runtime._status.turn_count += 1
        _live_runtime.scheduler.note_assistant_turn()

        # Update runtime status
        _live_runtime._status.last_response_text = _display_text(response)
        _live_runtime._status.last_response_at = datetime.utcnow()
        # Track memory refs for UI (iteration 010b hotfix)
        refs = response.memory_refs[:10] if response.memory_refs else []
        _live_runtime._status.last_memory_refs = refs
        _live_runtime._status.last_tools_used = list(response.tool_usage)
        _live_runtime._status.last_web_sources = list(response.source_refs)

        # Optionally speak the response
        if _live_runtime.speech_service is not None:
            try:
                await _live_runtime._speak_response(
                    _spoken_text(response),
                    session_id,
                    "ui_text_input",
                )
            except Exception as e:
                logger.warning("TTS failed for text input response: %s", e)
                _live_runtime._status.last_tts_result = "runtime_error"
                _live_runtime._status.last_tts_error = str(e)

                from apps.api.live_ws import broadcast_event, build_event_message

                await broadcast_event(build_event_message(
                    stage="tts",
                    status="error",
                    detail=f"TTS runtime error: {e}",
                    event_type="speech_event",
                ))

        # Broadcast event
        from apps.api.live_ws import broadcast_event, build_event_message

        await broadcast_event(build_event_message(
            stage="dialogue",
            status="complete",
            detail=f"Response: {response.message[:50]}...",
        ))

        await broadcast_event(build_event_message(
            stage="memory",
            status="complete",
            detail=f"Retrieved {len(refs)} memory hit(s) for response",
            event_type="memory_event",
        ))

        return TextInputResponse(
            success=True,
            session_id=str(session_id),
            response_text=_display_text(response),
            memory_refs=[
                {"text": ref, "score": None} for ref in response.memory_refs
            ],
            strategy=response.strategy.value if response.strategy else "",
            backend=response.backend,
            model_name=response.model_name,
            fallback_used=response.fallback_used,
            tts_result=_live_runtime._status.last_tts_result,
            tts_error=_live_runtime._status.last_tts_error,
            tts_wav_path=_live_runtime._status.last_tts_wav_path,
            modality="text",
            tool_usage=response.tool_usage,
            source_refs=response.source_refs,
        )

    except Exception as e:
        logger.error("Text input processing failed: %s", e)
        return TextInputResponse(
            success=False,
            error=str(e),
        )


@router.post("/mic/toggle", response_model=MicToggleResponse)
async def toggle_microphone(request: MicToggleRequest) -> MicToggleResponse:
    """Toggle the microphone state for speech input.

    Actions:
    - 'start': Enable listening
    - 'stop': Disable listening
    - 'toggle': Toggle current state
    """
    if _live_runtime is None:
        return MicToggleResponse(
            success=False,
            disabled_reason="Live runtime not active",
            error="Live runtime not active",
        )

    speech_input = _live_runtime.speech_input_service
    if speech_input is None:
        return MicToggleResponse(
            success=False,
            disabled_reason=_resolve_speech_disabled_reason(),
            error="Speech input not enabled",
        )

    try:
        current_listening = speech_input.status().listening

        if request.action == "start":
            ok, reason = await _live_runtime.start_microphone()
            new_listening = ok and speech_input.status().listening
            if not ok:
                return MicToggleResponse(
                    success=False,
                    listening=False,
                    mic_mode=speech_input.status().mic_mode,
                    disabled_reason=reason,
                    speech_loop_state=_live_runtime.status.speech_loop_state,
                    error=f"Unable to start speech input: {reason}",
                )
        elif request.action == "stop":
            if current_listening:
                await _live_runtime.stop_microphone()
            new_listening = False
        elif request.action == "toggle":
            if current_listening:
                await _live_runtime.stop_microphone()
                new_listening = False
            else:
                ok, reason = await _live_runtime.start_microphone()
                new_listening = ok and speech_input.status().listening
                if not ok:
                    return MicToggleResponse(
                        success=False,
                        listening=False,
                        mic_mode=speech_input.status().mic_mode,
                        disabled_reason=reason,
                        speech_loop_state=_live_runtime.status.speech_loop_state,
                        error=f"Unable to start speech input: {reason}",
                    )
        else:
            return MicToggleResponse(
                success=False,
                error=f"Unknown action: {request.action}",
            )

        if not new_listening and request.action in ("start", "toggle"):
            reason = _resolve_speech_disabled_reason()
            return MicToggleResponse(
                success=False,
                listening=False,
                mic_mode=speech_input.status().mic_mode,
                disabled_reason=reason,
                speech_loop_state=_live_runtime.status.speech_loop_state,
                error=f"Unable to start speech input: {reason}",
            )

        # Broadcast event
        from apps.api.live_ws import broadcast_event, build_event_message

        await broadcast_event(build_event_message(
            stage="speech_input",
            status="toggled",
            detail=f"Listening: {new_listening}",
            event_type="speech_event",
        ))

        return MicToggleResponse(
            success=True,
            listening=new_listening,
            mic_mode=speech_input.status().mic_mode,
            disabled_reason="",
            speech_loop_state=_live_runtime.status.speech_loop_state,
        )

    except Exception as e:
        logger.error("Mic toggle failed: %s", e)
        return MicToggleResponse(
            success=False,
            disabled_reason=_resolve_speech_disabled_reason(),
            speech_loop_state=_live_runtime.status.speech_loop_state,
            error=str(e),
        )


@router.post("/voice/interrupt")
async def interrupt_voice() -> dict[str, Any]:
    """Interrupt current assistant speech and return to listening."""
    if _live_runtime is None:
        return {"success": False, "reason": "Live runtime not active"}

    await _live_runtime.interrupt_speaking()
    return {
        "success": True,
        "speech_loop_state": _live_runtime.status.speech_loop_state,
        "interrupt_to_audio_stop_ms": _live_runtime.status.interrupt_to_audio_stop_ms,
    }


@router.get("/memory/recent", response_model=MemoryRecentResponse)
async def get_recent_memories() -> MemoryRecentResponse:
    """Get recent memory hits for display in the UI."""
    if _orchestrator is None:
        return MemoryRecentResponse(
            empty_reason="memory service error",
            error="Orchestrator not available",
        )

    if _live_runtime is None:
        reason, hint = _resolve_memory_unavailable_reason()
        return MemoryRecentResponse(empty_reason="no active user", recognition_reason=reason, memory_unavailable_hint=hint)

    user_id = _live_runtime.status.current_user_id
    if user_id is None:
        reason, hint = _resolve_memory_unavailable_reason()
        return MemoryRecentResponse(
            empty_reason="no active user",
            recognition_reason=reason,
            memory_unavailable_hint=hint,
        )

    try:
        # Get memory summary
        summary = await _orchestrator.get_memory_summary(user_id)
        memories = [
            {
                "text": m.content,
                "type": "episodic",
                "created_at": m.timestamp.isoformat(),
            }
            for m in (summary.recent_episodic or [])[:10]
        ]
        facts = [
            {"key": "fact", "value": f.content, "confidence": f.confidence}
            for f in (summary.confirmed_facts or [])[:10]
        ]
        empty_reason = "" if memories else "no retrieved memories yet"
        recognition_reason, hint = _resolve_memory_unavailable_reason()

        return MemoryRecentResponse(
            memories=memories,
            facts=facts,
            stats=MemoryStats(
                total_memories=summary.episodic_count,
                total_facts=summary.semantic_count,
                total_sessions=len(summary.session_summaries),
            ),
            empty_reason=empty_reason,
            recognition_reason=recognition_reason,
            memory_unavailable_hint=hint,
        )
    except Exception as e:
        logger.warning("Failed to get recent memories: %s", e)
        return MemoryRecentResponse(
            empty_reason="memory service error",
            error=str(e),
        )


@router.get("/memory/debug/inspect/{user_id}", response_model=MemoryRecentResponse)
async def debug_inspect_user_memory(user_id: str) -> MemoryRecentResponse:
    """DEBUG ONLY: inspect memory for a selected enrolled user.

    This endpoint is intended for local debugging when live recognition is flaky.
    """
    if _orchestrator is None:
        return MemoryRecentResponse(
            empty_reason="memory service error",
            error="Orchestrator not available",
            recognition_reason="DEBUG ONLY",
        )

    try:
        from uuid import UUID

        parsed = UUID(user_id)
    except Exception:
        return MemoryRecentResponse(
            empty_reason="memory service error",
            error=f"Invalid user_id: {user_id}",
            recognition_reason="DEBUG ONLY",
        )

    user = await _orchestrator.get_user(parsed)
    if user is None:
        return MemoryRecentResponse(
            empty_reason="no active user",
            error=f"User not found: {user_id}",
            recognition_reason="DEBUG ONLY",
            memory_unavailable_hint="Select an enrolled user for debug inspection.",
        )

    try:
        summary = await _orchestrator.get_memory_summary(parsed)
        memories = [
            {
                "text": m.content,
                "type": "episodic",
                "created_at": m.timestamp.isoformat(),
            }
            for m in (summary.recent_episodic or [])[:20]
        ]
        facts = [
            {"key": "fact", "value": f.content, "confidence": f.confidence}
            for f in (summary.confirmed_facts or [])[:20]
        ]
        return MemoryRecentResponse(
            memories=memories,
            facts=facts,
            stats=MemoryStats(
                total_memories=summary.episodic_count,
                total_facts=summary.semantic_count,
                total_sessions=len(summary.session_summaries),
            ),
            empty_reason="" if (memories or facts) else "no retrieved memories yet",
            recognition_reason="DEBUG ONLY: selected enrolled user",
            memory_unavailable_hint="Bypasses live recognition state for local diagnostics.",
        )
    except Exception as e:
        return MemoryRecentResponse(
            empty_reason="memory service error",
            error=str(e),
            recognition_reason="DEBUG ONLY",
        )
