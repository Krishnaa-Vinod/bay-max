"""WebSocket endpoint for live runtime telemetry streaming."""

import asyncio
import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from baymax.live.schemas import LiveRuntimeStatus, SessionLifecycleState

logger = logging.getLogger(__name__)

router = APIRouter()

# Global state for the live telemetry
_live_runtime = None
_connected_clients: list[WebSocket] = []
_event_buffer: list[dict[str, Any]] = []
_max_event_buffer = 100


def set_live_runtime(runtime):
    """Set the live runtime reference for telemetry queries."""
    global _live_runtime
    _live_runtime = runtime


def get_live_runtime():
    """Get the current live runtime reference."""
    return _live_runtime


def add_pipeline_event(event: dict[str, Any]):
    """Add a pipeline event to the buffer for UI consumption."""
    global _event_buffer
    _event_buffer.append(event)
    if len(_event_buffer) > _max_event_buffer:
        _event_buffer = _event_buffer[-_max_event_buffer:]


def get_event_buffer() -> list[dict[str, Any]]:
    """Get the current event buffer."""
    return _event_buffer.copy()


def clear_event_buffer():
    """Clear the event buffer."""
    global _event_buffer
    _event_buffer = []


class TelemetryBroadcaster:
    """Manages WebSocket connections and broadcasts telemetry."""

    def __init__(self):
        self._clients: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        """Add a new WebSocket client."""
        await websocket.accept()
        async with self._lock:
            self._clients.append(websocket)
        logger.info("WebSocket client connected (total: %d)", len(self._clients))

    async def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket client."""
        async with self._lock:
            if websocket in self._clients:
                self._clients.remove(websocket)
        logger.info("WebSocket client disconnected (total: %d)", len(self._clients))

    async def broadcast(self, message: dict[str, Any]):
        """Broadcast a message to all connected clients."""
        if not self._clients:
            return

        disconnected = []
        async with self._lock:
            for client in self._clients:
                try:
                    await client.send_json(message)
                except Exception as e:
                    logger.warning("Failed to send to client: %s", e)
                    disconnected.append(client)

            for client in disconnected:
                if client in self._clients:
                    self._clients.remove(client)

    @property
    def client_count(self) -> int:
        return len(self._clients)


broadcaster = TelemetryBroadcaster()


def build_snapshot_message(
    status: LiveRuntimeStatus,
    memory_hits: list[dict] | None = None,
) -> dict[str, Any]:
    """Build a snapshot message for the WebSocket."""
    # Get cooldown state from scheduler if available
    cooldown_remaining = 0.0
    if _live_runtime is not None:
        scheduler = _live_runtime.scheduler
        cooldown_state = scheduler.cooldown_state
        if cooldown_state.last_any_response_at:
            elapsed = (datetime.utcnow() - cooldown_state.last_any_response_at).total_seconds()
            cooldown_remaining = max(0, scheduler._any_response_min_interval - elapsed)

    # Get session info
    session_id = None
    session_duration = 0.0
    turn_count = 0
    if _live_runtime is not None:
        session_id = _live_runtime.supervisor.current_session_id
        if session_id and _live_runtime._start_time > 0:
            import time
            session_duration = time.time() - _live_runtime._start_time

    return {
        "type": "snapshot",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "session": {
            "id": str(session_id) if session_id else None,
            "state": status.session_status.value,
            "duration_sec": round(session_duration, 1),
            "turn_count": turn_count,
        },
        "perception": {
            "face_detected": (
                status.current_user_id is not None
                or status.session_status == SessionLifecycleState.ACTIVE
            ),
            "user_name": status.current_user_display_name,
            "user_id": str(status.current_user_id) if status.current_user_id else None,
            "recognition_confidence": 0.0,  # TODO: track this in status
            "posture": "unknown",  # TODO: track in status
            "engagement": 0.5,  # TODO: track in status
            "valence": status.emotion_valence,
            "arousal": status.emotion_arousal,
            "affect_confidence": status.emotion_confidence,
            "affect_backend": status.affect_backend,
            "affect_stable_duration_sec": status.emotion_stable_duration_sec,
            "affect_debug": status.emotion_debug_summary,
        },
        "pipeline_state": _get_pipeline_state(status),
        "last_response": status.last_response_text or "",
        "last_spoken_text": status.last_spoken_text or "",
        "memory_hits": memory_hits or [],
        "cooldown_remaining_sec": round(cooldown_remaining, 1),
        "last_event": status.last_event or "",
        "frame_count": status.frame_count,
        "analysis_count": status.analysis_count,
        "uptime_sec": round(status.uptime_sec, 1),
        "speech_input": {
            "enabled": status.speech_input_enabled,
            "listening": status.listening,
            "vad_active": status.vad_active,
            "stt_backend": status.stt_backend,
            "speaking_lock": status.speaking_lock_active,
            "last_heard": status.last_heard_text or "",
            "mic_mode": status.mic_mode,
        },
        "tts": {
            "backend": status.tts_backend,
            "voice": status.tts_voice,
            "queue_depth": status.speech_queue_depth,
        },
    }


def _get_pipeline_state(status: LiveRuntimeStatus) -> str:
    """Determine the current pipeline state for the UI avatar."""
    if not status.live_mode_active:
        return "OFFLINE"
    if status.speaking_lock_active:
        return "SPEAKING"
    if status.vad_active:
        return "LISTENING"
    if status.session_status == SessionLifecycleState.IDLE:
        return "IDLE"
    if status.session_status == SessionLifecycleState.PAUSED:
        return "COOLDOWN"
    return "ACTIVE"


def build_event_message(
    stage: str,
    status: str,
    detail: str,
    event_type: str = "pipeline_event",
) -> dict[str, Any]:
    """Build a pipeline event message."""
    return {
        "type": event_type,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "stage": stage,
        "status": status,
        "detail": detail,
    }


@router.websocket("/ws/live")
async def websocket_live_telemetry(websocket: WebSocket):
    """WebSocket endpoint for live runtime telemetry.

    Sends:
    - snapshot: Full state snapshot every ~500ms
    - pipeline_event: Real-time pipeline stage events
    - speech_event: Speech input/output events
    - memory_event: Memory retrieval/storage events
    - error: Error notifications
    """
    await broadcaster.connect(websocket)

    try:
        # Send initial bootstrap state
        if _live_runtime is not None:
            status = _live_runtime.status
            snapshot = build_snapshot_message(status)
            snapshot["type"] = "bootstrap"
            await websocket.send_json(snapshot)

            # Send buffered events
            for event in get_event_buffer():
                await websocket.send_json(event)

        # Keep connection alive and send periodic snapshots
        while True:
            try:
                # Wait for incoming messages (for keep-alive pings)
                message = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=0.5,
                )
                # Handle ping/pong
                if message == "ping":
                    await websocket.send_text("pong")

            except TimeoutError:
                # No message received, send snapshot
                if _live_runtime is not None:
                    status = _live_runtime.status
                    snapshot = build_snapshot_message(status)
                    await websocket.send_json(snapshot)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning("WebSocket error: %s", e)
    finally:
        await broadcaster.disconnect(websocket)


async def broadcast_event(event: dict[str, Any]):
    """Broadcast an event to all connected WebSocket clients."""
    add_pipeline_event(event)
    await broadcaster.broadcast(event)
