"""Artifact logger for live run sessions."""

import json
import logging
import os
from datetime import datetime
from uuid import UUID

from baymax.live.schemas import (
    CompanionEvent,
    LiveArtifactRef,
    LiveRunSummary,
    ProactiveDecision,
    SessionLifecycleEvent,
)

logger = logging.getLogger(__name__)


class _JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, UUID):
            return str(o)
        return super().default(o)


class ArtifactLogger:
    """Logs events, responses, session transitions, and snapshots to artifact files."""

    def __init__(self, artifact_dir: str, enabled: bool = True) -> None:
        self._dir = artifact_dir
        self._enabled = enabled
        self._events_path = os.path.join(artifact_dir, "events.jsonl")
        self._responses_path = os.path.join(artifact_dir, "responses.jsonl")
        self._session_timeline_path = os.path.join(artifact_dir, "session_timeline.json")
        self._manifest_path = os.path.join(artifact_dir, "manifest.json")
        self._summary_path = os.path.join(artifact_dir, "summary.md")
        self._session_events: list[dict] = []
        self._artifacts: list[LiveArtifactRef] = []

        if enabled:
            os.makedirs(artifact_dir, exist_ok=True)
            logger.info("Artifact logging enabled at %s", artifact_dir)

    @property
    def artifact_dir(self) -> str:
        return self._dir

    @property
    def artifacts(self) -> list[LiveArtifactRef]:
        return list(self._artifacts)

    def log_event(self, event: CompanionEvent) -> None:
        if not self._enabled:
            return
        entry = event.model_dump(mode="json")
        with open(self._events_path, "a") as f:
            f.write(json.dumps(entry, cls=_JSONEncoder) + "\n")

    def log_suppression(self, decision: ProactiveDecision) -> None:
        if not self._enabled:
            return
        entry = {
            "type": "suppression",
            "event_type": decision.event.event_type.value,
            "timestamp": datetime.utcnow().isoformat(),
            "reason": decision.suppression_reason,
            "cooldown_remaining_sec": decision.cooldown_remaining_sec,
        }
        with open(self._events_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_response(
        self,
        event: CompanionEvent,
        response_text: str,
        strategy: str,
        memory_refs: list[str] | None = None,
        backend: str = "",
        trigger_reason: str = "",
    ) -> None:
        if not self._enabled:
            return
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event.event_type.value,
            "trigger_reason": trigger_reason,
            "session_id": str(event.session_id) if event.session_id else None,
            "user_id": str(event.user_id) if event.user_id else None,
            "strategy": strategy,
            "response_text": response_text,
            "memory_refs": memory_refs or [],
            "backend": backend,
        }
        with open(self._responses_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_session_event(self, event: SessionLifecycleEvent) -> None:
        if not self._enabled:
            return
        entry = event.model_dump(mode="json")
        self._session_events.append(entry)

    def save_snapshot(
        self,
        frame,
        prefix: str = "snapshot",
        session_id: UUID | None = None,
    ) -> str | None:
        """Save an annotated frame snapshot. Returns the saved path or None."""
        if not self._enabled:
            return None
        try:
            import cv2
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{ts}.jpg"
            path = os.path.join(self._dir, filename)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Convert RGB to BGR for cv2
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            cv2.imwrite(path, frame_bgr)
            ref = LiveArtifactRef(
                path=path,
                artifact_type="snapshot",
                session_id=session_id,
            )
            self._artifacts.append(ref)
            return path
        except Exception:
            logger.warning("Failed to save snapshot", exc_info=True)
            return None

    def write_session_timeline(self) -> None:
        if not self._enabled:
            return
        with open(self._session_timeline_path, "w") as f:
            json.dump(self._session_events, f, indent=2, cls=_JSONEncoder)
        ref = LiveArtifactRef(
            path=self._session_timeline_path,
            artifact_type="session_timeline",
        )
        self._artifacts.append(ref)

    def write_manifest(self, summary: LiveRunSummary) -> None:
        if not self._enabled:
            return

        manifest = {
            "run_id": str(summary.run_id),
            "source": summary.source,
            "started_at": summary.started_at.isoformat(),
            "ended_at": summary.ended_at.isoformat() if summary.ended_at else None,
            "total_frames": summary.total_frames,
            "total_analyses": summary.total_analyses,
            "total_events": summary.total_events,
            "total_responses": summary.total_responses,
            "total_suppressions": summary.total_suppressions,
            "sessions_created": summary.sessions_created,
            "sessions_resumed": summary.sessions_resumed,
            "users_recognized": summary.users_recognized,
            "backend": summary.backend,
            "model_name": summary.model_name,
            "errors": summary.errors,
            "artifacts": [
                {"path": a.path, "type": a.artifact_type}
                for a in self._artifacts
            ],
        }

        with open(self._manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        self._artifacts.append(LiveArtifactRef(
            path=self._manifest_path,
            artifact_type="manifest",
        ))

    def write_summary(self, summary: LiveRunSummary) -> None:
        if not self._enabled:
            return

        lines = [
            "# Bay-Max Live Run Summary",
            "",
            f"**Run ID**: {summary.run_id}",
            f"**Source**: {summary.source}",
            f"**Started**: {summary.started_at.isoformat()}",
            f"**Ended**: {summary.ended_at.isoformat() if summary.ended_at else 'N/A'}",
            "",
            "## Counts",
            f"- Frames processed: {summary.total_frames}",
            f"- Analysis frames: {summary.total_analyses}",
            f"- Events detected: {summary.total_events}",
            f"- Responses generated: {summary.total_responses}",
            f"- Suppressions (cooldown): {summary.total_suppressions}",
            f"- Sessions created: {summary.sessions_created}",
            f"- Sessions resumed: {summary.sessions_resumed}",
            "",
            "## Users Recognized",
        ]
        for u in summary.users_recognized:
            lines.append(f"- {u}")
        if not summary.users_recognized:
            lines.append("- (none)")

        lines.extend([
            "",
            "## Backend",
            f"- Backend: {summary.backend}",
            f"- Model: {summary.model_name}",
            "",
            "## Artifacts",
        ])
        for a in self._artifacts:
            lines.append(f"- `{a.path}` ({a.artifact_type})")

        if summary.errors:
            lines.extend(["", "## Errors"])
            for e in summary.errors:
                lines.append(f"- {e}")

        with open(self._summary_path, "w") as f:
            f.write("\n".join(lines) + "\n")

        self._artifacts.append(LiveArtifactRef(
            path=self._summary_path,
            artifact_type="summary",
        ))
