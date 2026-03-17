"""Main live runtime that ties all components together."""

import asyncio
import logging
import signal
import time
from datetime import datetime
from uuid import UUID

from baymax.config.settings import get_settings
from baymax.core.enums import EngagementLevel, PostureLabel, TurnRole
from baymax.live.artifact_logger import ArtifactLogger
from baymax.live.event_engine import EventEngine
from baymax.live.frame_source import FolderFrameSource, OpenCVFrameSource
from baymax.live.overlay import render_overlay
from baymax.live.proactive_scheduler import ProactiveScheduler
from baymax.live.schemas import (
    CompanionEvent,
    CompanionEventType,
    LiveRunSummary,
    LiveRuntimeStatus,
    SessionLifecycleState,
)
from baymax.live.session_supervisor import SessionSupervisor
from baymax.orchestrator.service import Orchestrator

logger = logging.getLogger(__name__)


class LiveRuntime:
    """Continuous live companion runtime.

    Manages the frame loop, session lifecycle, event detection,
    proactive response generation, and artifact logging.
    """

    def __init__(
        self,
        orchestrator: Orchestrator | None = None,
        db_path: str = "baymax_live.db",
    ) -> None:
        settings = get_settings()
        self._settings = settings
        self._orch = orchestrator or Orchestrator(db_path=db_path)

        # Session supervisor
        self._supervisor = SessionSupervisor(
            presence_min_frames=settings.presence_min_consecutive_frames,
            absence_timeout_sec=settings.absence_timeout_sec,
            resume_window_sec=settings.session_resume_window_sec,
        )

        # Event engine
        self._event_engine = EventEngine(
            stability_sec=settings.event_min_stability_sec,
        )
        self._event_engine.set_quiet_interval(settings.quiet_companionship_interval_sec)

        # Proactive scheduler
        self._scheduler = ProactiveScheduler(
            proactive_min_interval_sec=settings.proactive_min_interval_sec,
            any_response_min_interval_sec=settings.any_response_min_interval_sec,
            quiet_companionship_interval_sec=settings.quiet_companionship_interval_sec,
        )

        # Artifact logger
        self._artifact_logger = ArtifactLogger(
            artifact_dir=settings.live_artifact_dir,
            enabled=settings.enable_artifact_logging,
        )

        # Runtime state
        self._status = LiveRuntimeStatus()
        self._running = False
        self._start_time = 0.0
        self._frame_count = 0
        self._analysis_count = 0
        self._sessions_created = 0
        self._sessions_resumed = 0
        self._users_recognized: set[str] = set()
        self._errors: list[str] = []

        # Overlay settings
        self._enable_overlay = settings.enable_live_overlay
        self._debug_overlay = settings.enable_dialogue_debug
        self._last_memory_refs_count = 0

    @property
    def status(self) -> LiveRuntimeStatus:
        self._update_status()
        return self._status

    @property
    def supervisor(self) -> SessionSupervisor:
        return self._supervisor

    @property
    def event_engine(self) -> EventEngine:
        return self._event_engine

    @property
    def scheduler(self) -> ProactiveScheduler:
        return self._scheduler

    @property
    def artifact_logger(self) -> ArtifactLogger:
        return self._artifact_logger

    async def run(
        self,
        source: str | None = None,
        max_frames: int = 0,
    ) -> LiveRunSummary:
        """Run the live companion loop.

        Args:
            source: Override for frame source path. If None, uses settings.
            max_frames: Max frames to process (0 = unlimited).
        """
        settings = self._settings
        self._running = True
        self._start_time = time.time()

        # Set up signal handling for clean shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._request_stop)
            except NotImplementedError:
                pass

        # Initialize orchestrator
        await self._orch.initialize()

        # Create frame source
        source_path = source or settings.live_video_replay_path
        if source_path:
            # Replay mode
            import os
            if os.path.isdir(source_path):
                frame_source = FolderFrameSource(
                    folder_path=source_path,
                    target_fps=settings.preview_fps,
                )
            else:
                frame_source = OpenCVFrameSource(
                    source=source_path,
                    target_fps=settings.preview_fps,
                )
            source_type = "replay"
            self._status.source = f"replay:{source_path}"
        else:
            frame_source = OpenCVFrameSource(
                source=settings.live_camera_index,
                target_fps=settings.preview_fps,
            )
            source_type = "webcam"
            self._status.source = f"webcam:{settings.live_camera_index}"

        self._status.live_mode_active = True
        logger.info("Starting live runtime: source=%s", self._status.source)

        try:
            frame_source.open()
        except Exception as e:
            self._errors.append(f"Failed to open frame source: {e}")
            logger.error("Failed to open frame source: %s", e)
            return self._build_summary(source_type)

        analysis_interval = settings.analysis_interval_sec
        last_analysis_time = 0.0
        max_run_sec = settings.live_max_run_sec

        try:
            while self._running and frame_source.is_open():
                # Check run time limit
                if max_run_sec > 0 and (time.time() - self._start_time) >= max_run_sec:
                    logger.info("Max run time reached (%.0fs)", max_run_sec)
                    break

                # Check frame limit
                if max_frames > 0 and self._frame_count >= max_frames:
                    logger.info("Max frames reached (%d)", max_frames)
                    break

                frame = frame_source.read_frame()
                if frame is None:
                    if not frame_source.is_open():
                        break
                    await asyncio.sleep(0.01)
                    continue

                self._frame_count += 1
                now = time.time()

                # Determine if this is an analysis frame
                is_analysis = (now - last_analysis_time) >= analysis_interval
                if is_analysis:
                    last_analysis_time = now
                    self._analysis_count += 1
                    await self._process_analysis_frame(frame)

                # Render overlay if enabled
                if self._enable_overlay:
                    try:
                        self._update_status()
                        annotated = render_overlay(
                            frame=frame,
                            status=self._status,
                            debug=self._debug_overlay,
                            backend=self._orch.dialogue.backend_name,
                            model_name=self._orch.dialogue.model_name,
                            memory_refs_count=self._last_memory_refs_count,
                            posture=self._status.session_status.value,
                            engagement="medium",
                        )
                        # Show frame (if display available)
                        try:
                            import cv2
                            cv2.imshow("Bay-Max Live", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord("q"):
                                logger.info("User pressed 'q', stopping")
                                break
                        except Exception:
                            pass  # No display available (headless)
                    except Exception:
                        pass  # Overlay rendering is optional

                await asyncio.sleep(0.001)  # Yield to event loop

        except Exception as e:
            self._errors.append(f"Runtime error: {e}")
            logger.error("Runtime error: %s", e, exc_info=True)

        finally:
            # Clean shutdown
            end_evt = self._supervisor.force_end()
            if end_evt:
                self._artifact_logger.log_session_event(end_evt)

            frame_source.close()
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass

            self._running = False
            self._status.live_mode_active = False

        # Write final artifacts
        summary = self._build_summary(source_type)
        self._artifact_logger.write_session_timeline()
        self._artifact_logger.write_manifest(summary)
        self._artifact_logger.write_summary(summary)

        await self._orch.shutdown()
        logger.info(
            "Live runtime stopped. Frames: %d, Analyses: %d",
            self._frame_count, self._analysis_count,
        )
        return summary

    async def _process_analysis_frame(self, frame) -> None:
        """Run the full perception + event + response pipeline on an analysis frame."""
        session_id = self._supervisor.current_session_id

        # If no session, run lightweight detection to check for presence
        has_session = (
            session_id is not None
            and self._supervisor.state == SessionLifecycleState.ACTIVE
        )

        if has_session:
            # Full analysis through the orchestrator
            try:
                result = await self._orch.analyze_frame(session_id, frame)
                person_present = result.faces_detected > 0
                faces = result.recognized_faces
                user_id = faces[0].user_id if faces else None
                user_name = faces[0].user_display_name if faces else None
                confidence = faces[0].match_confidence if faces else 0.0
                recognized = len(faces) > 0
                eng = result.engagement
                posture = (
                    PostureLabel(eng.posture) if eng
                    else PostureLabel.UNKNOWN
                )
                engagement_level = (
                    EngagementLevel(eng.level) if eng
                    else EngagementLevel.MEDIUM
                )

                if recognized and user_name:
                    self._users_recognized.add(user_name)

            except Exception as e:
                logger.warning("Frame analysis failed: %s", e)
                self._errors.append(f"Frame analysis error: {e}")
                person_present = False
                user_id = None
                user_name = None
                confidence = 0.0
                recognized = False
                posture = PostureLabel.UNKNOWN
                engagement_level = EngagementLevel.MEDIUM
        else:
            # Lightweight detection -- just check for faces
            try:
                detector = self._orch._ensure_detector()
                detections = detector.detect(frame)
                person_present = len(detections) > 0
                user_id = None
                user_name = None
                confidence = 0.0
                recognized = False

                if person_present and detections[0].embedding is not None:
                    recognizer = self._orch._ensure_recognizer()
                    uid, score = recognizer.recognize(
                        detections[0].embedding,
                        self._orch._enrolled_embeddings,
                        threshold=self._orch._face_match_threshold,
                    )
                    if uid is not None:
                        user_id = uid
                        confidence = score
                        recognized = True
                        user = await self._orch.get_user(uid)
                        user_name = user.display_name if user else None
                        if user_name:
                            self._users_recognized.add(user_name)

                posture = PostureLabel.UNKNOWN
                engagement_level = EngagementLevel.MEDIUM
            except Exception as e:
                logger.warning("Lightweight detection failed: %s", e)
                person_present = False
                user_id = None
                user_name = None
                confidence = 0.0
                recognized = False
                posture = PostureLabel.UNKNOWN
                engagement_level = EngagementLevel.MEDIUM

        # --- Session lifecycle ---
        lifecycle_events = self._supervisor.update(person_present, user_id)
        for evt in lifecycle_events:
            logger.info("Session event: %s (session=%s)", evt.event_type, evt.session_id)
            self._artifact_logger.log_session_event(evt)

            if evt.event_type == "session_started":
                session = await self._orch.create_session(user_id=user_id)
                self._supervisor.set_session_id(session.id)
                self._sessions_created += 1
                logger.info("Created session: %s", session.id)

                # Snapshot on session start
                self._artifact_logger.save_snapshot(frame, "session_start", session.id)

            elif evt.event_type == "session_resumed":
                self._sessions_resumed += 1

            elif evt.event_type == "session_ended":
                # Consolidate if we have the info
                if evt.session_id and evt.user_id:
                    try:
                        await self._orch.consolidate(evt.session_id, evt.user_id)
                        logger.info("Consolidated session %s", evt.session_id)
                    except Exception as e:
                        logger.warning("Consolidation failed: %s", e)

        # Update session_id after potential creation
        session_id = self._supervisor.current_session_id

        # --- Event engine ---
        companion_events = self._event_engine.update(
            person_present=person_present,
            user_id=user_id,
            user_display_name=user_name,
            identity_confidence=confidence,
            recognized=recognized,
            posture=posture,
            engagement=engagement_level,
            session_id=session_id,
        )

        for event in companion_events:
            self._artifact_logger.log_event(event)

            # Check if we should generate a proactive response
            decision = self._scheduler.evaluate(event)
            if decision.should_respond and session_id:
                await self._generate_proactive_response(
                    event=event,
                    session_id=session_id,
                    user_id=user_id,
                    trigger_reason=decision.trigger_reason,
                    frame=frame,
                )
            elif decision.suppressed:
                logger.debug(
                    "Suppressed: %s — %s",
                    event.event_type.value,
                    decision.suppression_reason,
                )
                self._artifact_logger.log_suppression(decision)

        # Update status
        self._status.current_user_id = user_id
        self._status.current_user_display_name = user_name

    async def _generate_proactive_response(
        self,
        event: CompanionEvent,
        session_id: UUID,
        user_id: UUID | None,
        trigger_reason: str,
        frame=None,
    ) -> None:
        """Generate and log a proactive response for an event."""
        # Map event type to context for the orchestrator
        context = self._event_to_context(event, trigger_reason)

        try:
            response = await self._orch.respond(
                session_id=session_id,
                user_id=user_id,
                context=context,
            )

            # Store the system turn
            await self._orch.add_turn(
                session_id=session_id,
                role=TurnRole.SYSTEM,
                text=response.message,
                user_id=user_id,
            )

            # Record cooldown
            self._scheduler.record_response(event)

            # Update status
            self._status.last_response_text = response.message
            self._status.last_response_at = datetime.utcnow()
            self._last_memory_refs_count = len(response.memory_refs)

            # Log artifacts
            self._artifact_logger.log_response(
                event=event,
                response_text=response.message,
                strategy=response.strategy.value,
                memory_refs=response.memory_refs,
                backend=response.backend,
                trigger_reason=trigger_reason,
            )

            # Snapshot on response
            if frame is not None:
                self._artifact_logger.save_snapshot(frame, "response", session_id)

            logger.info(
                "Proactive response [%s]: %s",
                trigger_reason,
                response.message[:80],
            )

        except Exception as e:
            logger.error("Failed to generate proactive response: %s", e)
            self._errors.append(f"Response generation error: {e}")

    def _event_to_context(self, event: CompanionEvent, trigger_reason: str) -> str:
        """Convert an event to context text for the dialogue system."""
        name = event.user_display_name or "there"

        if event.event_type == CompanionEventType.RECOGNIZED_USER_ARRIVED:
            return (
                f"A returning user ({name}) has arrived."
                " Greet them warmly using any relevant memories."
            )
        elif event.event_type == CompanionEventType.UNKNOWN_USER_ARRIVED:
            return (
                "A new person has appeared."
                " Greet them gently without assuming you know them."
            )
        elif event.event_type == CompanionEventType.RECOGNITION_GAINED:
            return f"I just recognized this person as {name}. Welcome them back."
        elif event.event_type == CompanionEventType.POSTURE_CHANGED:
            from_p = event.details.get("from", "unknown")
            to_p = event.details.get("to", "unknown")
            return (
                f"The user's posture changed from {from_p} to {to_p}."
                " Check in gently if appropriate."
            )
        elif event.event_type == CompanionEventType.ENGAGEMENT_CHANGED:
            from_e = event.details.get("from", "medium")
            to_e = event.details.get("to", "medium")
            return (
                f"The user's engagement changed from {from_e} to {to_e}."
                " Check in if engagement dropped."
            )
        elif event.event_type == CompanionEventType.QUIET_COMPANIONSHIP_DUE:
            return f"It's been quiet for a while with {name} present. Offer gentle companionship."
        elif event.event_type == CompanionEventType.SESSION_RESUMED:
            return f"Welcome back, {name}! The session has resumed."
        else:
            return f"Event: {event.event_type.value}"

    def _update_status(self) -> None:
        """Refresh the runtime status object."""
        self._status.session_status = self._supervisor.state
        self._status.frame_count = self._frame_count
        self._status.analysis_count = self._analysis_count
        if self._start_time > 0:
            self._status.uptime_sec = time.time() - self._start_time
        if self._event_engine.events:
            last = self._event_engine.events[-1]
            self._status.last_event = last.event_type.value
            self._status.last_event_at = last.timestamp

    def _request_stop(self) -> None:
        """Request a clean stop."""
        logger.info("Stop requested")
        self._running = False

    def _build_summary(self, source_type: str) -> LiveRunSummary:
        """Build the final run summary."""
        return LiveRunSummary(
            source=source_type,
            started_at=(
                datetime.utcfromtimestamp(self._start_time)
                if self._start_time else datetime.utcnow()
            ),
            ended_at=datetime.utcnow(),
            total_frames=self._frame_count,
            total_analyses=self._analysis_count,
            total_events=len(self._event_engine.events),
            total_responses=self._scheduler.total_responses,
            total_suppressions=self._scheduler.total_suppressions,
            sessions_created=self._sessions_created,
            sessions_resumed=self._sessions_resumed,
            users_recognized=sorted(self._users_recognized),
            backend=self._orch.dialogue.backend_name,
            model_name=self._orch.dialogue.model_name,
            errors=self._errors,
        )

    def stop(self) -> None:
        """Public method to stop the runtime."""
        self._request_stop()
