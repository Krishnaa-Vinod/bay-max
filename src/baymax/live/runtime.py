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
from baymax.perception.emotion import EmotionAnalyzer, get_emotion_analyzer
from baymax.perception.emotion_smoother import AffectSmoother
from baymax.tts.provider import NullTTSProvider, get_tts_provider
from baymax.tts.speech_service import SpeechService

logger = logging.getLogger(__name__)


class LiveRuntime:
    """Continuous live companion runtime.

    Manages the frame loop, session lifecycle, event detection,
    proactive response generation, speech input/output, and artifact logging.
    """

    def __init__(
        self,
        orchestrator: Orchestrator | None = None,
        db_path: str = "baymax.db",
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

        # Speech service (TTS)
        self._tts_init_error: str | None = None
        if settings.tts_enabled:
            try:
                tts_provider = get_tts_provider(
                    backend=settings.tts_backend,
                    voice=settings.tts_voice,
                    device=settings.tts_device,
                    model_dir=settings.model_dir,
                    sample_rate=settings.tts_rate,
                )
            except Exception as exc:
                self._tts_init_error = str(exc)
                logger.warning("Failed to initialize TTS provider, using null fallback: %s", exc)
                tts_provider = NullTTSProvider()
        else:
            tts_provider = NullTTSProvider()

        self._speech_service = SpeechService(
            provider=tts_provider,
            output_dir=settings.tts_output_dir,
            enable_playback=settings.enable_audio_playback,
            audio_backend=settings.audio_backend,
        )

        # Iteration 008: Speech input service
        self._speech_input_service = None
        self._speech_input_enabled = settings.enable_speech_input
        self._speech_init_error: str | None = None
        self._speech_start_error: str | None = None
        if self._speech_input_enabled:
            try:
                from baymax.audio.speech_input_service import SpeechInputService

                self._speech_input_service = SpeechInputService(
                    sample_rate=settings.mic_sample_rate,
                    channels=settings.mic_channels,
                    mic_device=settings.mic_device,
                    vad_backend=settings.vad_backend,
                    vad_threshold=settings.vad_threshold,
                    vad_min_speech_ms=settings.vad_min_speech_ms,
                    vad_silence_ms=settings.vad_silence_ms,
                    stt_backend=settings.stt_backend,
                    stt_model=settings.whisper_model,
                    stt_device=settings.whisper_device,
                    echo_threshold=settings.echo_similarity_threshold,
                    mic_mode=settings.mic_mode,
                    post_speech_cooldown_ms=settings.post_speech_cooldown_ms,
                    artifact_dir=settings.audio_artifact_dir,
                    enable_artifacts=settings.enable_artifact_logging,
                )
            except Exception as exc:
                logger.warning("Failed to init SpeechInputService: %s", exc)
                self._speech_input_service = None
                self._speech_init_error = str(exc)

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
        self._speech_input_task: asyncio.Task | None = None
        self._speech_consumer_task: asyncio.Task | None = None

        # Perception state tracking for UI (iteration 010b hotfix)
        self._last_recognition_confidence: float | None = None
        self._last_face_detected: bool = False
        self._last_recognition_state: str = "no_face"
        self._last_posture: str | None = None
        self._last_engagement: str | None = None
        self._last_engagement_score: float | None = None

        # Overlay settings
        self._enable_overlay = settings.enable_live_overlay
        self._debug_overlay = getattr(
            settings, "enable_live_debug_hud", False
        )
        self._last_memory_refs_count = 0

        # Iteration 009: Affect analysis initialization
        self._affect_enabled = settings.enable_affect
        self._affect_sample_every_n_frames = settings.affect_sample_every_n_frames
        self._affect_frame_counter = 0
        self._affect_analyzer: EmotionAnalyzer | None = None
        self._affect_smoother: AffectSmoother | None = None

        if self._affect_enabled:
            try:
                # Get model path for MediaPipe face landmarker
                model_path = None
                if settings.affect_backend == "mediapipe":
                    import os
                    model_path = os.path.join(settings.model_dir, "face_landmarker.task")
                    if not os.path.exists(model_path):
                        logger.warning("MediaPipe model not found at %s, will attempt auto-download", model_path)
                        model_path = None

                self._affect_analyzer = get_emotion_analyzer(
                    backend=settings.affect_backend,
                    device=settings.affect_device,
                    model_path=model_path,
                )
                self._affect_smoother = AffectSmoother(
                    alpha=settings.affect_smoothing_alpha,
                    stability_threshold=0.1,
                    stability_duration_sec=settings.affect_stability_duration_sec,
                    confidence_threshold=settings.affect_confidence_threshold,
                )
                logger.info(
                    "Affect analysis initialized: backend=%s, sample_every=%d frames",
                    settings.affect_backend,
                    settings.affect_sample_every_n_frames,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to init affect analyzer with backend '%s': %s. "
                    "Falling back to null backend.",
                    settings.affect_backend,
                    exc,
                )
                # Fall back to null backend
                try:
                    self._affect_analyzer = get_emotion_analyzer(backend="null")
                    self._affect_smoother = AffectSmoother(
                        alpha=settings.affect_smoothing_alpha,
                        stability_threshold=0.1,
                        stability_duration_sec=settings.affect_stability_duration_sec,
                        confidence_threshold=settings.affect_confidence_threshold,
                    )
                    logger.info(
                        "Affect analysis initialized with null fallback, "
                        "sample_every=%d frames",
                        settings.affect_sample_every_n_frames,
                    )
                except Exception as fallback_exc:
                    logger.warning("Failed to init null affect backend: %s", fallback_exc)
                    self._affect_analyzer = None
                    self._affect_smoother = None

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

    @property
    def speech_service(self) -> SpeechService:
        return self._speech_service

    @property
    def speech_input_service(self):
        return self._speech_input_service

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

        # Start speech input if enabled
        if self._speech_input_service is not None:
            started = await self._speech_input_service.start()
            if started:
                self._speech_start_error = None
                logger.info("Speech input started")
                self._speech_input_task = asyncio.create_task(
                    self._speech_input_service.run_loop()
                )
                self._speech_consumer_task = asyncio.create_task(
                    self._consume_transcriptions()
                )
            else:
                self._speech_start_error = self._speech_input_service.last_error or "Microphone or VAD startup failed"
                logger.warning("Speech input failed to start")

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

        if source_type == "webcam":
            active_source = getattr(frame_source, "resolved_source", settings.live_camera_index)
            self._status.source = f"webcam:{active_source}"

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

                # Always update status
                self._update_status()

                # Render overlay if enabled, otherwise use raw frame
                display_frame = frame
                if self._enable_overlay:
                    try:
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
                        display_frame = annotated

                        # Show frame in OpenCV window (if display available)
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

                # Always update frame for UI endpoint (regardless of overlay setting)
                try:
                    from apps.api.live_http import update_annotated_frame
                    update_annotated_frame(display_frame)
                except ImportError:
                    pass  # API not running

                await asyncio.sleep(0.001)  # Yield to event loop

        except Exception as e:
            self._errors.append(f"Runtime error: {e}")
            logger.error("Runtime error: %s", e, exc_info=True)

        finally:
            # Stop speech input
            if self._speech_input_service is not None:
                await self._speech_input_service.stop()
            if self._speech_input_task is not None:
                self._speech_input_task.cancel()
                try:
                    await self._speech_input_task
                except (asyncio.CancelledError, Exception):
                    pass
            if self._speech_consumer_task is not None:
                self._speech_consumer_task.cancel()
                try:
                    await self._speech_consumer_task
                except (asyncio.CancelledError, Exception):
                    pass

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

    async def _consume_transcriptions(self) -> None:
        """Consume transcriptions from speech input and process as user turns."""
        while self._running:
            try:
                if self._speech_input_service is None:
                    await asyncio.sleep(0.1)
                    continue

                try:
                    result = await asyncio.wait_for(
                        self._speech_input_service.output_queue.get(),
                        timeout=0.5,
                    )
                except TimeoutError:
                    continue

                if not result.success or not result.text.strip():
                    continue

                await self._handle_speech_turn(result.text)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Transcription consumer error: %s", exc)
                await asyncio.sleep(0.5)

    async def _handle_speech_turn(self, text: str) -> None:
        """Process a speech transcription as a user turn through the full pipeline."""
        session_id = self._supervisor.current_session_id
        user_id = self._status.current_user_id

        if session_id is None:
            logger.debug("No active session for speech turn, skipping")
            return

        logger.info("Speech turn: '%s'", text[:80])

        try:
            # Store user turn with source='speech'
            await self._orch.add_turn(
                session_id=session_id,
                role=TurnRole.USER,
                text=text,
                user_id=user_id,
                source="speech",
            )
            self._status.turn_count += 1

            # Generate response
            response = await self._orch.respond(
                session_id=session_id,
                user_id=user_id,
                context=f"The user said: {text}",
            )

            # Store system response turn
            await self._orch.add_turn(
                session_id=session_id,
                role=TurnRole.SYSTEM,
                text=response.message,
                user_id=user_id,
            )
            self._status.turn_count += 1

            # Record cooldown
            self._status.last_response_text = response.message
            self._status.last_response_at = datetime.utcnow()
            self._last_memory_refs_count = len(response.memory_refs)

            logger.info("Speech response: %s", response.message[:80])

            # Synthesize and play TTS with speaking lock
            await self._speak_response(response.message, session_id, "speech_turn")

        except Exception as exc:
            logger.error("Failed to process speech turn: %s", exc)
            self._errors.append(f"Speech turn error: {exc}")

    async def _speak_response(
        self, text: str, session_id: UUID | None, trigger: str
    ) -> None:
        """Synthesize and play TTS, managing speaking lock for echo suppression."""
        self._status.last_tts_result = "requested"
        self._status.last_tts_error = ""
        self._status.last_tts_wav_path = ""
        self._status.last_tts_playback_ok = None

        try:
            from apps.api.live_ws import broadcast_event, build_event_message

            await broadcast_event(build_event_message(
                stage="tts",
                status="in_progress",
                detail=f"TTS requested ({self._speech_service.provider.name()}:{self._settings.tts_voice})",
                event_type="speech_event",
            ))
        except Exception:
            pass

        # Engage speaking lock
        if self._speech_input_service is not None:
            self._speech_input_service.set_speaking_lock(True)
            self._speech_input_service.record_spoken_text(text)

        try:
            speech_result = await self._speech_service.synthesize_and_play(
                text=text,
                session_id=str(session_id) if session_id else None,
                trigger_event=trigger,
            )
            self._status.last_tts_wav_path = speech_result.wav_path or ""
            self._status.last_tts_playback_ok = speech_result.playback_ok

            if speech_result.success:
                if speech_result.playback_ok is True:
                    self._status.last_tts_result = "synth_ok_playback_ok"
                    self._status.last_tts_error = ""
                    tts_detail = "TTS synth ok; playback ok"
                else:
                    self._status.last_tts_result = "synth_ok_no_playback"
                    self._status.last_tts_error = speech_result.playback_error or "Playback unavailable"
                    tts_detail = (
                        f"TTS synth ok; playback unavailable: {self._status.last_tts_error}"
                    )
                    if speech_result.wav_path:
                        tts_detail += f"; wav: {speech_result.wav_path}"

                self._artifact_logger.log_speech_event(
                    text=text,
                    backend=speech_result.backend,
                    voice=speech_result.voice,
                    wav_path=speech_result.wav_path,
                    success=True,
                    session_id=str(session_id) if session_id else None,
                    trigger_event=trigger,
                )

                try:
                    from apps.api.live_ws import broadcast_event, build_event_message

                    await broadcast_event(build_event_message(
                        stage="tts",
                        status="complete",
                        detail=tts_detail,
                        event_type="speech_event",
                    ))
                except Exception:
                    pass
            else:
                self._status.last_tts_result = "synth_failed"
                self._status.last_tts_error = speech_result.error or "Synthesis failed"
                self._artifact_logger.log_speech_event(
                    text=text,
                    backend=speech_result.backend,
                    voice=speech_result.voice,
                    wav_path=None,
                    success=False,
                    error=speech_result.error,
                    session_id=str(session_id) if session_id else None,
                    trigger_event=trigger,
                )
                try:
                    from apps.api.live_ws import broadcast_event, build_event_message

                    await broadcast_event(build_event_message(
                        stage="tts",
                        status="error",
                        detail=f"TTS synth failed: {self._status.last_tts_error}",
                        event_type="speech_event",
                    ))
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("TTS error (non-fatal): %s", exc)
            self._status.last_tts_result = "runtime_error"
            self._status.last_tts_error = str(exc)
            try:
                from apps.api.live_ws import broadcast_event, build_event_message

                await broadcast_event(build_event_message(
                    stage="tts",
                    status="error",
                    detail=f"TTS runtime error: {exc}",
                    event_type="speech_event",
                ))
            except Exception:
                pass
        finally:
            # Release speaking lock (triggers cooldown)
            if self._speech_input_service is not None:
                self._speech_input_service.set_speaking_lock(False)

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
                self._status.turn_count = 0
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

        # --- Affect analysis (Iteration 009) ---
        self._affect_frame_counter += 1
        if (
            self._affect_enabled
            and self._affect_analyzer is not None
            and self._affect_smoother is not None
            and session_id is not None
            and self._affect_frame_counter % self._affect_sample_every_n_frames == 0
        ):
            await self._run_affect_analysis(frame, session_id)

        # Update status
        self._status.current_user_id = user_id
        self._status.current_user_display_name = user_name
        self._last_face_detected = person_present

        if not person_present:
            self._last_recognition_state = "no_face"
        elif recognized and user_id is not None:
            self._last_recognition_state = "recognized_enrolled"
        elif confidence > 0.0:
            self._last_recognition_state = "below_threshold"
        else:
            self._last_recognition_state = "unknown_user"

        # Track perception values for UI (iteration 010b hotfix)
        self._last_recognition_confidence = confidence if confidence > 0 else None
        self._last_posture = posture.value if posture != PostureLabel.UNKNOWN else None
        self._last_engagement = engagement_level.value if engagement_level else None
        # engagement_score not directly available in this path; left as None

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
            self._status.turn_count += 1

            # Record cooldown
            self._scheduler.record_response(event)

            # Update status
            self._status.last_response_text = response.message
            self._status.last_response_at = datetime.utcnow()
            self._last_memory_refs_count = len(response.memory_refs)
            # Track memory refs for UI (iteration 010b hotfix)
            refs = response.memory_refs[:10] if response.memory_refs else []
            self._status.last_memory_refs = refs

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

            # Synthesize and play speech with speaking lock
            await self._speak_response(
                response.message, session_id, event.event_type.value
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

    async def _run_affect_analysis(self, frame, session_id: UUID) -> None:
        """Run facial affect analysis on the current frame."""
        if self._affect_analyzer is None or self._affect_smoother is None:
            return

        try:
            # Get face detection from orchestrator to reuse detected face region
            detector = self._orch._ensure_detector()
            detections = detector.detect(frame)

            if not detections:
                # No face detected - return unknown affect
                logger.debug("No face detected for affect analysis")
                return

            # Use first detected face
            face = detections[0]
            face_bbox = {
                "x1": face.bbox.x1,
                "y1": face.bbox.y1,
                "x2": face.bbox.x2,
                "y2": face.bbox.y2,
            }

            # Run affect analysis
            emotion_result = self._affect_analyzer.analyze(frame, face_bbox)

            # Update smoother
            smoothed_state = self._affect_smoother.update(emotion_result)

            # Update state manager with affect values
            self._orch.state_manager.update_affect(
                session_id,
                affect_enabled=True,
                valence=smoothed_state.valence,
                arousal=smoothed_state.arousal,
                affect_confidence=smoothed_state.confidence,
                affect_stable_duration_sec=smoothed_state.stable_duration_sec,
                affect_sample_count=smoothed_state.sample_count,
            )

            # Update runtime status affect fields
            self._status.affect_enabled = True
            self._status.affect_backend = self._affect_analyzer.name()
            self._status.emotion_valence = smoothed_state.valence
            self._status.emotion_arousal = smoothed_state.arousal
            self._status.emotion_confidence = smoothed_state.confidence
            self._status.emotion_stable_duration_sec = smoothed_state.stable_duration_sec

            # Generate debug summary
            if smoothed_state.confidence >= self._settings.affect_confidence_threshold:
                if smoothed_state.valence > 0.3:
                    valence_desc = "positive"
                elif smoothed_state.valence < -0.3:
                    valence_desc = "subdued"
                else:
                    valence_desc = "neutral"

                if smoothed_state.arousal > 0.6:
                    arousal_desc = "energetic"
                elif smoothed_state.arousal < 0.3:
                    arousal_desc = "calm"
                else:
                    arousal_desc = "moderate"

                self._status.emotion_debug_summary = (
                    f"{valence_desc}, {arousal_desc} (conf: {smoothed_state.confidence:.2f})"
                )
            else:
                self._status.emotion_debug_summary = (
                    f"low confidence ({smoothed_state.confidence:.2f})"
                )

            logger.debug(
                "Affect analysis: valence=%.3f, arousal=%.3f, confidence=%.3f, stable=%.1fs",
                smoothed_state.valence,
                smoothed_state.arousal,
                smoothed_state.confidence,
                smoothed_state.stable_duration_sec,
            )

        except Exception as e:
            logger.warning("Affect analysis failed: %s", e)

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

        # Perception tracking for UI (iteration 010b hotfix)
        self._status.face_detected = self._last_face_detected
        self._status.recognition_state = self._last_recognition_state
        self._status.face_match_threshold = self._orch._face_match_threshold
        self._status.recognition_confidence = self._last_recognition_confidence
        self._status.posture = self._last_posture
        self._status.engagement = self._last_engagement
        self._status.engagement_score = self._last_engagement_score
        self._status.session_binding = (
            "identified_user"
            if self._status.current_user_id is not None
            else "anonymous"
        )
        self._status.dialogue_backend = self._orch.dialogue.backend_name
        self._status.dialogue_model = self._orch.dialogue.model_name
        self._status.dialogue_requested_backend = self._orch.dialogue_requested_backend
        self._status.dialogue_requested_model = self._orch.dialogue_requested_model
        self._status.dialogue_fallback_warning = self._orch.dialogue_fallback_warning

        # Speech output status
        qs = self._speech_service.queue_status()
        self._status.last_spoken_text = qs.last_spoken_text
        self._status.speech_queue_depth = qs.queue_depth
        self._status.tts_backend = self._speech_service.provider.name()
        self._status.tts_voice = self._settings.tts_voice
        self._status.tts_voice_preset = self._settings.tts_voice_preset
        if self._tts_init_error and not self._status.last_tts_error:
            self._status.last_tts_error = f"TTS fallback active: {self._tts_init_error}"

        # Speech input status (Iteration 008)
        if self._speech_input_service is not None:
            sis = self._speech_input_service.status()
            self._status.speech_input_enabled = sis.speech_input_enabled
            self._status.listening = sis.listening
            self._status.vad_active = sis.vad_active
            self._status.stt_backend = sis.stt_backend
            self._status.speaking_lock_active = sis.speaking_lock_active
            self._status.last_heard_text = sis.last_heard_text
            self._status.transcription_latency_ms = sis.transcription_latency_ms
            self._status.mic_mode = sis.mic_mode
            self._status.speech_disabled_reason = (
                ""
                if sis.speech_input_enabled
                else self._speech_start_error or "Speech input service not running"
            )
        else:
            self._status.speech_input_enabled = False
            if not self._speech_input_enabled:
                self._status.speech_disabled_reason = (
                    "Speech backend disabled (BAYMAX_ENABLE_SPEECH_INPUT=false)"
                )
            elif self._settings.stt_backend == "null":
                self._status.speech_disabled_reason = (
                    "Speech backend is null (BAYMAX_STT_BACKEND=null)"
                )
            elif self._speech_init_error:
                self._status.speech_disabled_reason = (
                    f"Speech input dependency/init error: {self._speech_init_error}"
                )
            else:
                self._status.speech_disabled_reason = (
                    "Speech input service unavailable"
                )

        # Affect analysis status (Iteration 009)
        if self._affect_enabled and self._affect_smoother is not None:
            self._status.affect_enabled = True
            self._status.affect_backend = (
                self._affect_analyzer.name() if self._affect_analyzer else "null"
            )
            # Get current smoothed state
            smoothed = self._affect_smoother.get_current_state()
            self._status.emotion_valence = smoothed.valence
            self._status.emotion_arousal = smoothed.arousal
            self._status.emotion_confidence = smoothed.confidence
            self._status.emotion_stable_duration_sec = smoothed.stable_duration_sec
        else:
            self._status.affect_enabled = False

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
