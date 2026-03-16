"""Orchestrator service coordinating the end-to-end flow."""

import logging
import time
from datetime import datetime
from uuid import UUID

import numpy as np

from baymax.config.settings import get_settings
from baymax.core.enums import EngagementLevel, PostureLabel
from baymax.dialogue.rule_based import RuleBasedDialogue
from baymax.memory.retrieve import retrieve_memories
from baymax.memory.store_sqlite import SQLiteMetadataStore
from baymax.perception.facenet_adapter import CosineRecognizer, FacenetDetector
from baymax.perception.heuristics import (
    MotionTracker,
    compute_engagement,
    compute_lean,
    compute_posture,
)
from baymax.perception.interfaces import FaceDetector, FaceRecognizer, StubFaceDetector
from baymax.perception.pose_interface import PoseEstimator, StubPoseEstimator
from baymax.perception.tracker import SimpleTracker
from baymax.planner.supportive_planner import SupportivePlanner
from baymax.schemas.memory import MemoryQueryResult
from baymax.schemas.perception import (
    BodyStateObservation,
    EngagementResult,
    FaceEmbeddingRecord,
    FrameAnalysisResult,
    PoseResult,
    RecognitionObservation,
    RecognizedFace,
    UnknownFace,
)
from baymax.schemas.response import SupportiveResponse
from baymax.schemas.session import Session
from baymax.schemas.user import FaceEnrollment, UserProfile, UserProfileCreate
from baymax.state.manager import StateManager
from baymax.state.models import InteractionState

logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinates the flow from frame ingest to supportive response."""

    def __init__(self, db_path: str = "baymax.db") -> None:
        self.store = SQLiteMetadataStore(db_path=db_path)
        self.state_manager = StateManager()
        self.planner = SupportivePlanner()
        self.dialogue = RuleBasedDialogue()
        self.tracker = SimpleTracker()
        self.motion_tracker = MotionTracker()

        settings = get_settings()
        device = settings.device
        if device == "cuda_if_available_else_cpu":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self._device = device
        self._face_match_threshold = settings.face_match_threshold
        self._max_faces = settings.max_faces_per_frame
        self._pose_backend = settings.pose_backend
        self._pose_min_confidence = settings.pose_min_confidence
        self._enable_annotations = settings.enable_annotations
        self._artifact_dir = settings.artifact_dir

        # Perception backends (lazy-loaded)
        self._detector: FaceDetector | None = None
        self._recognizer: FaceRecognizer | None = None
        self._pose_estimator: PoseEstimator | None = None

        # Cached enrolled embeddings (refreshed on enroll)
        self._enrolled_embeddings: list[FaceEmbeddingRecord] = []

        # Track previous state per session for observation writing
        self._prev_user_ids: dict[UUID, UUID | None] = {}
        self._prev_posture: dict[UUID, PostureLabel] = {}
        self._prev_engagement: dict[UUID, EngagementLevel] = {}
        self._prev_pose_visible: dict[UUID, bool] = {}

    def _ensure_detector(self) -> FaceDetector:
        if self._detector is None:
            try:
                det = FacenetDetector(device=self._device, max_faces=self._max_faces)
                det.load_model()
                self._detector = det
                logger.info("FacenetDetector loaded on %s", self._device)
            except Exception:
                logger.warning(
                    "Failed to load FacenetDetector, falling back to stub",
                    exc_info=True,
                )
                self._detector = StubFaceDetector()
        return self._detector

    def _ensure_recognizer(self) -> FaceRecognizer:
        if self._recognizer is None:
            self._recognizer = CosineRecognizer()
        return self._recognizer

    def _ensure_pose_estimator(self) -> PoseEstimator:
        if self._pose_estimator is None:
            if self._pose_backend == "mediapipe":
                try:
                    from baymax.perception.mediapipe_pose import MediaPipePoseEstimator

                    est = MediaPipePoseEstimator(
                        min_detection_confidence=self._pose_min_confidence,
                    )
                    est.load_model()
                    self._pose_estimator = est
                    logger.info("MediaPipePoseEstimator loaded")
                except Exception:
                    logger.warning(
                        "Failed to load MediaPipePoseEstimator, falling back to stub",
                        exc_info=True,
                    )
                    self._pose_estimator = StubPoseEstimator()
            else:
                logger.info("Pose backend '%s' not supported, using stub", self._pose_backend)
                self._pose_estimator = StubPoseEstimator()
        return self._pose_estimator

    async def initialize(self) -> None:
        """Initialize the orchestrator and its dependencies."""
        await self.store.initialize()
        # Pre-load enrolled embeddings
        self._enrolled_embeddings = await self.store.get_all_face_embeddings()
        logger.info("Loaded %d enrolled face embeddings", len(self._enrolled_embeddings))

    async def create_user(self, request: UserProfileCreate) -> UserProfile:
        """Create a new user profile."""
        user = UserProfile(display_name=request.display_name, notes=request.notes)
        return await self.store.create_user(user)

    async def get_user(self, user_id: UUID) -> UserProfile | None:
        """Fetch a user profile by ID."""
        return await self.store.get_user(user_id)

    async def list_users(self) -> list[UserProfile]:
        """List all active user profiles."""
        return await self.store.list_users()

    async def enroll_face(self, user_id: UUID, image: np.ndarray) -> FaceEnrollment:
        """Enroll a face from an uploaded image.

        Detects exactly one face, computes its embedding, and stores it.
        """
        detector = self._ensure_detector()
        faces = detector.detect(image)

        face_count = len(faces)

        if face_count == 0:
            enrollment = FaceEnrollment(
                user_id=user_id,
                status="failed",
                face_count_detected=0,
                message="No face detected in the image.",
            )
            await self.store.create_face_enrollment(enrollment)
            return enrollment

        if face_count > 1:
            enrollment = FaceEnrollment(
                user_id=user_id,
                status="failed",
                face_count_detected=face_count,
                message=f"Expected exactly 1 face, found {face_count}. Please use a clearer photo.",
            )
            await self.store.create_face_enrollment(enrollment)
            return enrollment

        face = faces[0]
        if face.embedding is None:
            enrollment = FaceEnrollment(
                user_id=user_id,
                status="failed",
                face_count_detected=1,
                message="Face detected but embedding extraction failed.",
            )
            await self.store.create_face_enrollment(enrollment)
            return enrollment

        # Store the embedding
        emb_record = FaceEmbeddingRecord(
            user_id=user_id,
            embedding=face.embedding,
        )
        await self.store.store_face_embedding(emb_record)

        # Create successful enrollment record
        enrollment = FaceEnrollment(
            user_id=user_id,
            status="enrolled",
            confidence=face.detection_confidence,
            encoding_ref=str(emb_record.id),
            embedding_model="InceptionResnetV1-vggface2",
            face_count_detected=1,
            message="Face enrolled successfully.",
        )
        await self.store.create_face_enrollment(enrollment)

        # Refresh cached embeddings
        self._enrolled_embeddings = await self.store.get_all_face_embeddings()

        return enrollment

    async def analyze_frame(
        self, session_id: UUID, frame: np.ndarray
    ) -> FrameAnalysisResult:
        """Run detection + recognition + pose + engagement on a frame and update state.

        Returns a FrameAnalysisResult with recognized/unknown faces, pose, engagement,
        and observations.
        """
        t0 = time.time()

        detector = self._ensure_detector()
        recognizer = self._ensure_recognizer()
        pose_estimator = self._ensure_pose_estimator()

        # 1. Detect faces
        detections = detector.detect(frame)

        # 2. Update tracker
        tracks = self.tracker.update(detections)

        # 3. Recognize each detected face
        recognized: list[RecognizedFace] = []
        unknown: list[UnknownFace] = []
        primary_user_id: UUID | None = None
        primary_confidence: float = 0.0
        primary_display_name: str | None = None

        for det in detections:
            if det.embedding is None:
                unknown.append(UnknownFace(
                    bbox=det.bbox,
                    detection_confidence=det.detection_confidence,
                ))
                continue

            user_id, score = recognizer.recognize(
                det.embedding,
                self._enrolled_embeddings,
                threshold=self._face_match_threshold,
            )

            if user_id is not None:
                # Look up display name
                user = await self.store.get_user(user_id)
                display_name = user.display_name if user else None
                recognized.append(RecognizedFace(
                    user_id=user_id,
                    user_display_name=display_name,
                    bbox=det.bbox,
                    detection_confidence=det.detection_confidence,
                    match_confidence=score,
                ))
                if score > primary_confidence:
                    primary_user_id = user_id
                    primary_confidence = score
                    primary_display_name = display_name

                # Associate track with user
                for track in tracks:
                    if det.embedding == track.embedding:
                        self.tracker.set_user_id(track.track_id, user_id, score)
                        break
            else:
                unknown.append(UnknownFace(
                    bbox=det.bbox,
                    detection_confidence=det.detection_confidence,
                    best_match_score=score if score > 0 else None,
                ))

        # 4. Pose estimation
        pose_result: PoseResult = pose_estimator.estimate(frame)

        # 5. Compute body-state heuristics
        posture = compute_posture(pose_result)
        lean = compute_lean(pose_result)
        motion = self.motion_tracker.update(pose_result)
        face_visible = len(detections) > 0
        face_recognized = len(recognized) > 0

        engagement_result = compute_engagement(
            posture=posture,
            lean=lean,
            motion=motion,
            face_visible=face_visible,
            face_recognized=face_recognized,
            pose_visible=pose_result.pose_present,
        )

        # 6. Update interaction state (face recognition)
        summary_parts = []
        if recognized:
            names = [r.user_display_name or str(r.user_id)[:8] for r in recognized]
            summary_parts.append(f"Recognized: {', '.join(names)}")
        if unknown:
            summary_parts.append(f"Unknown faces: {len(unknown)}")
        if pose_result.pose_present:
            summary_parts.append(f"Posture: {posture}, Lean: {lean}")
        frame_summary = "; ".join(summary_parts) if summary_parts else "No faces detected"

        self.state_manager.update_recognition(
            session_id,
            user_id=primary_user_id,
            user_display_name=primary_display_name,
            identity_confidence=primary_confidence,
            face_count=len(detections),
            active_track_ids=self.tracker.active_track_ids,
            frame_summary=frame_summary,
        )

        # 7. Update body-state in state
        self.state_manager.update_body_state(session_id, engagement_result)

        # 8. Write meaningful observations (face + body-state)
        obs_count = await self._write_observations(
            session_id=session_id,
            recognized=recognized,
            unknown=unknown,
            primary_user_id=primary_user_id,
            primary_confidence=primary_confidence,
        )
        obs_count += await self._write_body_state_observations(
            session_id=session_id,
            engagement_result=engagement_result,
            user_id=primary_user_id,
        )

        # Update previous state for this session
        self._prev_user_ids[session_id] = primary_user_id
        self._prev_posture[session_id] = posture
        self._prev_engagement[session_id] = engagement_result.level
        self._prev_pose_visible[session_id] = pose_result.pose_present

        latency_ms = (time.time() - t0) * 1000

        state = self.state_manager.get_or_create(session_id)
        return FrameAnalysisResult(
            session_id=session_id,
            timestamp=datetime.utcnow(),
            faces_detected=len(detections),
            recognized_faces=recognized,
            unknown_faces=unknown,
            pose_result=pose_result if pose_result.pose_present else None,
            engagement=engagement_result,
            state=state.model_dump(mode="json"),
            observations_written=obs_count,
            latency_ms=round(latency_ms, 1),
        )

    async def _write_observations(
        self,
        session_id: UUID,
        recognized: list[RecognizedFace],
        unknown: list[UnknownFace],
        primary_user_id: UUID | None,
        primary_confidence: float,
    ) -> int:
        """Write recognition observations only on meaningful events.

        Meaningful events:
        - first_recognition: first time a user is seen in this session
        - user_switch: different user recognized than before
        - unknown_to_known: previously unknown face now matched
        - known_to_unknown: previously matched face now unrecognized
        """
        prev_user = self._prev_user_ids.get(session_id)
        count = 0

        # First recognition in session
        if prev_user is None and primary_user_id is not None:
            user = await self.store.get_user(primary_user_id)
            name = user.display_name if user else str(primary_user_id)[:8]
            obs = RecognitionObservation(
                session_id=session_id,
                user_id=primary_user_id,
                event_type="first_recognition",
                content=(
                    f"Recognized {name} for the first time this session"
                    f" (confidence: {primary_confidence:.2f})."
                ),
                confidence=primary_confidence,
                evidence_refs=[f"session:{session_id}"],
            )
            await self.store.store_observation(obs)
            count += 1

        # User switch
        elif (
            prev_user is not None
            and primary_user_id is not None
            and prev_user != primary_user_id
        ):
            user = await self.store.get_user(primary_user_id)
            name = user.display_name if user else str(primary_user_id)[:8]
            obs = RecognitionObservation(
                session_id=session_id,
                user_id=primary_user_id,
                event_type="user_switch",
                content=f"User switched to {name} (confidence: {primary_confidence:.2f}).",
                confidence=primary_confidence,
                evidence_refs=[f"session:{session_id}", f"prev_user:{prev_user}"],
            )
            await self.store.store_observation(obs)
            count += 1

        # Known to unknown
        elif prev_user is not None and primary_user_id is None and len(unknown) > 0:
            obs = RecognitionObservation(
                session_id=session_id,
                user_id=prev_user,
                event_type="known_to_unknown",
                content="Previously recognized user is no longer matched.",
                confidence=0.0,
                evidence_refs=[f"session:{session_id}"],
            )
            await self.store.store_observation(obs)
            count += 1

        return count

    async def _write_body_state_observations(
        self,
        session_id: UUID,
        engagement_result: EngagementResult,
        user_id: UUID | None,
    ) -> int:
        """Write body-state observations only on meaningful transitions.

        Meaningful events:
        - pose_first_seen: first time pose is detected in this session
        - pose_lost: pose was visible but is no longer detected
        - posture_change: posture label changed
        - engagement_change: engagement level changed
        """
        count = 0
        prev_pose_visible = self._prev_pose_visible.get(session_id, False)
        prev_posture = self._prev_posture.get(session_id, PostureLabel.UNKNOWN)
        prev_engagement = self._prev_engagement.get(session_id, EngagementLevel.MEDIUM)

        # Pose first seen
        if not prev_pose_visible and engagement_result.pose_visible:
            obs = BodyStateObservation(
                session_id=session_id,
                user_id=user_id,
                event_type="pose_first_seen",
                content=(
                    f"Body pose detected for the first time."
                    f" Posture: {engagement_result.posture},"
                    f" lean: {engagement_result.lean}."
                ),
                confidence=0.8,
                evidence_refs=[f"session:{session_id}"],
            )
            await self.store.store_observation(obs)
            count += 1
            return count  # Don't double-report on same frame

        # Pose lost
        if prev_pose_visible and not engagement_result.pose_visible:
            obs = BodyStateObservation(
                session_id=session_id,
                user_id=user_id,
                event_type="pose_lost",
                content="Body pose is no longer detected.",
                confidence=0.8,
                evidence_refs=[f"session:{session_id}"],
            )
            await self.store.store_observation(obs)
            count += 1
            return count

        if not engagement_result.pose_visible:
            return count

        # Posture change
        if (
            engagement_result.posture != PostureLabel.UNKNOWN
            and prev_posture != PostureLabel.UNKNOWN
            and engagement_result.posture != prev_posture
        ):
            obs = BodyStateObservation(
                session_id=session_id,
                user_id=user_id,
                event_type="posture_change",
                content=(
                    f"Posture changed from {prev_posture} to {engagement_result.posture}."
                ),
                confidence=0.7,
                evidence_refs=[f"session:{session_id}"],
            )
            await self.store.store_observation(obs)
            count += 1

        # Engagement change
        if engagement_result.level != prev_engagement:
            obs = BodyStateObservation(
                session_id=session_id,
                user_id=user_id,
                event_type="engagement_change",
                content=(
                    f"Engagement changed from {prev_engagement} to {engagement_result.level}"
                    f" (score: {engagement_result.score:.2f})."
                ),
                confidence=0.7,
                evidence_refs=[f"session:{session_id}"],
            )
            await self.store.store_observation(obs)
            count += 1

        return count

    async def create_session(self, user_id: UUID | None = None) -> Session:
        """Create a new interaction session."""
        session = Session(user_id=user_id)
        stored = await self.store.create_session(session)
        self.state_manager.get_or_create(stored.id, user_id=user_id)
        return stored

    async def get_session_state(self, session_id: UUID) -> InteractionState | None:
        """Get the current interaction state for a session."""
        session = await self.store.get_session(session_id)
        if session is None:
            return None
        return self.state_manager.get_or_create(session_id, user_id=session.user_id)

    async def query_memories(
        self,
        user_id: UUID,
        query: str = "",
        limit: int = 10,
    ) -> MemoryQueryResult:
        """Query memories for a user."""
        return await retrieve_memories(
            store=self.store,
            user_id=user_id,
            query=query,
            limit=limit,
        )

    async def respond(
        self,
        session_id: UUID,
        user_id: UUID | None = None,
        context: str = "",
    ) -> SupportiveResponse:
        """Generate a supportive response for the current session."""
        state = self.state_manager.get_or_create(session_id, user_id=user_id)

        # Retrieve memories if we have a user
        memories = MemoryQueryResult(
            user_id=user_id or UUID("00000000-0000-0000-0000-000000000000"),
            query=context,
        )
        if user_id:
            memories = await retrieve_memories(
                store=self.store,
                user_id=user_id,
                query=context,
                limit=5,
            )

        # Plan strategy
        strategy = self.planner.plan(state, memories)

        # Generate response
        response = self.dialogue.generate(strategy, state, memories)

        # Update state
        self.state_manager.increment_turn(session_id)

        return response

    async def shutdown(self) -> None:
        """Shut down the orchestrator."""
        await self.store.close()
