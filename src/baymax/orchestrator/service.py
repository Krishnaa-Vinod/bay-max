"""Orchestrator service coordinating the end-to-end flow."""

import collections
import logging
import os
import time
from datetime import datetime
from uuid import UUID

import numpy as np

from baymax.config.settings import get_settings
from baymax.core.enums import (
    CorrectionAction,
    EngagementLevel,
    FactStatus,
    MemoryStatus,
    PostureLabel,
    ResponseStrategy,
    TurnIntent,
    TurnRole,
)
from baymax.dialogue.factory import AVAILABLE_BACKENDS, create_dialogue_provider, get_active_model
from baymax.dialogue.interfaces import DialogueProvider
from baymax.dialogue.intent_router import classify_turn_intent
from baymax.dialogue.prompt_builder import build_prompt_context
from baymax.dialogue.rule_based import RuleBasedDialogue
from baymax.dialogue.safety import check_safety
from baymax.memory.consolidate import consolidate_session
from baymax.memory.embedding import StubTextEmbedder, TextEmbedder
from baymax.memory.retrieve import retrieve_memories
from baymax.memory.store_sqlite import SQLiteMetadataStore
from baymax.memory.vector_store import LanceDBVectorStore, StubVectorStore, VectorStore
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
from baymax.schemas.memory import (
    ChatTurn,
    ConsolidationResult,
    MemoryCorrectionRequest,
    MemoryCorrectionResult,
    MemoryHit,
    MemoryQueryResult,
    MemorySummaryResponse,
)
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
from baymax.schemas.response import DialogueBackendInfo, SupportiveResponse
from baymax.schemas.session import Session
from baymax.schemas.user import FaceEnrollment, UserProfile, UserProfileCreate
from baymax.state.manager import StateManager
from baymax.state.models import InteractionState
from baymax.tools.web_fetch import fetch_url, summarize_sources
from baymax.tools.web_search import search_web

logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinates the flow from frame ingest to supportive response."""

    def __init__(self, db_path: str = "baymax.db") -> None:
        self.store = SQLiteMetadataStore(db_path=db_path)
        self._initialized = False
        self.state_manager = StateManager()
        self.planner = SupportivePlanner()
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
        self._face_match_relaxed_threshold = settings.face_match_relaxed_threshold
        self._face_recognition_grace_sec = settings.face_recognition_grace_sec
        self._max_faces = settings.max_faces_per_frame
        self._pose_backend = settings.pose_backend
        self._pose_min_confidence = settings.pose_min_confidence
        self._enable_annotations = settings.enable_annotations
        self._artifact_dir = settings.artifact_dir

        # Iteration 004: memory settings
        self._vector_backend = settings.vector_backend
        self._text_embedding_model = settings.text_embedding_model
        self._memory_top_k = settings.memory_top_k
        self._memory_similarity_threshold = settings.memory_similarity_threshold
        self._smoothing_window = settings.session_smoothing_window
        self._enable_consolidation = settings.enable_memory_consolidation

        # Iteration 005: dialogue settings
        self._dialogue_backend_name = settings.dialogue_backend
        self._dialogue_top_k_memories = settings.dialogue_top_k_memories
        self._dialogue_memory_relevance_threshold = settings.dialogue_memory_relevance_threshold
        self._dialogue_max_history_turns = settings.dialogue_max_history_turns
        self._enable_rule_based_fallback = settings.enable_rule_based_fallback
        self._enable_safe_health_mode = settings.enable_safe_health_mode
        self._dialogue_active_model = get_active_model(settings)
        self._dialogue_requested_backend = settings.dialogue_backend
        self._dialogue_requested_model = get_active_model(settings)
        self._dialogue_fallback_warning = ""

        # Iteration 011: web-tool broker settings
        self._enable_web_tools = settings.enable_web_tools
        self._web_search_provider = settings.web_search_provider
        self._web_search_max_results = settings.web_search_max_results
        self._web_fetch_timeout_sec = settings.web_fetch_timeout_sec

        # Initialise the dialogue provider via factory (with fallback support)
        self.dialogue: DialogueProvider = create_dialogue_provider(settings)
        self._fallback_dialogue = RuleBasedDialogue()

        if (
            self.dialogue.backend_name == "rule_based"
            and self._dialogue_requested_backend.lower() != "rule_based"
        ):
            self._dialogue_fallback_warning = "LLM unavailable — using rule-based fallback"
        elif self.dialogue.backend_name == "rule_based":
            self._dialogue_fallback_warning = "Limited mode active: rule-based dialogue"

        # Perception backends (lazy-loaded)
        self._detector: FaceDetector | None = None
        self._recognizer: FaceRecognizer | None = None
        self._pose_estimator: PoseEstimator | None = None

        # Iteration 004: text embedding + vector store (lazy-loaded)
        self._text_embedder: TextEmbedder | None = None
        self._vector_store: VectorStore | None = None

        # Cached enrolled embeddings (refreshed on enroll)
        self._enrolled_embeddings: list[FaceEmbeddingRecord] = []

        # Track previous state per session for observation writing
        self._prev_user_ids: dict[UUID, UUID | None] = {}
        self._prev_posture: dict[UUID, PostureLabel] = {}
        self._prev_engagement: dict[UUID, EngagementLevel] = {}
        self._prev_pose_visible: dict[UUID, bool] = {}

        # Iteration 004: temporal smoothing buffers
        self._engagement_buffer: dict[UUID, collections.deque] = {}
        self._posture_buffer: dict[UUID, collections.deque] = {}
        self._recent_recognitions: dict[UUID, tuple[UUID, float]] = {}

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

    def _ensure_text_embedder(self) -> TextEmbedder:
        if self._text_embedder is None:
            try:
                from baymax.memory.embedding import SentenceTransformerEmbedder
                self._text_embedder = SentenceTransformerEmbedder(
                    model_name=self._text_embedding_model
                )
                logger.info("SentenceTransformerEmbedder loaded")
            except Exception:
                logger.warning(
                    "Failed to load SentenceTransformerEmbedder, using stub",
                    exc_info=True,
                )
                self._text_embedder = StubTextEmbedder()
        return self._text_embedder

    def _ensure_vector_store(self) -> VectorStore:
        if self._vector_store is None:
            if self._vector_backend == "lancedb":
                try:
                    settings = get_settings()
                    db_path = f"{settings.data_dir}/lancedb"
                    self._vector_store = LanceDBVectorStore(db_path=db_path)
                    logger.info("LanceDBVectorStore at %s", db_path)
                except Exception:
                    logger.warning(
                        "Failed to init LanceDBVectorStore, using stub",
                        exc_info=True,
                    )
                    self._vector_store = StubVectorStore()
            else:
                self._vector_store = StubVectorStore()
        return self._vector_store

    def _smoothed_engagement(self, session_id: UUID, level: EngagementLevel) -> EngagementLevel:
        """Apply temporal smoothing to engagement level changes."""
        buf = self._engagement_buffer.setdefault(
            session_id, collections.deque(maxlen=self._smoothing_window)
        )
        buf.append(level)
        if len(buf) < self._smoothing_window:
            return level
        # Return the most frequent level in the window
        counter = collections.Counter(buf)
        return counter.most_common(1)[0][0]

    def _smoothed_posture(self, session_id: UUID, posture: PostureLabel) -> PostureLabel:
        """Apply temporal smoothing to posture label changes."""
        buf = self._posture_buffer.setdefault(
            session_id, collections.deque(maxlen=self._smoothing_window)
        )
        buf.append(posture)
        if len(buf) < self._smoothing_window:
            return posture
        counter = collections.Counter(buf)
        return counter.most_common(1)[0][0]

    async def initialize(self) -> None:
        """Initialize the orchestrator and its dependencies."""
        if self._initialized:
            return
        await self.store.initialize()
        # Pre-load enrolled embeddings
        self._enrolled_embeddings = await self.store.get_all_face_embeddings()
        logger.info("Loaded %d enrolled face embeddings", len(self._enrolled_embeddings))
        self._initialized = True

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
        """Run detection + recognition + pose + engagement on a frame and update state."""
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

            candidate_user_id, score = recognizer.best_match(
                det.embedding,
                self._enrolled_embeddings,
            )

            user_id: UUID | None = None
            if candidate_user_id is not None and score >= self._face_match_threshold:
                user_id = candidate_user_id
            elif candidate_user_id is not None and score >= self._face_match_relaxed_threshold:
                recent = self._recent_recognitions.get(session_id)
                if recent is not None:
                    recent_user_id, recent_ts = recent
                    if recent_user_id == candidate_user_id and (time.time() - recent_ts) <= self._face_recognition_grace_sec:
                        user_id = candidate_user_id

            if user_id is not None:
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
                    self._recent_recognitions[session_id] = (user_id, time.time())

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
        raw_posture = compute_posture(pose_result)
        lean = compute_lean(pose_result)
        motion = self.motion_tracker.update(pose_result)
        face_visible = len(detections) > 0
        face_recognized = len(recognized) > 0

        # Apply temporal smoothing (Iteration 004)
        posture = self._smoothed_posture(session_id, raw_posture)

        engagement_result = compute_engagement(
            posture=posture,
            lean=lean,
            motion=motion,
            face_visible=face_visible,
            face_recognized=face_recognized,
            pose_visible=pose_result.pose_present,
        )

        # Smooth engagement level
        smoothed_level = self._smoothed_engagement(session_id, engagement_result.level)
        engagement_result = EngagementResult(
            level=smoothed_level,
            score=engagement_result.score,
            posture=posture,
            lean=lean,
            motion=motion,
            face_visible=face_visible,
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
        # Only write if smoothed values differ from previous (suppresses noisy writes)
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
        """Write recognition observations only on meaningful events."""
        prev_user = self._prev_user_ids.get(session_id)
        count = 0

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
        """Write body-state observations only on meaningful transitions."""
        count = 0
        prev_pose_visible = self._prev_pose_visible.get(session_id, False)
        prev_posture = self._prev_posture.get(session_id, PostureLabel.UNKNOWN)
        prev_engagement = self._prev_engagement.get(session_id, EngagementLevel.MEDIUM)

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
            return count

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

    # --- Iteration 004: Typed conversation turns (T403) ---

    async def add_turn(
        self,
        session_id: UUID,
        role: TurnRole,
        text: str,
        user_id: UUID | None = None,
        source: str = "typed",
    ) -> ChatTurn:
        """Add a typed or spoken conversation turn to a session."""
        turn = ChatTurn(
            session_id=session_id,
            user_id=user_id,
            role=role,
            text=text,
            source=source,
        )
        stored = await self.store.store_chat_turn(turn)
        if role == TurnRole.USER:
            self.state_manager.increment_turn(session_id)
        return stored

    async def get_turns(self, session_id: UUID) -> list[ChatTurn]:
        """Get all chat turns for a session."""
        return await self.store.get_chat_turns(session_id)

    # --- Iteration 004: Session consolidation (T404) ---

    async def consolidate(
        self,
        session_id: UUID,
        user_id: UUID,
    ) -> ConsolidationResult:
        """Consolidate a session into episodic memories and semantic facts.

        Also embeds new memories in the vector store if available.
        """
        result = await consolidate_session(
            store=self.store,
            session_id=session_id,
            user_id=user_id,
        )

        # Embed new memories in vector store
        if result.summary and result.episodic_memories_created > 0:
            try:
                embedder = self._ensure_text_embedder()
                vs = self._ensure_vector_store()
                # Re-fetch the episodic memories we just created
                mq_result = await retrieve_memories(
                    store=self.store,
                    user_id=user_id,
                    limit=result.episodic_memories_created + 5,
                )
                for mem in mq_result.episodic_memories:
                    vec = embedder.embed_single(mem.content)
                    await vs.add(
                        memory_id=mem.id,
                        embedding=vec,
                        metadata={
                            "user_id": str(user_id),
                            "memory_type": "episodic",
                            "content": mem.content,
                        },
                    )
            except Exception:
                logger.warning("Failed to embed memories in vector store", exc_info=True)

        return result

    # --- Iteration 005: Dialogue backends info ---

    def get_dialogue_backends(self) -> DialogueBackendInfo:
        """Return information about configured dialogue backends."""
        return DialogueBackendInfo(
            available_backends=AVAILABLE_BACKENDS,
            active_backend=self.dialogue.backend_name,
            active_model=self.dialogue.model_name,
            fallback_enabled=self._enable_rule_based_fallback,
        )

    @property
    def dialogue_requested_backend(self) -> str:
        """Configured dialogue backend before runtime fallback resolution."""
        return self._dialogue_requested_backend

    @property
    def dialogue_requested_model(self) -> str:
        """Configured dialogue model before runtime fallback resolution."""
        return self._dialogue_requested_model

    @property
    def dialogue_fallback_warning(self) -> str:
        """Human-readable fallback warning for UI surfaces."""
        return self._dialogue_fallback_warning

    # --- Iteration 004: Memory-aware responses (T406) + Iteration 005: Grounded dialogue ---

    async def respond(
        self,
        session_id: UUID,
        user_id: UUID | None = None,
        context: str = "",
        modality: str = "text",
    ) -> SupportiveResponse:
        """Generate a supportive response for the current session.

        With Iteration 005 this method:
          - Checks user context for safety concerns before generating
          - Builds a grounded prompt context from state + memories + recent turns
          - Routes through the configured dialogue backend
          - Falls back to rule-based dialogue if the backend fails
        """
        state = self.state_manager.get_or_create(session_id, user_id=user_id)
        turn_intent: TurnIntent | None = None

        # Safety-check the incoming context (if safe_health_mode is on)
        safety_flags: list[str] = []
        if self._enable_safe_health_mode and context:
            safety_decision = check_safety(context)
            if not safety_decision.is_safe:
                safety_flags = safety_decision.flags
                self.state_manager.increment_turn(session_id)
                redirect_text = safety_decision.redirect_response or (
                    "I'm here to support you. "
                    "For medical concerns, please consult a qualified professional."
                )
                return SupportiveResponse(
                    session_id=session_id,
                    user_id=user_id,
                    strategy=self.planner.plan(
                        state,
                        MemoryQueryResult(
                            user_id=user_id or UUID("00000000-0000-0000-0000-000000000000"),
                            query=context,
                        ),
                        context=context,
                        turn_intent=TurnIntent.DIRECT_QUESTION,
                    ),
                    message=redirect_text,
                    spoken_text=redirect_text,
                    display_text=redirect_text,
                    backend=self.dialogue.backend_name,
                    model_name=self.dialogue.model_name,
                    fallback_used=False,
                    safety_flags=safety_flags,
                    limited_mode=(self.dialogue.backend_name == "rule_based"),
                    limited_mode_reason=(
                        "LLM unavailable; using deterministic templates"
                        if self.dialogue.backend_name == "rule_based" else ""
                    ),
                    state_summary={
                        "engagement": state.engagement_level.value,
                        "posture": state.posture.value,
                        "turn_count": str(state.turn_count),
                    },
                )

        # Retrieve memories if we have a user
        memories = MemoryQueryResult(
            user_id=user_id or UUID("00000000-0000-0000-0000-000000000000"),
            query=context,
        )
        memory_refs: list[str] = []
        source_refs: list[dict] = []
        tool_usage: list[str] = []
        if user_id:
            memories = await retrieve_memories(
                store=self.store,
                user_id=user_id,
                query=context,
                limit=self._dialogue_top_k_memories,
            )
            # Also try semantic search via vector store
            semantic_hits = await self._semantic_search(user_id, context)
            for hit in semantic_hits:
                memory_refs.append(hit.content)

            # Add SQL-retrieved memories to refs
            for m in memories.episodic_memories:
                if m.content not in memory_refs:
                    memory_refs.append(m.content)
            for f in memories.semantic_facts:
                if f.content not in memory_refs:
                    memory_refs.append(f.content)

        # Iteration 011: run web tools only when likely needed.
        if self._should_use_web_tools(context):
            try:
                tool_usage.append("search_web")
                web_results = search_web(
                    query=context,
                    provider=self._web_search_provider,
                    max_results=self._web_search_max_results,
                    timeout_sec=self._web_fetch_timeout_sec,
                )
                source_refs = [item.to_dict() for item in web_results]

                if source_refs:
                    # Optionally fetch top result for richer grounding.
                    top_url = source_refs[0].get("url")
                    if isinstance(top_url, str) and top_url:
                        try:
                            tool_usage.append("fetch_url")
                            fetched = fetch_url(top_url, timeout_sec=self._web_fetch_timeout_sec)
                            source_refs.insert(0, fetched.to_dict())
                        except Exception as fetch_exc:
                            logger.info("web fetch failed for %s: %s", top_url, fetch_exc)

                    tool_usage.append("summarize_sources")
                    summary = summarize_sources(source_refs)
                    if summary:
                        memory_refs.append(f"Web context:\n{summary}")
            except Exception as web_exc:
                logger.info("Web tool path unavailable: %s", web_exc)

        # Fetch recent turns for grounded prompting
        recent_turns = []
        if user_id or session_id:
            try:
                all_turns = await self.store.get_chat_turns(session_id)
                # Keep the most recent N turns
                recent_turns = all_turns[-self._dialogue_max_history_turns:]
            except Exception:
                logger.warning("Failed to fetch recent turns for prompt", exc_info=True)

        # Classify intent for user-initiated turns.
        if modality != "system":
            turn_intent = classify_turn_intent(
                context,
                recent_turns=[
                    {"role": t.role.value, "text": t.text}
                    for t in recent_turns[-6:]
                ],
            )

        # Plan strategy
        strategy = self.planner.plan(
            state,
            memories,
            context=context,
            turn_intent=turn_intent,
            proactive=(modality == "system"),
        )

        # If fresh information is likely needed, keep response answer-first with web grounding.
        if strategy == ResponseStrategy.ANSWER and self._should_use_web_tools(context):
            strategy = ResponseStrategy.WEB_ANSWER

        # Build grounded prompt context (Iteration 005)
        backend_limited_mode = self.dialogue.backend_name == "rule_based"
        prompt_context = build_prompt_context(
            strategy=strategy,
            state=state,
            memories=memories,
            recent_turns=recent_turns,
            context=context,
            top_k_memories=self._dialogue_top_k_memories,
            turn_intent=turn_intent,
            memory_relevance_threshold=self._dialogue_memory_relevance_threshold,
            backend_limited_mode=backend_limited_mode,
        )

        # Generate response — with fallback on failure
        response: SupportiveResponse
        fallback_used = False

        try:
            response = self.dialogue.generate(strategy, state, memories, prompt_context)
        except Exception as exc:
            if self._enable_rule_based_fallback:
                logger.warning(
                    "Dialogue backend '%s' failed, falling back to rule_based: %s",
                    self.dialogue.backend_name,
                    exc,
                )
                response = self._fallback_dialogue.generate(strategy, state, memories)
                fallback_used = True
            else:
                raise

        # Attach memory refs, state summary, and backend metadata
        state_summary = {
            "engagement": state.engagement_level.value,
            "posture": state.posture.value,
            "turn_count": str(state.turn_count),
        }
        if state.user_display_name:
            state_summary["user"] = state.user_display_name

        response.memory_refs = prompt_context.memory_refs[:self._dialogue_top_k_memories]
        response.state_summary = state_summary
        response.fallback_used = fallback_used
        response.tool_usage = tool_usage
        response.source_refs = source_refs
        response.metadata["turn_modality"] = modality
        if turn_intent is not None:
            response.metadata["turn_intent"] = turn_intent.value
        if safety_flags:
            response.safety_flags = safety_flags

        response.limited_mode = backend_limited_mode or fallback_used
        if response.limited_mode and not response.limited_mode_reason:
            response.limited_mode_reason = (
                "LLM unavailable; using deterministic templates"
            )

        # Keep spoken output natural; keep source list in display text only.
        base_text = response.message.strip()
        response.spoken_text = response.spoken_text or base_text
        response.display_text = response.display_text or base_text

        if source_refs:
            rendered_refs = [
                f"- {item.get('title', 'Source')}: {item.get('url', '')}"
                for item in source_refs[:3]
            ]
            response.display_text = f"{response.display_text}\n\nSources:\n" + "\n".join(rendered_refs)

        # Preserve legacy message field for existing callers.
        response.message = response.display_text

        # Update state
        self.state_manager.increment_turn(session_id)

        return response

    def _should_use_web_tools(self, context: str) -> bool:
        """Heuristic gate for web tool usage to avoid unnecessary live calls."""
        if not self._enable_web_tools:
            return False

        lowered = context.lower().strip()
        if not lowered:
            return False

        if os.getenv("BAYMAX_WEB_TOOLS_FORCE", "").lower() in ("1", "true", "yes"):
            return True

        explicit = ("search" in lowered) or ("look up" in lowered) or ("verify" in lowered)
        live_fact = any(
            key in lowered
            for key in ("today", "latest", "current", "price", "news", "recent", "this week")
        )
        uncertainty = any(
            key in lowered
            for key in ("not sure", "uncertain", "double-check", "can you confirm")
        )
        return explicit or live_fact or uncertainty

    async def _semantic_search(
        self,
        user_id: UUID,
        query: str,
    ) -> list[MemoryHit]:
        """Search for semantically similar memories via vector store."""
        if not query:
            return []
        try:
            embedder = self._ensure_text_embedder()
            vs = self._ensure_vector_store()
            query_vec = embedder.embed_single(query)
            results = await vs.search(
                query_embedding=query_vec,
                top_k=self._memory_top_k,
                filter_user_id=user_id,
            )
            hits = []
            for r in results:
                if r["score"] >= self._memory_similarity_threshold:
                    hits.append(MemoryHit(
                        memory_id=UUID(r["memory_id"]),
                        memory_type=r["metadata"].get("memory_type", "episodic"),
                        content=r["metadata"].get("content", ""),
                        score=r["score"],
                        user_id=user_id,
                    ))
            return hits
        except Exception:
            logger.warning("Semantic search failed", exc_info=True)
            return []

    # --- Iteration 004: Memory summary (T407) ---

    async def get_memory_summary(self, user_id: UUID) -> MemorySummaryResponse:
        """Get a summary of all memories for a user."""
        episodic_count = await self.store.count_episodic_memories(user_id)
        semantic_count = await self.store.count_semantic_facts(user_id)
        session_summaries = await self.store.get_session_summaries(user_id)

        mq_result = await retrieve_memories(
            store=self.store,
            user_id=user_id,
            limit=10,
        )

        return MemorySummaryResponse(
            user_id=user_id,
            episodic_count=episodic_count,
            semantic_count=semantic_count,
            session_summaries=session_summaries,
            recent_episodic=mq_result.episodic_memories[:5],
            confirmed_facts=mq_result.semantic_facts[:10],
        )

    # --- Iteration 004: Memory correction (T407) ---

    async def correct_memory(
        self, request: MemoryCorrectionRequest
    ) -> MemoryCorrectionResult:
        """Apply a correction action to a semantic fact."""
        fact = await self.store.get_semantic_fact(request.fact_id)
        if fact is None:
            raise ValueError(f"Semantic fact {request.fact_id} not found")

        previous_content = fact.content

        if request.action == CorrectionAction.CONFIRM:
            fact.status = MemoryStatus.ACTIVE
            fact.confidence = min(fact.confidence + 0.2, 1.0)
            fact.last_confirmed = datetime.utcnow()
            await self.store.update_semantic_fact(fact)
            return MemoryCorrectionResult(
                fact_id=request.fact_id,
                action=request.action,
                previous_content=previous_content,
                new_content=fact.content,
                new_status=FactStatus.CONFIRMED,
            )

        elif request.action == CorrectionAction.REJECT:
            fact.status = MemoryStatus.DELETED
            await self.store.update_semantic_fact(fact)
            # Also remove from vector store
            try:
                vs = self._ensure_vector_store()
                await vs.delete(request.fact_id)
            except Exception:
                logger.warning("Failed to delete from vector store", exc_info=True)
            return MemoryCorrectionResult(
                fact_id=request.fact_id,
                action=request.action,
                previous_content=previous_content,
                new_status=FactStatus.REJECTED,
            )

        elif request.action == CorrectionAction.UPDATE:
            if not request.updated_content:
                raise ValueError("updated_content is required for update action")
            fact.content = request.updated_content
            fact.status = MemoryStatus.CORRECTED
            fact.last_confirmed = datetime.utcnow()
            await self.store.update_semantic_fact(fact)
            return MemoryCorrectionResult(
                fact_id=request.fact_id,
                action=request.action,
                previous_content=previous_content,
                new_content=request.updated_content,
                new_status=FactStatus.CONFIRMED,
            )

        raise ValueError(f"Unknown correction action: {request.action}")

    # --- Iteration 004: Semantic memory query via vector store ---

    async def query_memory_semantic(
        self,
        user_id: UUID,
        query: str,
        top_k: int | None = None,
    ) -> list[MemoryHit]:
        """Query memories using semantic similarity."""
        return await self._semantic_search(user_id, query)

    async def shutdown(self) -> None:
        """Shut down the orchestrator."""
        if not self._initialized:
            return
        await self.store.close()
        self._initialized = False
