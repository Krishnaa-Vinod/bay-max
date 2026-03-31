"""Assistant voice state machine (Iteration 013).

Provides a single authoritative state machine for the voice assistant turn-taking loop.
This centralizes the decision of when the assistant is allowed to listen, think, speak,
or be interrupted.

State transitions:
    idle ─[activation]─► armed ─[voice detected]─► listening
                              ▲                         │
                              │   ┌──────────────────┬──┘
                              │   ▼                  ▼
                       cooldown ◄── speaking ◄── thinking
                              │        │
                              ▼        ▼
                           idle   interrupted ─► listening/idle
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from baymax.live.schemas import ActivationMode, AssistantVoiceState

logger = logging.getLogger(__name__)


@dataclass
class StateTransition:
    """Record of a state transition for debugging and telemetry."""

    from_state: AssistantVoiceState
    to_state: AssistantVoiceState
    trigger: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class AssistantStateMachine:
    """Manages voice assistant state transitions.

    This class owns the canonical state for the assistant voice loop. All state
    changes should go through this class to maintain consistency across the
    backend, WebSocket telemetry, and UI.

    The state machine supports three activation modes:
    - push_to_talk: Explicit activation via button/API
    - continuous_vad: Always listening with VAD segmentation
    - wake_phrase_gate: ASR-gated phrase activation

    Follow-up behavior:
    - After speaking, enters a brief follow-up window (cooldown state)
    - During follow-up, user can continue naturally without re-activation
    - If no follow-up within timeout, returns to idle or armed based on mode
    """

    # Valid transitions from each state
    VALID_TRANSITIONS: dict[AssistantVoiceState, set[AssistantVoiceState]] = {
        AssistantVoiceState.IDLE: {
            AssistantVoiceState.ARMED,
            AssistantVoiceState.LISTENING,  # For push_to_talk direct activation
        },
        AssistantVoiceState.ARMED: {
            AssistantVoiceState.LISTENING,
            AssistantVoiceState.IDLE,
        },
        AssistantVoiceState.LISTENING: {
            AssistantVoiceState.THINKING,
            AssistantVoiceState.IDLE,  # User cancelled or VAD timeout with no speech
            AssistantVoiceState.ARMED,  # Back to armed if wake phrase not matched
        },
        AssistantVoiceState.THINKING: {
            AssistantVoiceState.SPEAKING,
            AssistantVoiceState.COOLDOWN,  # Skip speaking (text-only mode)
            AssistantVoiceState.IDLE,  # Error during processing
        },
        AssistantVoiceState.SPEAKING: {
            AssistantVoiceState.COOLDOWN,
            AssistantVoiceState.INTERRUPTED,
            AssistantVoiceState.IDLE,  # Error or forced stop
        },
        AssistantVoiceState.COOLDOWN: {
            AssistantVoiceState.LISTENING,  # User follow-up detected
            AssistantVoiceState.ARMED,  # Follow-up window expired, stay ready
            AssistantVoiceState.IDLE,  # Follow-up window expired, go idle
        },
        AssistantVoiceState.INTERRUPTED: {
            AssistantVoiceState.LISTENING,  # Process barge-in
            AssistantVoiceState.IDLE,  # Cancel barge-in
            AssistantVoiceState.ARMED,  # Return to armed after interrupt handled
        },
    }

    def __init__(
        self,
        activation_mode: ActivationMode = ActivationMode.CONTINUOUS_VAD,
        follow_up_window_sec: float = 4.0,
        cooldown_duration_sec: float = 1.5,
        stay_armed_after_follow_up: bool = True,
        on_state_change: Callable[[StateTransition], None] | None = None,
    ) -> None:
        """Initialize the state machine.

        Args:
            activation_mode: How the assistant activates for listening.
            follow_up_window_sec: How long to wait for follow-up after speaking.
            cooldown_duration_sec: Brief pause after TTS before accepting new input.
            stay_armed_after_follow_up: If True, return to armed after follow-up expires
                (unless in push_to_talk mode). If False, always return to idle.
            on_state_change: Optional callback invoked on every state transition.
        """
        self._state = AssistantVoiceState.IDLE
        self._activation_mode = activation_mode
        self._follow_up_window_sec = follow_up_window_sec
        self._cooldown_duration_sec = cooldown_duration_sec
        self._stay_armed_after_follow_up = stay_armed_after_follow_up
        self._on_state_change = on_state_change

        # Timing tracking
        self._state_entered_at: float = time.time()
        self._follow_up_expires_at: float = 0.0
        self._cooldown_expires_at: float = 0.0

        # Transition history for debugging
        self._history: list[StateTransition] = []
        self._max_history = 100

    @property
    def state(self) -> AssistantVoiceState:
        """Current assistant state."""
        return self._state

    @property
    def activation_mode(self) -> ActivationMode:
        """Current activation mode."""
        return self._activation_mode

    @activation_mode.setter
    def activation_mode(self, mode: ActivationMode) -> None:
        """Update activation mode (e.g., user changed settings)."""
        self._activation_mode = mode
        logger.info("Activation mode changed to %s", mode.value)

    @property
    def follow_up_window_sec(self) -> float:
        """Follow-up window duration."""
        return self._follow_up_window_sec

    @follow_up_window_sec.setter
    def follow_up_window_sec(self, value: float) -> None:
        """Update follow-up window duration."""
        self._follow_up_window_sec = max(0.0, value)

    @property
    def is_follow_up_active(self) -> bool:
        """Whether we're in the follow-up window after speaking."""
        if self._state != AssistantVoiceState.COOLDOWN:
            return False
        return time.time() < self._follow_up_expires_at

    @property
    def follow_up_remaining_sec(self) -> float:
        """Seconds remaining in follow-up window."""
        if not self.is_follow_up_active:
            return 0.0
        return max(0.0, self._follow_up_expires_at - time.time())

    @property
    def is_cooldown_active(self) -> bool:
        """Whether the post-speech cooldown is still active."""
        if self._state != AssistantVoiceState.COOLDOWN:
            return False
        return time.time() < self._cooldown_expires_at

    @property
    def history(self) -> list[StateTransition]:
        """Recent state transition history."""
        return list(self._history)

    @property
    def state_duration_sec(self) -> float:
        """How long we've been in the current state."""
        return time.time() - self._state_entered_at

    def can_transition(self, to_state: AssistantVoiceState) -> bool:
        """Check if a transition to the given state is valid."""
        valid_targets = self.VALID_TRANSITIONS.get(self._state, set())
        return to_state in valid_targets

    def transition(self, to_state: AssistantVoiceState, trigger: str = "") -> bool:
        """Attempt to transition to a new state.

        Args:
            to_state: Target state.
            trigger: Description of what triggered this transition.

        Returns:
            True if transition succeeded, False if invalid.
        """
        if not self.can_transition(to_state):
            logger.warning(
                "Invalid state transition %s -> %s (trigger: %s)",
                self._state.value,
                to_state.value,
                trigger,
            )
            return False

        from_state = self._state
        now = time.time()

        # Create transition record
        transition = StateTransition(
            from_state=from_state,
            to_state=to_state,
            trigger=trigger,
        )

        # Update state
        self._state = to_state
        self._state_entered_at = now

        # Handle state-specific timing
        if to_state == AssistantVoiceState.COOLDOWN:
            self._cooldown_expires_at = now + self._cooldown_duration_sec
            self._follow_up_expires_at = now + self._follow_up_window_sec
        elif to_state == AssistantVoiceState.SPEAKING:
            # Reset follow-up timing when we start speaking
            self._follow_up_expires_at = 0.0

        # Record history
        self._history.append(transition)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        logger.debug(
            "State transition: %s -> %s (trigger: %s)",
            from_state.value,
            to_state.value,
            trigger,
        )

        # Notify callback
        if self._on_state_change:
            try:
                self._on_state_change(transition)
            except Exception as exc:
                logger.warning("State change callback error: %s", exc)

        return True

    def tick(self) -> AssistantVoiceState | None:
        """Called periodically to handle time-based transitions.

        Returns the new state if an automatic transition occurred, None otherwise.
        """
        now = time.time()

        # Handle follow-up window expiry
        if self._state == AssistantVoiceState.COOLDOWN:
            # First check if basic cooldown is done
            if now >= self._cooldown_expires_at:
                # Then check if follow-up window is also done
                if now >= self._follow_up_expires_at:
                    # Decide where to go based on activation mode and settings
                    if self._activation_mode == ActivationMode.PUSH_TO_TALK:
                        # Push-to-talk always returns to idle
                        self.transition(AssistantVoiceState.IDLE, "follow_up_expired_ptt")
                        return AssistantVoiceState.IDLE
                    elif self._stay_armed_after_follow_up:
                        # Return to armed for continuous modes
                        self.transition(AssistantVoiceState.ARMED, "follow_up_expired")
                        return AssistantVoiceState.ARMED
                    else:
                        # Go to idle if stay_armed_after_follow_up is False
                        self.transition(AssistantVoiceState.IDLE, "follow_up_expired_no_arm")
                        return AssistantVoiceState.IDLE

        return None

    # --- Convenience methods for common transitions ---

    def activate(self) -> bool:
        """Transition from idle to armed (start listening intent)."""
        if self._state == AssistantVoiceState.IDLE:
            return self.transition(AssistantVoiceState.ARMED, "user_activation")
        return False

    def deactivate(self) -> bool:
        """Return to idle state."""
        if self._state in (
            AssistantVoiceState.ARMED,
            AssistantVoiceState.COOLDOWN,
            AssistantVoiceState.INTERRUPTED,
        ):
            return self.transition(AssistantVoiceState.IDLE, "user_deactivation")
        return False

    def voice_detected(self) -> bool:
        """Transition from armed to listening when speech starts."""
        if self._state == AssistantVoiceState.ARMED:
            return self.transition(AssistantVoiceState.LISTENING, "voice_activity")
        elif self._state == AssistantVoiceState.COOLDOWN:
            # Follow-up speech during cooldown
            return self.transition(AssistantVoiceState.LISTENING, "follow_up_speech")
        return False

    def end_of_speech(self) -> bool:
        """Transition from listening to thinking when user stops speaking."""
        if self._state == AssistantVoiceState.LISTENING:
            return self.transition(AssistantVoiceState.THINKING, "end_of_speech")
        return False

    def start_speaking(self) -> bool:
        """Transition from thinking to speaking when TTS begins."""
        if self._state == AssistantVoiceState.THINKING:
            return self.transition(AssistantVoiceState.SPEAKING, "tts_start")
        return False

    def done_speaking(self) -> bool:
        """Transition from speaking to cooldown when TTS finishes."""
        if self._state == AssistantVoiceState.SPEAKING:
            return self.transition(AssistantVoiceState.COOLDOWN, "tts_complete")
        return False

    def interrupt(self) -> bool:
        """User interrupted while assistant was speaking (barge-in)."""
        if self._state == AssistantVoiceState.SPEAKING:
            return self.transition(AssistantVoiceState.INTERRUPTED, "user_barge_in")
        return False

    def handle_interrupt(self) -> bool:
        """Process the barge-in by transitioning to listening."""
        if self._state == AssistantVoiceState.INTERRUPTED:
            return self.transition(AssistantVoiceState.LISTENING, "process_barge_in")
        return False

    def reset(self) -> None:
        """Force reset to idle state (for error recovery)."""
        if self._state != AssistantVoiceState.IDLE:
            self.transition(AssistantVoiceState.IDLE, "forced_reset")
        self._follow_up_expires_at = 0.0
        self._cooldown_expires_at = 0.0

    def get_status_dict(self) -> dict:
        """Return current state as a dictionary for API/telemetry."""
        return {
            "assistant_state": self._state.value,
            "activation_mode": self._activation_mode.value,
            "follow_up_window_active": self.is_follow_up_active,
            "follow_up_remaining_sec": round(self.follow_up_remaining_sec, 2),
            "cooldown_active": self.is_cooldown_active,
            "state_duration_sec": round(self.state_duration_sec, 2),
        }
