"""Iteration 013: Friendly Voice Assistant Core tests.

Tests for:
- AssistantVoiceState enum
- AssistantStateMachine transitions
- ActivationGate implementations
- Wake phrase detection
"""

import time
from unittest.mock import MagicMock

import pytest

from baymax.audio.activation_gate import (
    ContinuousVADGate,
    PushToTalkGate,
    WakePhraseGate,
    create_activation_gate,
)
from baymax.live.assistant_state import AssistantStateMachine, StateTransition
from baymax.live.schemas import ActivationMode, AssistantVoiceState

# --- AssistantVoiceState Enum Tests ---


class TestAssistantVoiceState:
    """Tests for AssistantVoiceState enum."""

    def test_all_states_exist(self) -> None:
        """Verify all expected states are defined."""
        expected = {"idle", "armed", "listening", "thinking", "speaking", "cooldown", "interrupted"}
        actual = {s.value for s in AssistantVoiceState}
        assert actual == expected

    def test_states_are_strings(self) -> None:
        """States should be string values for JSON serialization."""
        for state in AssistantVoiceState:
            assert isinstance(state.value, str)


class TestActivationMode:
    """Tests for ActivationMode enum."""

    def test_all_modes_exist(self) -> None:
        """Verify all expected activation modes are defined."""
        expected = {"push_to_talk", "continuous_vad", "wake_phrase_gate"}
        actual = {m.value for m in ActivationMode}
        assert actual == expected


# --- AssistantStateMachine Tests ---


class TestAssistantStateMachine:
    """Tests for AssistantStateMachine."""

    def test_initial_state_is_idle(self) -> None:
        """State machine starts in idle state."""
        sm = AssistantStateMachine()
        assert sm.state == AssistantVoiceState.IDLE

    def test_valid_transition_idle_to_armed(self) -> None:
        """Can transition from idle to armed."""
        sm = AssistantStateMachine()
        assert sm.activate()
        assert sm.state == AssistantVoiceState.ARMED

    def test_valid_transition_armed_to_listening(self) -> None:
        """Can transition from armed to listening on voice detection."""
        sm = AssistantStateMachine()
        sm.activate()
        assert sm.voice_detected()
        assert sm.state == AssistantVoiceState.LISTENING

    def test_valid_transition_listening_to_thinking(self) -> None:
        """Can transition from listening to thinking on end of speech."""
        sm = AssistantStateMachine()
        sm.activate()
        sm.voice_detected()
        assert sm.end_of_speech()
        assert sm.state == AssistantVoiceState.THINKING

    def test_valid_transition_thinking_to_speaking(self) -> None:
        """Can transition from thinking to speaking when TTS starts."""
        sm = AssistantStateMachine()
        sm.activate()
        sm.voice_detected()
        sm.end_of_speech()
        assert sm.start_speaking()
        assert sm.state == AssistantVoiceState.SPEAKING

    def test_valid_transition_speaking_to_cooldown(self) -> None:
        """Can transition from speaking to cooldown when TTS finishes."""
        sm = AssistantStateMachine()
        sm.activate()
        sm.voice_detected()
        sm.end_of_speech()
        sm.start_speaking()
        assert sm.done_speaking()
        assert sm.state == AssistantVoiceState.COOLDOWN

    def test_invalid_transition_rejected(self) -> None:
        """Invalid transitions are rejected."""
        sm = AssistantStateMachine()
        # Cannot go directly from idle to speaking
        assert not sm.transition(AssistantVoiceState.SPEAKING, "test")
        assert sm.state == AssistantVoiceState.IDLE

    def test_interrupt_from_speaking(self) -> None:
        """Can interrupt while speaking (barge-in)."""
        sm = AssistantStateMachine()
        sm.activate()
        sm.voice_detected()
        sm.end_of_speech()
        sm.start_speaking()
        assert sm.interrupt()
        assert sm.state == AssistantVoiceState.INTERRUPTED

    def test_handle_interrupt_goes_to_listening(self) -> None:
        """Handling interrupt transitions to listening."""
        sm = AssistantStateMachine()
        sm.activate()
        sm.voice_detected()
        sm.end_of_speech()
        sm.start_speaking()
        sm.interrupt()
        assert sm.handle_interrupt()
        assert sm.state == AssistantVoiceState.LISTENING

    def test_follow_up_from_cooldown(self) -> None:
        """Can start follow-up speech from cooldown."""
        sm = AssistantStateMachine()
        sm.activate()
        sm.voice_detected()
        sm.end_of_speech()
        sm.start_speaking()
        sm.done_speaking()
        assert sm.state == AssistantVoiceState.COOLDOWN
        assert sm.voice_detected()
        assert sm.state == AssistantVoiceState.LISTENING

    def test_follow_up_window_active_after_speaking(self) -> None:
        """Follow-up window is active after entering cooldown."""
        sm = AssistantStateMachine(follow_up_window_sec=5.0)
        sm.activate()
        sm.voice_detected()
        sm.end_of_speech()
        sm.start_speaking()
        sm.done_speaking()
        assert sm.is_follow_up_active
        assert sm.follow_up_remaining_sec > 0

    def test_state_history_recorded(self) -> None:
        """State transitions are recorded in history."""
        sm = AssistantStateMachine()
        sm.activate()
        sm.voice_detected()
        history = sm.history
        assert len(history) == 2
        assert history[0].from_state == AssistantVoiceState.IDLE
        assert history[0].to_state == AssistantVoiceState.ARMED
        assert history[1].from_state == AssistantVoiceState.ARMED
        assert history[1].to_state == AssistantVoiceState.LISTENING

    def test_callback_on_state_change(self) -> None:
        """Callback is invoked on state changes."""
        callback = MagicMock()
        sm = AssistantStateMachine(on_state_change=callback)
        sm.activate()
        callback.assert_called_once()
        transition = callback.call_args[0][0]
        assert isinstance(transition, StateTransition)
        assert transition.to_state == AssistantVoiceState.ARMED

    def test_reset_returns_to_idle(self) -> None:
        """Reset forces state back to idle."""
        sm = AssistantStateMachine()
        sm.activate()
        sm.voice_detected()
        sm.reset()
        assert sm.state == AssistantVoiceState.IDLE

    def test_get_status_dict(self) -> None:
        """Status dict contains expected fields."""
        sm = AssistantStateMachine(activation_mode=ActivationMode.WAKE_PHRASE_GATE)
        status = sm.get_status_dict()
        assert status["assistant_state"] == "idle"
        assert status["activation_mode"] == "wake_phrase_gate"
        assert "follow_up_window_active" in status
        assert "state_duration_sec" in status


class TestStateMachineTick:
    """Tests for time-based state transitions."""

    def test_tick_expires_follow_up_to_armed(self) -> None:
        """Tick transitions from cooldown to armed when follow-up expires (non-PTT mode)."""
        sm = AssistantStateMachine(
            activation_mode=ActivationMode.CONTINUOUS_VAD,
            follow_up_window_sec=0.1,
            cooldown_duration_sec=0.05,
        )
        sm.activate()
        sm.voice_detected()
        sm.end_of_speech()
        sm.start_speaking()
        sm.done_speaking()
        assert sm.state == AssistantVoiceState.COOLDOWN

        # Wait for expiry
        time.sleep(0.15)
        result = sm.tick()
        assert result == AssistantVoiceState.ARMED
        assert sm.state == AssistantVoiceState.ARMED

    def test_tick_expires_follow_up_to_idle_ptt(self) -> None:
        """Tick transitions from cooldown to idle when follow-up expires (PTT mode)."""
        sm = AssistantStateMachine(
            activation_mode=ActivationMode.PUSH_TO_TALK,
            follow_up_window_sec=0.1,
            cooldown_duration_sec=0.05,
        )
        sm.transition(AssistantVoiceState.ARMED, "test")  # Skip idle for test
        sm.voice_detected()
        sm.end_of_speech()
        sm.start_speaking()
        sm.done_speaking()

        time.sleep(0.15)
        result = sm.tick()
        assert result == AssistantVoiceState.IDLE


# --- Activation Gate Tests ---


class TestPushToTalkGate:
    """Tests for PushToTalkGate."""

    def test_activated_when_button_pressed(self) -> None:
        """Activates when button is pressed."""
        gate = PushToTalkGate()
        decision = gate.check(text="hello", is_button_pressed=True)
        assert decision.is_activated
        assert decision.confidence == 1.0
        assert decision.remaining_text == "hello"

    def test_not_activated_without_button(self) -> None:
        """Does not activate without button press."""
        gate = PushToTalkGate()
        decision = gate.check(text="hello", is_button_pressed=False)
        assert not decision.is_activated

    def test_gate_name(self) -> None:
        """Gate has correct name."""
        gate = PushToTalkGate()
        assert gate.name == "push_to_talk"


class TestContinuousVADGate:
    """Tests for ContinuousVADGate."""

    def test_activated_with_any_speech(self) -> None:
        """Activates with any speech."""
        gate = ContinuousVADGate()
        decision = gate.check(text="hello there")
        assert decision.is_activated
        assert decision.remaining_text == "hello there"

    def test_not_activated_without_speech(self) -> None:
        """Does not activate without speech."""
        gate = ContinuousVADGate()
        decision = gate.check(text="")
        assert not decision.is_activated

        decision = gate.check(text=None)
        assert not decision.is_activated

    def test_gate_name(self) -> None:
        """Gate has correct name."""
        gate = ContinuousVADGate()
        assert gate.name == "continuous_vad"


class TestWakePhraseGate:
    """Tests for WakePhraseGate."""

    def test_exact_match_activates(self) -> None:
        """Exact wake phrase match activates."""
        gate = WakePhraseGate(phrases=["hey baymax", "baymax"])
        decision = gate.check(text="hey baymax what time is it")
        assert decision.is_activated
        assert decision.trigger_phrase == "hey baymax"
        assert decision.confidence == 1.0
        assert "time" in decision.remaining_text

    def test_case_insensitive(self) -> None:
        """Wake phrase matching is case insensitive."""
        gate = WakePhraseGate(phrases=["hey baymax"])
        decision = gate.check(text="HEY BAYMAX what is the weather")
        assert decision.is_activated

    def test_fuzzy_match(self) -> None:
        """Fuzzy matching handles slight variations."""
        gate = WakePhraseGate(phrases=["hey baymax"], fuzzy_threshold=0.7)
        # "hey bamax" is close to "hey baymax"
        decision = gate.check(text="hey bamax what's up")
        # Should still activate due to fuzzy matching
        assert decision.is_activated or decision.confidence >= 0.7

    def test_no_match_rejects(self) -> None:
        """Non-wake phrases are rejected."""
        gate = WakePhraseGate(phrases=["hey baymax"])
        decision = gate.check(text="what is the weather today")
        assert not decision.is_activated
        assert decision.reason == "no_wake_phrase_detected"

    def test_strips_filler_words(self) -> None:
        """Filler words are stripped before matching."""
        gate = WakePhraseGate(phrases=["hey baymax"])
        decision = gate.check(text="um uh hey baymax hello")
        assert decision.is_activated

    def test_button_override(self) -> None:
        """Button press overrides wake phrase requirement."""
        gate = WakePhraseGate(phrases=["hey baymax"])
        decision = gate.check(text="what is the time", is_button_pressed=True)
        assert decision.is_activated
        assert decision.reason == "button_override"

    def test_multiple_phrases(self) -> None:
        """Multiple wake phrases are supported."""
        gate = WakePhraseGate(phrases=["hey baymax", "hi baymax", "baymax"])

        decision = gate.check(text="hi baymax help me")
        assert decision.is_activated
        assert decision.trigger_phrase == "hi baymax"

        decision = gate.check(text="baymax what is 2+2")
        assert decision.is_activated
        assert decision.trigger_phrase == "baymax"

    def test_gate_name(self) -> None:
        """Gate has correct name."""
        gate = WakePhraseGate()
        assert gate.name == "wake_phrase_gate"

    def test_phrases_property(self) -> None:
        """Can access configured phrases."""
        gate = WakePhraseGate(phrases=["hey baymax", "hi baymax"])
        assert "hey baymax" in gate.phrases
        assert "hi baymax" in gate.phrases


class TestActivationGateFactory:
    """Tests for create_activation_gate factory."""

    def test_creates_push_to_talk(self) -> None:
        """Factory creates PushToTalkGate."""
        gate = create_activation_gate("push_to_talk")
        assert isinstance(gate, PushToTalkGate)

    def test_creates_continuous_vad(self) -> None:
        """Factory creates ContinuousVADGate."""
        gate = create_activation_gate("continuous_vad")
        assert isinstance(gate, ContinuousVADGate)

    def test_creates_wake_phrase_gate(self) -> None:
        """Factory creates WakePhraseGate with phrases."""
        gate = create_activation_gate(
            "wake_phrase_gate",
            wake_phrases=["hey assistant"],
            fuzzy_threshold=0.8,
        )
        assert isinstance(gate, WakePhraseGate)
        assert "hey assistant" in gate.phrases

    def test_invalid_mode_raises(self) -> None:
        """Invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown activation mode"):
            create_activation_gate("invalid_mode")


# --- Full Flow Tests ---


class TestVoiceAssistantFlow:
    """Integration tests for typical voice assistant flows."""

    def test_typical_question_answer_flow(self) -> None:
        """Test typical user question -> assistant answer flow."""
        sm = AssistantStateMachine(activation_mode=ActivationMode.CONTINUOUS_VAD)

        # Start armed
        assert sm.activate()
        assert sm.state == AssistantVoiceState.ARMED

        # User speaks
        assert sm.voice_detected()
        assert sm.state == AssistantVoiceState.LISTENING

        # End of speech, process
        assert sm.end_of_speech()
        assert sm.state == AssistantVoiceState.THINKING

        # Generate response, start TTS
        assert sm.start_speaking()
        assert sm.state == AssistantVoiceState.SPEAKING

        # TTS complete
        assert sm.done_speaking()
        assert sm.state == AssistantVoiceState.COOLDOWN

        # Follow-up window allows continuation
        assert sm.is_follow_up_active

    def test_barge_in_flow(self) -> None:
        """Test user interrupting assistant mid-speech."""
        sm = AssistantStateMachine()
        sm.activate()
        sm.voice_detected()
        sm.end_of_speech()
        sm.start_speaking()

        # User barges in
        assert sm.state == AssistantVoiceState.SPEAKING
        assert sm.interrupt()
        assert sm.state == AssistantVoiceState.INTERRUPTED

        # Process barge-in
        assert sm.handle_interrupt()
        assert sm.state == AssistantVoiceState.LISTENING

    def test_follow_up_conversation(self) -> None:
        """Test multi-turn conversation with follow-ups."""
        sm = AssistantStateMachine(follow_up_window_sec=10.0)

        # First turn
        sm.activate()
        sm.voice_detected()
        sm.end_of_speech()
        sm.start_speaking()
        sm.done_speaking()

        # In cooldown with follow-up active
        assert sm.state == AssistantVoiceState.COOLDOWN
        assert sm.is_follow_up_active

        # User follows up (no re-activation needed)
        assert sm.voice_detected()
        assert sm.state == AssistantVoiceState.LISTENING

    def test_wake_phrase_with_state_machine(self) -> None:
        """Test wake phrase gate integration with state machine."""
        gate = WakePhraseGate(phrases=["hey baymax"])
        sm = AssistantStateMachine(activation_mode=ActivationMode.WAKE_PHRASE_GATE)
        sm.activate()  # Armed, waiting for wake phrase

        # Check transcript
        decision = gate.check("hey baymax what time is it")
        assert decision.is_activated

        # If activated, proceed with voice detection
        if decision.is_activated:
            sm.voice_detected()
            assert sm.state == AssistantVoiceState.LISTENING
