"""Supportive response planner implementation with affect-aware bias."""

import logging

from baymax.core.enums import EmotionLabel, EngagementLevel, ResponseStrategy
from baymax.planner.interfaces import ResponsePlanner
from baymax.schemas.memory import MemoryQueryResult
from baymax.state.models import InteractionState

logger = logging.getLogger(__name__)


class SupportivePlanner(ResponsePlanner):
    """Rule-based planner that chooses supportive response strategies with soft affect bias."""

    _RECALL_KEYWORDS = [
        "remember", "recall", "memory", "memories",
        "last time", "previously", "before",
    ]
    _DIRECT_ASK_KEYWORDS = [
        "how", "what", "why", "can you", "could you", "help me",
        "give me", "tell me", "suggest", "steps", "advice", "tip",
    ]

    def __init__(
        self,
        affect_bias_enabled: bool = True,
        affect_bias_confidence_threshold: float = 0.6,
        negative_valence_threshold: float = -0.3,
        positive_valence_threshold: float = 0.4,
        low_arousal_threshold: float = 0.4,
        high_arousal_threshold: float = 0.55,
    ):
        """Initialize supportive planner.

        Args:
            affect_bias_enabled: Whether to apply affect-based strategy adjustments
            affect_bias_confidence_threshold: Minimum confidence to apply affect bias
            negative_valence_threshold: Valence below this triggers gentler responses
            positive_valence_threshold: Valence above this allows lighter engagement
            low_arousal_threshold: Arousal below this suggests calmer approach
            high_arousal_threshold: Arousal above this suggests more engaging approach
        """
        self.affect_bias_enabled = affect_bias_enabled
        self.affect_bias_confidence_threshold = affect_bias_confidence_threshold
        self.negative_valence_threshold = negative_valence_threshold
        self.positive_valence_threshold = positive_valence_threshold
        self.low_arousal_threshold = low_arousal_threshold
        self.high_arousal_threshold = high_arousal_threshold

    def plan(
        self,
        state: InteractionState,
        memories: MemoryQueryResult,
        context: str = "",
    ) -> ResponseStrategy:
        # Get base strategy using existing logic
        base_strategy = self._get_base_strategy(state, memories, context)

        # Apply affect bias if enabled and confidence is sufficient
        if (self.affect_bias_enabled
            and state.affect_enabled
            and state.affect_confidence >= self.affect_bias_confidence_threshold):

            adjusted_strategy = self._apply_affect_bias(base_strategy, state)

            if adjusted_strategy != base_strategy:
                logger.debug(
                    "Affect bias adjusted strategy: %s -> %s (valence=%.3f, arousal=%.3f, confidence=%.3f)",
                    base_strategy, adjusted_strategy, state.valence, state.arousal, state.affect_confidence
                )

            return adjusted_strategy

        return base_strategy

    def _get_base_strategy(
        self,
        state: InteractionState,
        memories: MemoryQueryResult,
        context: str,
    ) -> ResponseStrategy:
        lower_context = context.lower() if context else ""

        """Get base strategy using original logic."""
        # Detect explicit recall intent from user context
        if context and memories.total_count > 0:
            if any(kw in lower_context for kw in self._RECALL_KEYWORDS):
                return ResponseStrategy.RECALL

        # Explicit user asks should get concrete guidance, not a first-turn greeting.
        has_direct_ask = bool(context) and (
            "?" in context
            or any(kw in lower_context for kw in self._DIRECT_ASK_KEYWORDS)
        )
        if has_direct_ask:
            return ResponseStrategy.SUGGEST

        # First interaction: greet
        if state.turn_count == 0:
            return ResponseStrategy.GREET

        # User seems disengaged
        if state.engagement_level in (EngagementLevel.LOW, EngagementLevel.ABSENT):
            return ResponseStrategy.CHECK_IN

        # Emotional responses
        if state.dominant_emotion == EmotionLabel.SAD:
            return ResponseStrategy.EMPATHIZE
        if state.dominant_emotion == EmotionLabel.ANXIOUS:
            return ResponseStrategy.EMPATHIZE
        if state.dominant_emotion == EmotionLabel.FRUSTRATED:
            return ResponseStrategy.EMPATHIZE
        if state.dominant_emotion == EmotionLabel.TIRED:
            return ResponseStrategy.SUGGEST

        # Has memories to recall
        if memories.total_count > 0:
            return ResponseStrategy.RECALL

        # Default: encourage
        return ResponseStrategy.ENCOURAGE

    def _apply_affect_bias(self, base_strategy: ResponseStrategy, state: InteractionState) -> ResponseStrategy:
        """Apply soft affect bias to adjust strategy tone.

        This applies gentle adjustments based on valence/arousal without overriding safety
        or critical interaction flows (greeting, recall requests, etc.).
        """
        valence = state.valence
        arousal = state.arousal

        # Never override greeting or explicit recall - these are interaction-critical
        if base_strategy in (ResponseStrategy.GREET, ResponseStrategy.RECALL):
            return base_strategy

        # Low valence + low arousal: user appears subdued/thoughtful
        # Bias toward gentler, more validating approach
        if valence <= self.negative_valence_threshold and arousal <= self.low_arousal_threshold:
            if base_strategy == ResponseStrategy.ENCOURAGE:
                # Switch to empathize for gentler tone
                return ResponseStrategy.EMPATHIZE
            elif base_strategy == ResponseStrategy.CHECK_IN:
                # Keep check-in but will be gentler due to affect context in prompt
                return base_strategy
            # Other strategies remain unchanged but tone will be adjusted in prompt builder
            return base_strategy

        # Low valence + higher arousal: user appears tense/concerned
        # Bias toward calming, empathetic approach
        elif valence <= self.negative_valence_threshold and arousal > self.high_arousal_threshold:
            if base_strategy == ResponseStrategy.ENCOURAGE:
                return ResponseStrategy.EMPATHIZE
            # Suggest strategy can help with actionable calming ideas
            elif base_strategy == ResponseStrategy.CHECK_IN:
                return ResponseStrategy.SUGGEST
            return base_strategy

        # High valence + higher arousal: user appears positive and energetic
        # Can support slightly more engaging/upbeat approach
        elif valence >= self.positive_valence_threshold and arousal > self.high_arousal_threshold:
            # These states allow for naturally more engaging responses
            # but don't change strategy - tone adjustment happens in prompt builder
            return base_strategy

        # High valence + low arousal: user appears calm and content
        # Gentle, warm approach works well - no adjustment needed
        elif valence >= self.positive_valence_threshold and arousal <= self.low_arousal_threshold:
            return base_strategy

        # Neutral or mixed affect patterns: stick with base strategy
        else:
            return base_strategy
