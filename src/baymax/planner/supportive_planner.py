"""Supportive response planner implementation."""

from baymax.core.enums import EmotionLabel, EngagementLevel, ResponseStrategy
from baymax.planner.interfaces import ResponsePlanner
from baymax.schemas.memory import MemoryQueryResult
from baymax.state.models import InteractionState


class SupportivePlanner(ResponsePlanner):
    """Rule-based planner that chooses supportive response strategies."""

    def plan(
        self, state: InteractionState, memories: MemoryQueryResult
    ) -> ResponseStrategy:
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
