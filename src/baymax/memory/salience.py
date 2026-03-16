"""Salience scoring utility for memory prioritization."""

from baymax.core.enums import EmotionLabel, EngagementLevel


def compute_salience(
    engagement: EngagementLevel = EngagementLevel.MEDIUM,
    emotion: EmotionLabel = EmotionLabel.NEUTRAL,
    is_first_interaction: bool = False,
    user_expressed_need: bool = False,
) -> float:
    """Compute a salience score (0.0 to 1.0) for an observation or memory.

    Higher salience means the memory is more important to retain and surface.
    """
    score = 0.3  # base

    engagement_weights = {
        EngagementLevel.HIGH: 0.2,
        EngagementLevel.MEDIUM: 0.1,
        EngagementLevel.LOW: 0.05,
        EngagementLevel.ABSENT: 0.0,
    }
    score += engagement_weights.get(engagement, 0.1)

    emotional_weights = {
        EmotionLabel.SAD: 0.25,
        EmotionLabel.ANXIOUS: 0.25,
        EmotionLabel.FRUSTRATED: 0.2,
        EmotionLabel.TIRED: 0.15,
        EmotionLabel.HAPPY: 0.1,
        EmotionLabel.NEUTRAL: 0.05,
        EmotionLabel.UNKNOWN: 0.0,
    }
    score += emotional_weights.get(emotion, 0.05)

    if is_first_interaction:
        score += 0.15

    if user_expressed_need:
        score += 0.2

    return min(score, 1.0)
