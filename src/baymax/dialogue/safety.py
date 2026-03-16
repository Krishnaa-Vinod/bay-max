"""Safety gating for dialogue generation.

Detects diagnosis-style or unsafe medical requests and returns safe redirects.
Bay-Max is a supportive companion, not a clinician.
"""

from baymax.schemas.response import SafetyDecision

# Phrases that suggest the user is asking for a medical diagnosis or treatment
_DIAGNOSIS_PATTERNS: list[str] = [
    "diagnose me",
    "do i have",
    "what disease",
    "what condition",
    "am i sick",
    "tell me if i have",
    "can you tell if",
    "detect disease",
    "detect illness",
    "medical diagnosis",
    "clinical assessment",
    "clinical diagnosis",
    "what is wrong with me",
    "what's wrong with me",
    "whats wrong with me",
    "prognosis",
    "do i have cancer",
    "do i have diabetes",
    "do i have depression",
    "i have symptoms of",
    "diagnose from my face",
    "diagnose from face",
    "what medication",
    "prescribe",
    "should i take",
    "what drug",
    "medical advice",
    "doctor advice",
]

# Phrases suggesting the user wants unsafe certainty about health status
_MEDICAL_CERTAINTY_PATTERNS: list[str] = [
    "are you sure i am",
    "confirm i have",
    "100% certain",
    "tell me for sure",
    "definitely sick",
]

_REDIRECT_MESSAGE = (
    "I'm here to support you, but I'm not able to make medical diagnoses or "
    "interpret health symptoms. "
    "If you're concerned about your health, please speak with a qualified healthcare professional. "
    "I'm happy to listen and offer companionship."
)


def check_safety(text: str) -> SafetyDecision:
    """Check whether the user's message raises safety concerns.

    Args:
        text: The user-facing text or LLM output to check.

    Returns:
        SafetyDecision with is_safe=False and a redirect when a concern is found.
    """
    lower = text.lower()
    flags: list[str] = []

    for pattern in _DIAGNOSIS_PATTERNS:
        if pattern in lower:
            flags.append(f"diagnosis_request:{pattern}")
            break  # One match is enough to flag

    for pattern in _MEDICAL_CERTAINTY_PATTERNS:
        if pattern in lower:
            flags.append(f"medical_certainty:{pattern}")
            break

    if flags:
        return SafetyDecision(
            is_safe=False,
            flags=flags,
            redirect_response=_REDIRECT_MESSAGE,
        )

    return SafetyDecision(is_safe=True)


def check_output_safety(generated_text: str) -> SafetyDecision:
    """Check an LLM-generated response for unsafe medical claims.

    Args:
        generated_text: The LLM output to validate.

    Returns:
        SafetyDecision — if not safe, the response should be replaced.
    """
    lower = generated_text.lower()
    flags: list[str] = []

    dangerous_output_patterns = [
        "you have",
        "you are diagnosed",
        "you suffer from",
        "i diagnose",
        "my diagnosis is",
        "your condition is",
        "you should take",
        "i recommend medication",
        "take this drug",
    ]
    for pattern in dangerous_output_patterns:
        if pattern in lower:
            flags.append(f"unsafe_output:{pattern}")

    if flags:
        return SafetyDecision(
            is_safe=False,
            flags=flags,
            redirect_response=_REDIRECT_MESSAGE,
        )

    return SafetyDecision(is_safe=True)
