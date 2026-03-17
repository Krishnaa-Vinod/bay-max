"""Clinical language safety enforcement for affect memory consolidation."""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Clinical/diagnostic terms that should NOT appear in memory summaries
CLINICAL_BLOCKLIST = [
    # Direct diagnostic terms
    "diagnosis", "diagnosed", "diagnosing", "diagnostic",
    "disorder", "syndrome", "condition", "illness", "disease",
    "symptom", "symptoms", "clinical", "clinically",
    "pathology", "pathological",

    # Mental health diagnostic terms
    "depression", "depressed", "bipolar", "manic", "mania",
    "anxiety disorder", "panic disorder", "ptsd", "ocd",
    "schizophrenia", "psychotic", "psychosis",
    "adhd", "add", "autism", "autistic",
    "borderline", "antisocial", "narcissistic",

    # Medical certainty claims
    "suffers from", "suffering from", "has been diagnosed",
    "shows signs of", "exhibits symptoms",
    "medical condition", "psychiatric condition",
    "mental illness", "psychological disorder",

    # Treatment/prescription terms
    "needs medication", "requires treatment",
    "should see a doctor", "medical intervention",
    "therapy required", "hospitalization",

    # Physiological diagnostic claims
    "blood pressure", "heart rate abnormal",
    "neurological", "cardiovascular",
    "metabolic disorder", "hormonal imbalance"
]

# Safe alternative phrasing patterns
SAFE_ALTERNATIVES = {
    "depressed": "appeared subdued",
    "depression": "low spirits",
    "anxious": "appeared concerned",
    "anxiety": "worry",
    "manic": "very energetic",
    "bipolar": "variable moods",
    "stressed": "appeared tense",
    "tired": "appeared low-energy",
    "exhausted": "very low-energy",
    "hyperactive": "very active",
    "withdrawn": "quiet and reserved",
    "agitated": "restless",
    "moody": "variable affect"
}

def check_clinical_language(text: str) -> tuple[bool, list[str]]:
    """Check if text contains clinical/diagnostic language.

    Args:
        text: Text to check

    Returns:
        Tuple of (has_clinical_terms, list_of_found_terms)
    """
    found_terms = []
    text_lower = text.lower()

    for term in CLINICAL_BLOCKLIST:
        if term in text_lower:
            found_terms.append(term)

    return len(found_terms) > 0, found_terms


def sanitize_clinical_language(text: str) -> str:
    """Replace clinical language with safer alternatives.

    Args:
        text: Input text that may contain clinical terms

    Returns:
        Sanitized text with clinical terms replaced
    """
    sanitized = text

    # Apply replacements (case-insensitive)
    for clinical_term, safe_alternative in SAFE_ALTERNATIVES.items():
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(clinical_term) + r'\b'
        sanitized = re.sub(pattern, safe_alternative, sanitized, flags=re.IGNORECASE)

    # Remove or flag any remaining clinical terms
    has_clinical, found_terms = check_clinical_language(sanitized)
    if has_clinical:
        logger.warning(
            "Remaining clinical terms found after sanitization: %s in text: %s",
            found_terms, sanitized[:100]
        )
        # For any remaining terms, replace with generic alternatives
        for term in found_terms:
            if term in sanitized.lower():
                sanitized = re.sub(
                    r'\b' + re.escape(term) + r'\b',
                    "unclear patterns",
                    sanitized,
                    flags=re.IGNORECASE
                )

    return sanitized


def is_affect_summary_safe(summary: dict[str, Any]) -> tuple[bool, list[str]]:
    """Check if an affect summary is safe for memory consolidation.

    Args:
        summary: Affect summary dict from AffectSmoother.get_affect_summary()

    Returns:
        Tuple of (is_safe, list_of_issues)
    """
    issues = []

    # Check main content
    if "content" in summary:
        has_clinical, found_terms = check_clinical_language(summary["content"])
        if has_clinical:
            issues.append(f"Clinical terms in content: {found_terms}")

    # Check valence/arousal summaries
    for key in ["valence_summary", "arousal_summary"]:
        if key in summary:
            has_clinical, found_terms = check_clinical_language(summary[key])
            if has_clinical:
                issues.append(f"Clinical terms in {key}: {found_terms}")

    # Check confidence level
    if summary.get("confidence", 0.0) < 0.5:
        issues.append("Confidence too low for memory storage")

    return len(issues) == 0, issues


def create_safe_affect_memory_content(
    valence_summary: str,
    arousal_summary: str,
    confidence: float,
    duration_minutes: float
) -> str:
    """Create safe, non-clinical affect memory content.

    Args:
        valence_summary: Summary of valence patterns
        arousal_summary: Summary of arousal patterns
        confidence: Confidence in the observations
        duration_minutes: Duration of the observation period

    Returns:
        Safe memory content string
    """
    # Sanitize inputs
    safe_valence = sanitize_clinical_language(valence_summary)
    safe_arousal = sanitize_clinical_language(arousal_summary)

    # Create observational (not diagnostic) phrasing
    if confidence >= 0.8:
        certainty = "consistently"
    elif confidence >= 0.6:
        certainty = "generally"
    else:
        certainty = "occasionally"

    if safe_valence == "generally positive" and "energetic" in safe_arousal:
        content = f"User {certainty} appeared upbeat and engaged during this session"
    elif safe_valence == "generally positive" and "calm" in safe_arousal:
        content = f"User {certainty} appeared content and peaceful during this session"
    elif safe_valence == "generally subdued" and "calm" in safe_arousal:
        content = f"User {certainty} appeared quiet and thoughtful during this session"
    elif safe_valence == "generally subdued" and "energetic" in safe_arousal:
        content = f"User {certainty} appeared tense or restless during this session"
    elif safe_valence == "neutral":
        content = f"User maintained a calm, steady presence with {safe_arousal}"
    else:
        content = f"User displayed {safe_valence} affect with {safe_arousal}"

    # Add duration context
    if duration_minutes >= 15:
        content += f" over {duration_minutes:.1f} minutes"

    return content


def validate_memory_content(content: str) -> tuple[bool, str]:
    """Validate memory content for clinical safety.

    Args:
        content: Memory content to validate

    Returns:
        Tuple of (is_valid, error_or_sanitized_content)
    """
    has_clinical, found_terms = check_clinical_language(content)

    if has_clinical:
        logger.info("Sanitizing clinical language in memory content: %s", found_terms)
        sanitized = sanitize_clinical_language(content)

        # Double-check after sanitization
        still_has_clinical, remaining_terms = check_clinical_language(sanitized)
        if still_has_clinical:
            return False, f"Unable to sanitize clinical terms: {remaining_terms}"

        return True, sanitized

    return True, content