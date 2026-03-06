"""
Bias selector — filters raw model scores using the confidence threshold.
"""

from app.config import CONFIDENCE_THRESHOLD
from app.logger import get_logger

logger = get_logger(__name__)


def select_biases(scores: dict[str, float]) -> list[dict[str, float]]:
    """Filter scores that exceed the confidence threshold.

    Args:
        scores: Dictionary mapping bias label → confidence score (0.0–1.0).

    Returns:
        List of dicts with 'type' and 'confidence', sorted by confidence descending.
    """
    detected = [
        {"type": label, "confidence": round(score, 4)}
        for label, score in scores.items()
        if score >= CONFIDENCE_THRESHOLD
    ]

    detected.sort(key=lambda x: x["confidence"], reverse=True)

    logger.info(
        "Bias selection: %d/%d labels above threshold (%.2f)",
        len(detected),
        len(scores),
        CONFIDENCE_THRESHOLD,
    )
    return detected
