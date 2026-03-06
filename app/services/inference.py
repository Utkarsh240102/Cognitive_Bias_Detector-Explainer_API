"""
Model inference — loads BART zero-shot model and classifies text against bias labels.
"""

import torch
from transformers import pipeline

from app.config import MODEL_NAME, DEVICE, BIAS_LABELS
from app.logger import get_logger

logger = get_logger(__name__)

# ── Global model reference ──────────────────────────────────────
_classifier = None


def load_model() -> None:
    """Load the zero-shot classification model into memory."""
    global _classifier

    device = 0 if DEVICE == "cuda" and torch.cuda.is_available() else -1
    device_name = "CUDA (GPU)" if device == 0 else "CPU"
    logger.info("Loading model '%s' on %s...", MODEL_NAME, device_name)

    _classifier = pipeline(
        "zero-shot-classification",
        model=MODEL_NAME,
        device=device,
    )

    logger.info("Model loaded successfully.")


def classify(text: str) -> dict[str, float]:
    """Run zero-shot classification and return scores per bias label.

    Args:
        text: Cleaned text to classify.

    Returns:
        Dictionary mapping bias label → confidence score (0.0–1.0).

    Raises:
        RuntimeError: If the model has not been loaded yet.
    """
    if _classifier is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    result = _classifier(text, candidate_labels=BIAS_LABELS, multi_label=True)

    scores = dict(zip(result["labels"], result["scores"]))
    logger.info("Classification complete: %d labels scored", len(scores))
    return scores
