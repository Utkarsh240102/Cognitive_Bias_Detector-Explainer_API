"""
Model inference — supports both fine-tuned RoBERTa and zero-shot BART.

When USE_FINETUNED_MODEL is True:
    Loads a locally trained RoBERTa multi-label classifier from trained_model/.

When USE_FINETUNED_MODEL is False:
    Falls back to the zero-shot BART-large-mnli pipeline.
"""

import torch
from transformers import pipeline, RobertaForSequenceClassification, RobertaTokenizerFast

from app.config import (
    USE_FINETUNED_MODEL,
    FINETUNED_MODEL_DIR,
    MODEL_NAME,
    DEVICE,
    BIAS_LABELS,
    MODEL_CACHE_DIR,
)
from app.logger import get_logger

logger = get_logger(__name__)

# ── Global model references ─────────────────────────────────────
_classifier = None        # Zero-shot pipeline (BART)
_ft_model = None          # Fine-tuned model (RoBERTa)
_ft_tokenizer = None      # Fine-tuned tokenizer
_device = None            # torch device


def load_model() -> None:
    """Load the classification model into memory."""
    global _classifier, _ft_model, _ft_tokenizer, _device

    _device = "cuda" if DEVICE == "cuda" and torch.cuda.is_available() else "cpu"
    device_name = "CUDA (GPU)" if _device == "cuda" else "CPU"

    if USE_FINETUNED_MODEL:
        logger.info(
            "Loading fine-tuned RoBERTa from '%s' on %s...",
            FINETUNED_MODEL_DIR,
            device_name,
        )
        _ft_model = RobertaForSequenceClassification.from_pretrained(
            str(FINETUNED_MODEL_DIR)
        ).to(_device)
        _ft_model.eval()
        _ft_tokenizer = RobertaTokenizerFast.from_pretrained(str(FINETUNED_MODEL_DIR))
        logger.info("Fine-tuned RoBERTa loaded successfully.")
    else:
        device_id = 0 if _device == "cuda" else -1
        logger.info("Loading zero-shot model '%s' on %s...", MODEL_NAME, device_name)
        logger.info("Model cache directory: %s", MODEL_CACHE_DIR)

        _classifier = pipeline(
            "zero-shot-classification",
            model=MODEL_NAME,
            device=device_id,
            model_kwargs={"cache_dir": str(MODEL_CACHE_DIR)},
        )
        logger.info("Zero-shot model loaded successfully.")


def classify(text: str) -> dict[str, float]:
    """Classify text and return scores per bias label.

    Args:
        text: Cleaned text to classify.

    Returns:
        Dictionary mapping bias label → confidence score (0.0–1.0).

    Raises:
        RuntimeError: If the model has not been loaded yet.
    """
    if USE_FINETUNED_MODEL:
        return _classify_finetuned(text)
    else:
        return _classify_zero_shot(text)


def _classify_finetuned(text: str) -> dict[str, float]:
    """Run inference with the fine-tuned RoBERTa model."""
    if _ft_model is None or _ft_tokenizer is None:
        raise RuntimeError("Fine-tuned model not loaded. Call load_model() first.")

    inputs = _ft_tokenizer(
        text, return_tensors="pt", truncation=True, max_length=128
    ).to(_device)

    with torch.no_grad():
        outputs = _ft_model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze().cpu().tolist()

    # Handle edge case where probs is a single float
    if isinstance(probs, float):
        probs = [probs]

    scores = dict(zip(BIAS_LABELS, probs))
    logger.info("Classification complete (fine-tuned): %d labels scored", len(scores))
    return scores


def _classify_zero_shot(text: str) -> dict[str, float]:
    """Run inference with the zero-shot BART pipeline."""
    if _classifier is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    result = _classifier(text, candidate_labels=BIAS_LABELS, multi_label=True)

    scores = dict(zip(result["labels"], result["scores"]))
    logger.info("Classification complete (zero-shot): %d labels scored", len(scores))
    return scores
