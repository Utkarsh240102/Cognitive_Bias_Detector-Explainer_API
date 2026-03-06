"""
LLM-powered explanation generator.

Uses a local Flan-T5 model to produce richer, context-aware explanations
for detected cognitive biases.  If this module fails for any reason the
caller (explainer.py) falls back to template-based explanations.
"""

from __future__ import annotations

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from app.config import LLM_MODEL_NAME, LLM_MAX_LENGTH, MODEL_CACHE_DIR
from app.logger import get_logger
from app.models.schemas import DetectedBias

logger = get_logger(__name__)

# ── Module-level state ──────────────────────────────────────────
_tokenizer = None
_model = None


def load_llm() -> None:
    """Load the Flan-T5 tokenizer and model into memory."""
    global _tokenizer, _model
    if _model is not None:
        logger.info("LLM already loaded — skipping.")
        return

    cache = str(MODEL_CACHE_DIR)
    logger.info("Loading LLM '%s' …", LLM_MODEL_NAME)
    _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, cache_dir=cache)
    _model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME, cache_dir=cache)
    logger.info("LLM loaded successfully.")


def _build_prompt(text: str, biases: list[DetectedBias]) -> str:
    """Construct a concise instruction prompt for Flan-T5."""
    bias_list = ", ".join(
        f"{b.type} ({b.confidence:.0%})" for b in biases
    )
    return (
        f"Explain the cognitive biases found in this statement.\n\n"
        f"Statement: \"{text}\"\n"
        f"Detected biases: {bias_list}\n\n"
        f"Provide a clear, concise explanation of why each bias applies."
    )


def generate_llm_explanation(text: str, biases: list[DetectedBias]) -> str:
    """Generate an LLM-powered explanation.

    Raises:
        RuntimeError: If the LLM model has not been loaded yet.
    """
    if _model is None or _tokenizer is None:
        raise RuntimeError("LLM is not loaded. Call load_llm() first.")

    prompt = _build_prompt(text, biases)
    logger.debug("LLM prompt (%d chars): %s", len(prompt), prompt[:120])

    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = _model.generate(**inputs, max_new_tokens=LLM_MAX_LENGTH, do_sample=False)
    output = _tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    logger.info("LLM explanation generated (%d chars)", len(output))
    return output
