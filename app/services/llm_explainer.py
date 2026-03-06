"""
LLM-powered explanation generator.

Uses a local Flan-T5 model to produce richer, context-aware explanations
for detected cognitive biases.  If this module fails for any reason the
caller (explainer.py) falls back to template-based explanations.
"""

from __future__ import annotations

from transformers import pipeline as hf_pipeline

from app.config import LLM_MODEL_NAME, LLM_MAX_LENGTH, MODEL_CACHE_DIR
from app.logger import get_logger
from app.models.schemas import DetectedBias

logger = get_logger(__name__)

# ── Module-level state ──────────────────────────────────────────
_llm_pipeline = None


def load_llm() -> None:
    """Load the text2text-generation pipeline (Flan-T5) into memory."""
    global _llm_pipeline
    if _llm_pipeline is not None:
        logger.info("LLM already loaded — skipping.")
        return

    logger.info("Loading LLM '%s' …", LLM_MODEL_NAME)
    _llm_pipeline = hf_pipeline(
        "text2text-generation",
        model=LLM_MODEL_NAME,
        model_kwargs={"cache_dir": str(MODEL_CACHE_DIR)},
    )
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
        RuntimeError: If the LLM pipeline has not been loaded yet.
    """
    if _llm_pipeline is None:
        raise RuntimeError("LLM pipeline is not loaded. Call load_llm() first.")

    prompt = _build_prompt(text, biases)
    logger.debug("LLM prompt (%d chars): %s", len(prompt), prompt[:120])

    result = _llm_pipeline(prompt, max_length=LLM_MAX_LENGTH, do_sample=False)
    output = result[0]["generated_text"].strip()

    logger.info("LLM explanation generated (%d chars)", len(output))
    return output
