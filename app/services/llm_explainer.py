"""
LLM-powered explanation generator.

Uses Groq (llama-3.3-70b-versatile) to produce richer, context-aware
explanations for detected cognitive biases.  If this module fails for
any reason the caller (explainer.py) falls back to template-based
explanations.
"""

from __future__ import annotations

from groq import Groq

from app.config import GROQ_API_KEY, GROQ_MODEL_NAME
from app.logger import get_logger
from app.models.schemas import DetectedBias

logger = get_logger(__name__)

# ── Module-level state ──────────────────────────────────────────
_client = None


def load_llm() -> None:
    """Initialise the Groq client with the API key."""
    global _client
    if _client is not None:
        logger.info("Groq client already initialised — skipping.")
        return

    if not GROQ_API_KEY or GROQ_API_KEY == "your-groq-api-key-here":
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file."
        )

    logger.info("Initialising Groq client (model: %s) …", GROQ_MODEL_NAME)
    _client = Groq(api_key=GROQ_API_KEY)
    logger.info("Groq client ready.")


def _build_prompt(text: str, biases: list[DetectedBias]) -> str:
    """Construct a concise instruction prompt for the LLM."""
    bias_list = ", ".join(
        f"{b.type} ({b.confidence:.0%})" for b in biases
    )
    return (
        f"You are a cognitive bias expert. Explain the cognitive biases "
        f"found in this statement in 2-4 sentences. Be specific about "
        f"WHY each bias applies to THIS statement.\n\n"
        f"Statement: \"{text}\"\n"
        f"Detected biases: {bias_list}\n\n"
        f"Provide a clear, concise explanation. Do not use bullet points, "
        f"lists, or any markdown formatting (no bold, no italics) — write "
        f"in plain paragraph form using plain text only."
    )


def generate_llm_explanation(text: str, biases: list[DetectedBias]) -> str:
    """Generate a Groq-powered explanation.

    Raises:
        RuntimeError: If the Groq client has not been initialised.
    """
    if _client is None:
        raise RuntimeError("Groq client is not initialised. Call load_llm() first.")

    prompt = _build_prompt(text, biases)
    logger.debug("Groq prompt (%d chars): %s", len(prompt), prompt[:120])

    response = _client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    output = response.choices[0].message.content.strip()

    logger.info("Groq explanation generated (%d chars): %s", len(output), output)
    return output
