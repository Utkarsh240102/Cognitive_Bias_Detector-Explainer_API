"""
LLM-powered explanation generator.

Uses Google Gemini Flash to produce richer, context-aware explanations
for detected cognitive biases.  If this module fails for any reason the
caller (explainer.py) falls back to template-based explanations.
"""

from __future__ import annotations

from google import genai

from app.config import GEMINI_API_KEY, GEMINI_MODEL_NAME
from app.logger import get_logger
from app.models.schemas import DetectedBias

logger = get_logger(__name__)

# ── Module-level state ──────────────────────────────────────────
_client = None


def load_llm() -> None:
    """Initialise the Gemini client with the API key."""
    global _client
    if _client is not None:
        logger.info("Gemini client already initialised — skipping.")
        return

    if not GEMINI_API_KEY or GEMINI_API_KEY == "your-gemini-api-key-here":
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your .env file."
        )

    logger.info("Initialising Gemini client (model: %s) …", GEMINI_MODEL_NAME)
    _client = genai.Client(api_key=GEMINI_API_KEY)
    logger.info("Gemini client ready.")


def _build_prompt(text: str, biases: list[DetectedBias]) -> str:
    """Construct a concise instruction prompt for Gemini."""
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
    """Generate a Gemini-powered explanation.

    Raises:
        RuntimeError: If the Gemini client has not been initialised.
    """
    if _client is None:
        raise RuntimeError("Gemini client is not initialised. Call load_llm() first.")

    prompt = _build_prompt(text, biases)
    logger.debug("Gemini prompt (%d chars): %s", len(prompt), prompt[:120])

    response = _client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=prompt,
    )
    output = response.text.strip()

    logger.info("Gemini explanation generated (%d chars): %s", len(output), output)
    return output
