"""
Neutral rewrite generator.

Uses the already-loaded Groq client to rephrase biased statements
into more balanced, neutral language while preserving the core message.
Falls back to returning the original text unchanged if Groq is
unavailable or raises an error.
"""

from __future__ import annotations

from app.config import GROQ_MODEL_NAME, LLM_ENABLED
from app.logger import get_logger
from app.models.schemas import DetectedBias
from app.services import llm_explainer

logger = get_logger(__name__)


def _build_rewrite_prompt(text: str, biases: list[DetectedBias]) -> str:
    """Construct the rewrite prompt for the LLM."""
    bias_list = ", ".join(b.type for b in biases)
    return (
        f"You are an expert editor specialising in neutral, unbiased language. "
        f"Rewrite the following statement to remove cognitive bias while "
        f"preserving the speaker's core message. The statement contains these "
        f"biases: {bias_list}.\n\n"
        f"Original statement: \"{text}\"\n\n"
        f"Rules:\n"
        f"- Keep the rewrite to 1-3 sentences.\n"
        f"- Do not add new claims or facts.\n"
        f"- Do not lecture or moralize — just rephrase neutrally.\n"
        f"- Do not use markdown formatting, bullet points, bold, or italics.\n"
        f"- Write in plain text only.\n"
        f"- Output only the rewritten statement, nothing else"
    )



def generate_rewrite(text: str, biases: list[DetectedBias]) -> str:
    """Return a neutral rewrite of *text*.

    If no biases were detected, or if Groq is disabled / unavailable,
    the original text is returned unchanged.
    """
    if not biases:
        logger.info("No biases detected — skipping rewrite, returning original.")
        return text

    if not LLM_ENABLED:
        logger.info("LLM disabled — skipping rewrite, returning original.")
        return text

    client = llm_explainer._client
    if client is None:
        logger.warning("Groq client not initialised — returning original text.")
        return text

    prompt = _build_rewrite_prompt(text, biases)

    try:
        logger.info("Requesting neutral rewrite from Groq …")
        response = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
        )
        rewrite = response.choices[0].message.content.strip()
        logger.info("Groq rewrite generated (%d chars): %s", len(rewrite), rewrite)
        return rewrite
    except Exception as exc:
        logger.warning("Groq rewrite failed (%s) — returning original text.", exc)
        return text
