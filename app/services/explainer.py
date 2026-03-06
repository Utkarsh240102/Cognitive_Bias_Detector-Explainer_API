"""
Explanation generation service.

Stage 5 starts here: this module will first provide template-based
explanations and later add optional LLM generation with template fallback.
"""

from app.config import LLM_ENABLED
from app.logger import get_logger
from app.models.schemas import DetectedBias
from app.services.llm_explainer import generate_llm_explanation

logger = get_logger(__name__)

# ── Explanation Templates ───────────────────────────────────────
TEMPLATES: dict[str, str] = {
    "Confirmation Bias": (
        "This statement reflects confirmation bias — it favors information "
        "that supports an existing belief while ignoring evidence to the contrary."
    ),
    "Overgeneralization": (
        "This statement reflects overgeneralization — it draws a broad "
        "conclusion from limited evidence and applies it too widely."
    ),
    "Stereotyping": (
        "This statement reflects stereotyping — it assigns traits or "
        "abilities to a group rather than judging individuals based on evidence."
    ),
    "Emotional Reasoning": (
        "This statement reflects emotional reasoning — it treats feelings "
        "as proof of something being true, rather than relying on objective evidence."
    ),
    "False Dilemma": (
        "This statement reflects a false dilemma — it presents the situation "
        "as if there are only two possible choices, ignoring other alternatives."
    ),
    "Catastrophizing": (
        "This statement reflects catastrophizing — it assumes the worst "
        "possible outcome and exaggerates the severity of the situation."
    ),
    "Hasty Generalization": (
        "This statement reflects hasty generalization — it jumps to a "
        "broad conclusion based on too few examples or limited experience."
    ),
    "Black-and-White Thinking": (
        "This statement reflects black-and-white thinking — it sees the "
        "situation in extreme terms with no middle ground or nuance."
    ),
}


_NO_BIAS_MESSAGE = (
    "No strong cognitive bias was detected above the current confidence threshold."
)


def generate_explanation(text: str, biases: list[DetectedBias]) -> str:
    """Generate a human-readable explanation for detected biases.

    Args:
        text: Original user input.
        biases: Biases detected by the classification pipeline.

    Returns:
        Explanation text for the response payload.
    """
    logger.info("Generating explanation for %d detected biases", len(biases))

    if not biases:
        return _NO_BIAS_MESSAGE

    # Try LLM first, fall back to templates on any failure
    if LLM_ENABLED:
        try:
            llm_result = generate_llm_explanation(text, biases)
            if llm_result:
                logger.info("Using LLM-generated explanation")
                return llm_result
        except Exception:
            logger.warning("LLM explanation failed — falling back to templates", exc_info=True)

    return _generate_template_explanation(biases)


def _generate_template_explanation(biases: list[DetectedBias]) -> str:
    """Build explanation text from templates for each detected bias."""
    if not biases:
        return _NO_BIAS_MESSAGE

    parts = []
    for bias in biases:
        template = TEMPLATES.get(bias.type)
        if template:
            parts.append(template)
        else:
            parts.append(f"A cognitive bias ({bias.type}) was detected.")

    return " ".join(parts)