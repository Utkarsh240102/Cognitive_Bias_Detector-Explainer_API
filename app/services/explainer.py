"""
Explanation generation service.

Stage 5 starts here: this module will first provide template-based
explanations and later add optional LLM generation with template fallback.
"""

from app.logger import get_logger
from app.models.schemas import DetectedBias

logger = get_logger(__name__)


def generate_explanation(text: str, biases: list[DetectedBias]) -> str:
    """Generate a human-readable explanation for detected biases.

    Args:
        text: Original user input.
        biases: Biases detected by the classification pipeline.

    Returns:
        Explanation text for the response payload.
    """
    logger.info("Generating explanation for %d detected biases", len(biases))
    raise NotImplementedError("Template explanations will be added in Step 5.2/5.3.")