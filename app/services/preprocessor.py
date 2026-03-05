"""
Text preprocessing — cleans and validates user input before model inference.
"""

import re
import unicodedata

from app.config import MIN_TEXT_LENGTH, MAX_TEXT_LENGTH
from app.logger import get_logger

logger = get_logger(__name__)


def preprocess(text: str) -> str:
    """Clean and validate raw user input.

    Steps:
        1. Normalize unicode characters
        2. Strip leading/trailing whitespace
        3. Collapse multiple spaces into one

    Raises:
        ValueError: If the cleaned text is too short or too long.
    """
    cleaned = _clean_text(text)
    _validate_length(cleaned)
    logger.info("Preprocessed text: %d → %d chars", len(text), len(cleaned))
    return cleaned


def _clean_text(text: str) -> str:
    """Normalize unicode, strip whitespace, and collapse extra spaces."""
    # Normalize unicode (e.g., accented chars → standard form)
    text = unicodedata.normalize("NFKC", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    # Collapse multiple spaces/tabs/newlines into a single space
    text = re.sub(r"\s+", " ", text)
    return text


def _validate_length(text: str) -> None:
    """Raise ValueError if cleaned text is outside allowed length bounds."""
    if len(text) < MIN_TEXT_LENGTH:
        raise ValueError(
            f"Text too short after cleaning ({len(text)} chars). "
            f"Minimum is {MIN_TEXT_LENGTH}."
        )
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(
            f"Text too long after cleaning ({len(text)} chars). "
            f"Maximum is {MAX_TEXT_LENGTH}."
        )
