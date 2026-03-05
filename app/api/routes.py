"""
API route definitions.
"""

from fastapi import APIRouter

from app.models.schemas import AnalyzeRequest, AnalyzeResponse, DetectedBias
from app.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(request: AnalyzeRequest):
    """Analyze text for cognitive biases."""
    logger.info("Received analysis request (%d chars)", len(request.text))

    # TODO: Replace with real model predictions in Stage 4
    dummy_biases = [
        DetectedBias(type="Stereotyping", confidence=0.92),
        DetectedBias(type="Overgeneralization", confidence=0.74),
        DetectedBias(type="Hasty Generalization", confidence=0.61),
    ]

    dummy_explanation = (
        "The statement assigns traits to a group based on group membership "
        "rather than individual evidence (Stereotyping). It also draws broad "
        "conclusions from limited evidence (Overgeneralization, Hasty Generalization)."
    )

    dummy_rewrite = (
        "While some individuals in this group may exhibit these traits, "
        "it is inaccurate to generalize this to all members."
    )

    return AnalyzeResponse(
        biases=dummy_biases,
        explanation=dummy_explanation,
        neutral_rewrite=dummy_rewrite,
    )
