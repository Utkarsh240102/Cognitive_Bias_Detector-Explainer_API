"""
API route definitions.
"""

from fastapi import APIRouter

from app.models.schemas import AnalyzeRequest, AnalyzeResponse
from app.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(request: AnalyzeRequest):
    """Analyze text for cognitive biases."""
    logger.info("Received analysis request (%d chars)", len(request.text))

    # Placeholder — real logic will be wired in Stage 4
    return AnalyzeResponse(
        biases=[],
        explanation="No analysis performed yet.",
        neutral_rewrite=request.text,
    )
