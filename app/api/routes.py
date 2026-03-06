"""
API route definitions.
"""

from fastapi import APIRouter

from app.models.schemas import AnalyzeRequest, AnalyzeResponse, DetectedBias
from app.logger import get_logger
from app.services.preprocessor import preprocess
from app.services.inference import classify
from app.services.bias_selector import select_biases

logger = get_logger(__name__)

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(request: AnalyzeRequest):
    """Analyze text for cognitive biases."""
    logger.info("Received analysis request (%d chars)", len(request.text))

    # Step 1: Preprocess
    cleaned = preprocess(request.text)

    # Step 2: Run model inference
    raw_scores = classify(cleaned)

    # Step 3: Filter biases above threshold
    detected = select_biases(raw_scores)
    biases = [DetectedBias(**b) for b in detected]

    # Placeholder explanation/rewrite — will be replaced in Stages 5 & 6
    explanation = "Analysis complete." if not biases else (
        "Detected biases: "
        + ", ".join(f"{b.type} ({b.confidence:.0%})" for b in biases)
        + "."
    )
    neutral_rewrite = request.text

    return AnalyzeResponse(
        biases=biases,
        explanation=explanation,
        neutral_rewrite=neutral_rewrite,
    )
