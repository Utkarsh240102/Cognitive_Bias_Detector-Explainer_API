"""
API route definitions.
"""

from fastapi import APIRouter

from app.models.schemas import AnalyzeRequest, AnalyzeResponse, DetectedBias
from app.logger import get_logger
from app.services.preprocessor import preprocess
from app.services.inference import classify
from app.services.bias_selector import select_biases
from app.services.explainer import generate_explanation
from app.services.rewriter import generate_rewrite

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

    # Step 4: Generate explanation
    explanation = generate_explanation(cleaned, biases)

    # Step 5: Generate neutral rewrite
    neutral_rewrite = generate_rewrite(cleaned, biases)

    return AnalyzeResponse(
        biases=biases,
        explanation=explanation,
        neutral_rewrite=neutral_rewrite,
    )
