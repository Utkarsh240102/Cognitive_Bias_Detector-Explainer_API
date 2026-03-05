"""
Pydantic schemas for API request and response validation.
"""

from pydantic import BaseModel, Field


# ── Request ─────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    """Incoming request body for bias analysis."""

    text: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="The text to analyze for cognitive biases.",
        examples=["Everyone from that community is dishonest."],
    )


# ── Response Components ─────────────────────────────────────────
class DetectedBias(BaseModel):
    """A single detected bias with its confidence score."""

    type: str = Field(..., description="Name of the cognitive bias.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score (0-1)."
    )


class AnalyzeResponse(BaseModel):
    """Full analysis response returned by the API."""

    biases: list[DetectedBias] = Field(
        default_factory=list,
        description="List of detected cognitive biases.",
    )
    explanation: str = Field(
        ..., description="Human-readable explanation of detected biases."
    )
    neutral_rewrite: str = Field(
        ..., description="A neutrally rewritten version of the input text."
    )
