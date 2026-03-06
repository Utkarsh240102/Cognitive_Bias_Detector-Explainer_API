"""
Central configuration for the Cognitive Bias Detector API.
"""

from pathlib import Path

# ── Project Root ─────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# ── Model Settings ──────────────────────────────────────────────
MODEL_NAME = "facebook/bart-large-mnli"
DEVICE = "cuda"  # Will fallback to CPU in inference.py if CUDA unavailable
MODEL_CACHE_DIR = BASE_DIR / "model_cache"  # Local cache — avoids global downloads

# ── Bias Labels ─────────────────────────────────────────────────
BIAS_LABELS: list[str] = [
    "Confirmation Bias",
    "Overgeneralization",
    "Stereotyping",
    "Emotional Reasoning",
    "False Dilemma",
    "Catastrophizing",
    "Hasty Generalization",
    "Black-and-White Thinking",
]

# ── Inference Settings ──────────────────────────────────────────
CONFIDENCE_THRESHOLD: float = 0.5

# ── Input Constraints ──────────────────────────────────────────
MIN_TEXT_LENGTH: int = 10
MAX_TEXT_LENGTH: int = 5000

# ── API Settings ────────────────────────────────────────────────
API_TITLE = "Cognitive Bias Detector & Explainer API"
API_DESCRIPTION = "Detects cognitive biases in text and provides structured explanations."
API_VERSION = "1.0.0"
