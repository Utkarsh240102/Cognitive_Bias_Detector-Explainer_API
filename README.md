# Cognitive Bias Detector & Explainer API

A FastAPI-powered REST API that detects cognitive biases in text, explains them, and suggests neutral rewrites.

## How It Works

```
User Input  →  Preprocess  →  Fine-Tuned RoBERTa Classification  →  Bias Selection (threshold 0.5)
                                                                              │
                            Neutral Rewrite (Groq)  ←  Explanation (Groq)  ←──┘
```

1. **Preprocessing** — Cleans and normalizes input text (unicode, whitespace, length validation).
2. **Classification** — Fine-tuned RoBERTa-base multi-label classifier (falls back to BART zero-shot if `USE_FINETUNED_MODEL=False` in config).
3. **Bias Selection** — Filters biases above the 0.5 confidence threshold, sorted by confidence descending.
4. **Explanation** — Groq (llama-3.3-70b-versatile) generates a detailed explanation of each detected bias (falls back to templates if unavailable).
5. **Neutral Rewrite** — Groq rewrites the statement in neutral, unbiased language.

## Detected Bias Types

| Bias | Description |
|------|-------------|
| Stereotyping | Assigning traits based on group membership |
| Overgeneralization | Drawing broad conclusions from limited evidence |
| Hasty Generalization | Reaching conclusions from insufficient samples |
| Confirmation Bias | Selectively seeking information that confirms beliefs |
| Emotional Reasoning | Treating feelings as evidence of truth |
| False Dilemma | Presenting only two options when more exist |
| Catastrophizing | Expecting the worst possible outcome |
| Black-and-White Thinking | Seeing situations in absolute extremes |

## Tech Stack

- **Framework:** FastAPI + Uvicorn
- **Classification Model:** Fine-tuned RoBERTa-base (multi-label, trained on synthetic bias dataset)
- **Fallback Model:** facebook/bart-large-mnli (zero-shot)
- **LLM:** Groq — llama-3.3-70b-versatile (explanations & rewrites)
- **Validation:** Pydantic v2
- **Testing:** pytest
- **Python:** 3.13+

## Setup

### 1. Clone and create virtual environment

```bash
git clone <your-repo-url>
cd Cognitive_Bias_Detector_&_Explainer_API

python -m venv myenv
# Windows
myenv\Scripts\activate
# Linux/Mac
source myenv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your-groq-api-key
```

### 4. Run the server using the code

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The fine-tuned RoBERTa model (~500 MB) loads from `trained_model/` on startup. To use the zero-shot BART fallback, set `USE_FINETUNED_MODEL = False` in `app/config.py`.

### 5. Open the docs

Visit [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive Swagger UI.

## Docker

```bash
docker build -t bias-detector .
docker run -p 8000:8000 -e GROQ_API_KEY=your-key bias-detector
```

## API Endpoints

### `POST /analyze`

Analyze text for cognitive biases.

**Request:**

```json
{
  "text": "All politicians are corrupt and only care about themselves."
}
```

**Response:**

```json
{
  "biases": [
    { "type": "Stereotyping", "confidence": 0.92 },
    { "type": "Overgeneralization", "confidence": 0.74 },
    { "type": "Hasty Generalization", "confidence": 0.61 }
  ],
  "explanation": "The statement assigns a negative trait to all politicians based on group membership...",
  "neutral_rewrite": "Some politicians may engage in corrupt practices, but it is inaccurate to generalize this to all politicians."
}
```

### `GET /health`

Health check for monitoring and load balancers.

```json
{
  "status": "ok",
  "version": "1.0.0",
  "model_loaded": true
}
```

## Input Constraints

- **Minimum length:** 10 characters
- **Maximum length:** 5,000 characters
- Text shorter or longer returns a `400 Bad Request` with a descriptive error message.

## Running Tests

```bash
# All tests
pytest tests/ -v

# Fast tests only (no model loading)
pytest tests/test_preprocessor.py tests/test_bias_selector.py tests/test_api.py tests/test_edge_cases.py -v

# Accuracy tests (loads BART model)
pytest tests/test_accuracy.py -v
```

## Project Structure

```
app/
  main.py              # FastAPI entry point, lifespan, exception handlers
  config.py            # Central configuration
  logger.py            # Structured logging
  api/
    routes.py          # /analyze and /health endpoints
  models/
    schemas.py         # Pydantic request/response schemas
  services/
    preprocessor.py    # Text cleaning & validation
    inference.py       # RoBERTa / BART model logic
    bias_selector.py   # Threshold filtering & sorting
    explainer.py       # Explanation generation (Groq + template fallback)
    llm_explainer.py   # Groq client & prompt building
    rewriter.py        # Neutral rewrite generation (Groq)
scripts/
  generate_dataset.py  # Groq LLM data generation
  prepare_dataset.py   # Data cleaning, deduplication, train/val split
  train.py             # RoBERTa fine-tuning script
  evaluate.py          # BART vs RoBERTa comparison script
data/
  dataset.csv          # Combined 1,972 labeled synthetic examples
  train.csv            # Training set (80%)
  val.csv              # Validation set (20%)
trained_model/         # Saved fine-tuned RoBERTa weights (gitignored)
tests/
  test_preprocessor.py # Preprocessor unit tests (13 tests)
  test_bias_selector.py# Bias selector unit tests (11 tests)
  test_api.py          # API integration tests (12 tests)
  test_accuracy.py     # Real-world accuracy tests (30 tests)
  test_edge_cases.py   # Edge case tests (19 tests)
```

## Error Responses

| Status | When |
|--------|------|
| `400` | Text too short or too long |
| `422` | Invalid request body (missing/wrong type) |
| `503` | Model not loaded |
