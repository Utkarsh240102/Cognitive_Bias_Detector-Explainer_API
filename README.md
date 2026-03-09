# Cognitive Bias Detector & Explainer API

A FastAPI-powered REST API that detects cognitive biases in text, explains them, and suggests neutral rewrites.

## How It Works

```
User Input  →  Preprocess  →  BART Zero-Shot Classification  →  Bias Selection (threshold 0.5)
                                                                        │
                        Neutral Rewrite (Gemini)  ←  Explanation (Gemini)  ←─┘
```

1. **Preprocessing** — Cleans and normalizes input text (unicode, whitespace, length validation).
2. **Classification** — `facebook/bart-large-mnli` performs zero-shot classification against 8 cognitive bias labels.
3. **Bias Selection** — Filters biases above the 0.5 confidence threshold, sorted by confidence descending.
4. **Explanation** — Google Gemini 2.5 Flash generates a detailed explanation of each detected bias (falls back to templates if unavailable).
5. **Neutral Rewrite** — Gemini rewrites the statement in neutral, unbiased language.

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
- **Classification Model:** facebook/bart-large-mnli (zero-shot)
- **LLM:** Google Gemini 2.5 Flash (explanations & rewrites)
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
GEMINI_API_KEY=your-google-gemini-api-key
```

### 4. Run the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The BART model (~1.6 GB) downloads automatically on first startup and is cached in `model_cache/`.

### 5. Open the docs

Visit [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive Swagger UI.

## Docker

```bash
docker build -t bias-detector .
docker run -p 8000:8000 -e GEMINI_API_KEY=your-key bias-detector
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
    inference.py       # BART model loading & classification
    bias_selector.py   # Threshold filtering & sorting
    explainer.py       # Explanation generation (Gemini + template fallback)
    llm_explainer.py   # Gemini client & prompt building
    rewriter.py        # Neutral rewrite generation (Gemini)
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
