"""
Integration tests for the POST /analyze endpoint.

These use FastAPI's TestClient and mock the expensive dependencies
(BART model + Gemini API) so tests run fast and offline.
"""

from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    """Create a TestClient that skips model loading."""
    with patch("app.main.load_model"), patch("app.main.load_llm"):
        with TestClient(app) as c:
            yield c


# ── Fake model scores used by mocked classify() ────────────────
FAKE_SCORES = {
    "Stereotyping": 0.92,
    "Overgeneralization": 0.74,
    "Hasty Generalization": 0.61,
    "Confirmation Bias": 0.15,
    "Emotional Reasoning": 0.10,
    "False Dilemma": 0.05,
    "Catastrophizing": 0.08,
    "Black-and-White Thinking": 0.20,
}


class TestAnalyzeEndpointSuccess:
    """Happy-path tests for POST /analyze."""

    @patch("app.api.routes.generate_rewrite", return_value="Neutral version.")
    @patch("app.api.routes.generate_explanation", return_value="Explanation text.")
    @patch("app.api.routes.classify", return_value=FAKE_SCORES)
    def test_returns_200(self, mock_cls, mock_exp, mock_rw, client):
        resp = client.post("/analyze", json={"text": "All old people are bad with tech"})
        assert resp.status_code == 200

    @patch("app.api.routes.generate_rewrite", return_value="Neutral version.")
    @patch("app.api.routes.generate_explanation", return_value="Explanation text.")
    @patch("app.api.routes.classify", return_value=FAKE_SCORES)
    def test_response_has_required_fields(self, mock_cls, mock_exp, mock_rw, client):
        resp = client.post("/analyze", json={"text": "All old people are bad with tech"})
        data = resp.json()
        assert "biases" in data
        assert "explanation" in data
        assert "neutral_rewrite" in data

    @patch("app.api.routes.generate_rewrite", return_value="Neutral version.")
    @patch("app.api.routes.generate_explanation", return_value="Explanation text.")
    @patch("app.api.routes.classify", return_value=FAKE_SCORES)
    def test_biases_filtered_above_threshold(self, mock_cls, mock_exp, mock_rw, client):
        resp = client.post("/analyze", json={"text": "All old people are bad with tech"})
        biases = resp.json()["biases"]
        # Only Stereotyping (0.92), Overgeneralization (0.74), Hasty Generalization (0.61) are >= 0.5
        assert len(biases) == 3
        types = [b["type"] for b in biases]
        assert "Stereotyping" in types
        assert "Overgeneralization" in types
        assert "Hasty Generalization" in types

    @patch("app.api.routes.generate_rewrite", return_value="Neutral version.")
    @patch("app.api.routes.generate_explanation", return_value="Explanation text.")
    @patch("app.api.routes.classify", return_value=FAKE_SCORES)
    def test_biases_sorted_descending(self, mock_cls, mock_exp, mock_rw, client):
        resp = client.post("/analyze", json={"text": "All old people are bad with tech"})
        biases = resp.json()["biases"]
        confidences = [b["confidence"] for b in biases]
        assert confidences == sorted(confidences, reverse=True)

    @patch("app.api.routes.generate_rewrite", return_value="Neutral version.")
    @patch("app.api.routes.generate_explanation", return_value="Explanation text.")
    @patch("app.api.routes.classify", return_value=FAKE_SCORES)
    def test_explanation_is_string(self, mock_cls, mock_exp, mock_rw, client):
        resp = client.post("/analyze", json={"text": "All old people are bad with tech"})
        assert isinstance(resp.json()["explanation"], str)

    @patch("app.api.routes.generate_rewrite", return_value="Neutral version.")
    @patch("app.api.routes.generate_explanation", return_value="Explanation text.")
    @patch("app.api.routes.classify", return_value=FAKE_SCORES)
    def test_neutral_rewrite_is_string(self, mock_cls, mock_exp, mock_rw, client):
        resp = client.post("/analyze", json={"text": "All old people are bad with tech"})
        assert isinstance(resp.json()["neutral_rewrite"], str)


class TestAnalyzeEndpointNoBias:
    """Tests when no bias exceeds the threshold."""

    NO_BIAS_SCORES = {label: 0.1 for label in FAKE_SCORES}

    @patch("app.api.routes.generate_rewrite", return_value="Same text.")
    @patch("app.api.routes.generate_explanation", return_value="No significant biases.")
    @patch("app.api.routes.classify", return_value=NO_BIAS_SCORES)
    def test_returns_empty_biases_list(self, mock_cls, mock_exp, mock_rw, client):
        resp = client.post("/analyze", json={"text": "The weather is nice today, quite pleasant."})
        assert resp.json()["biases"] == []


class TestAnalyzeEndpointValidation:
    """Tests for input validation / error responses."""

    def test_rejects_missing_text_field(self, client):
        resp = client.post("/analyze", json={})
        assert resp.status_code == 422

    def test_rejects_empty_string(self, client):
        resp = client.post("/analyze", json={"text": ""})
        assert resp.status_code == 422

    def test_rejects_too_short_text(self, client):
        resp = client.post("/analyze", json={"text": "Hi"})
        assert resp.status_code == 422

    def test_rejects_non_string_text(self, client):
        resp = client.post("/analyze", json={"text": 12345})
        assert resp.status_code == 422

    def test_rejects_no_body(self, client):
        resp = client.post("/analyze")
        assert resp.status_code == 422
