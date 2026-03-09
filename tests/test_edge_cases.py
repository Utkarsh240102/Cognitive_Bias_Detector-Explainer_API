"""
Edge case tests — empty text, very long text, single word, special
characters, repeated text, neutral text, and boundary inputs.

Uses mocked model so these run fast.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    """Create a TestClient that skips model loading."""
    with patch("app.main.load_model"), patch("app.main.load_llm"):
        with TestClient(app) as c:
            yield c


MOCK_SCORES = {
    "Stereotyping": 0.1, "Overgeneralization": 0.1,
    "Hasty Generalization": 0.1, "Confirmation Bias": 0.1,
    "Emotional Reasoning": 0.1, "False Dilemma": 0.1,
    "Catastrophizing": 0.1, "Black-and-White Thinking": 0.1,
}


class TestEdgeCaseValidation:
    """Inputs that should be rejected with 422."""

    def test_empty_string(self, client):
        resp = client.post("/analyze", json={"text": ""})
        assert resp.status_code == 422

    def test_whitespace_only(self, client):
        resp = client.post("/analyze", json={"text": "         "})
        assert resp.status_code == 422

    def test_single_character(self, client):
        resp = client.post("/analyze", json={"text": "A"})
        assert resp.status_code == 422

    def test_nine_characters(self, client):
        resp = client.post("/analyze", json={"text": "123456789"})
        assert resp.status_code == 422

    def test_text_exceeds_5000_chars(self, client):
        resp = client.post("/analyze", json={"text": "A" * 5001})
        assert resp.status_code == 422

    def test_null_text(self, client):
        resp = client.post("/analyze", json={"text": None})
        assert resp.status_code == 422

    def test_integer_text(self, client):
        resp = client.post("/analyze", json={"text": 42})
        assert resp.status_code == 422

    def test_list_text(self, client):
        resp = client.post("/analyze", json={"text": ["a", "b"]})
        assert resp.status_code == 422

    def test_missing_field(self, client):
        resp = client.post("/analyze", json={"wrong_key": "hello world test"})
        assert resp.status_code == 422


class TestEdgeCaseAccepted:
    """Inputs that should be accepted (status 200)."""

    @patch("app.api.routes.generate_rewrite", return_value="ok")
    @patch("app.api.routes.generate_explanation", return_value="ok")
    @patch("app.api.routes.classify", return_value=MOCK_SCORES)
    def test_exactly_10_characters(self, mock_cls, mock_exp, mock_rw, client):
        resp = client.post("/analyze", json={"text": "AAAAAAAAAA"})
        assert resp.status_code == 200

    @patch("app.api.routes.generate_rewrite", return_value="ok")
    @patch("app.api.routes.generate_explanation", return_value="ok")
    @patch("app.api.routes.classify", return_value=MOCK_SCORES)
    def test_exactly_5000_characters(self, mock_cls, mock_exp, mock_rw, client):
        resp = client.post("/analyze", json={"text": "A" * 5000})
        assert resp.status_code == 200

    @patch("app.api.routes.generate_rewrite", return_value="ok")
    @patch("app.api.routes.generate_explanation", return_value="ok")
    @patch("app.api.routes.classify", return_value=MOCK_SCORES)
    def test_text_with_special_characters(self, mock_cls, mock_exp, mock_rw, client):
        resp = client.post("/analyze", json={"text": "Hello! @#$%^&*() special chars here."})
        assert resp.status_code == 200

    @patch("app.api.routes.generate_rewrite", return_value="ok")
    @patch("app.api.routes.generate_explanation", return_value="ok")
    @patch("app.api.routes.classify", return_value=MOCK_SCORES)
    def test_text_with_unicode(self, mock_cls, mock_exp, mock_rw, client):
        resp = client.post("/analyze", json={"text": "Ça fait du bien — très intéressant!"})
        assert resp.status_code == 200

    @patch("app.api.routes.generate_rewrite", return_value="ok")
    @patch("app.api.routes.generate_explanation", return_value="ok")
    @patch("app.api.routes.classify", return_value=MOCK_SCORES)
    def test_text_with_emojis(self, mock_cls, mock_exp, mock_rw, client):
        resp = client.post("/analyze", json={"text": "This is a test with emojis 😊🔥💡"})
        assert resp.status_code == 200

    @patch("app.api.routes.generate_rewrite", return_value="ok")
    @patch("app.api.routes.generate_explanation", return_value="ok")
    @patch("app.api.routes.classify", return_value=MOCK_SCORES)
    def test_text_with_newlines(self, mock_cls, mock_exp, mock_rw, client):
        resp = client.post("/analyze", json={"text": "Line one\nLine two\nLine three"})
        assert resp.status_code == 200

    @patch("app.api.routes.generate_rewrite", return_value="ok")
    @patch("app.api.routes.generate_explanation", return_value="ok")
    @patch("app.api.routes.classify", return_value=MOCK_SCORES)
    def test_repeated_word(self, mock_cls, mock_exp, mock_rw, client):
        text = "test " * 50  # 250 chars of repeated word
        resp = client.post("/analyze", json={"text": text.strip()})
        assert resp.status_code == 200


class TestEdgeCaseResponseShape:
    """Verify response structure on edge-case inputs."""

    @patch("app.api.routes.generate_rewrite", return_value="original")
    @patch("app.api.routes.generate_explanation", return_value="No biases.")
    @patch("app.api.routes.classify", return_value=MOCK_SCORES)
    def test_neutral_text_returns_empty_biases(self, mock_cls, mock_exp, mock_rw, client):
        resp = client.post("/analyze", json={"text": "The meeting is at three o'clock."})
        data = resp.json()
        assert data["biases"] == []

    @patch("app.api.routes.generate_rewrite", return_value="original")
    @patch("app.api.routes.generate_explanation", return_value="No biases.")
    @patch("app.api.routes.classify", return_value=MOCK_SCORES)
    def test_response_content_type_is_json(self, mock_cls, mock_exp, mock_rw, client):
        resp = client.post("/analyze", json={"text": "A normal sentence for testing."})
        assert resp.headers["content-type"] == "application/json"

    def test_get_method_not_allowed(self, client):
        resp = client.get("/analyze")
        assert resp.status_code == 405
