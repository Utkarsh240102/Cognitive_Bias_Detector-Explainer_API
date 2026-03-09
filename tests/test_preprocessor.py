"""
Unit tests for app.services.preprocessor
"""

import pytest

from app.services.preprocessor import preprocess


class TestPreprocessBasicCleaning:
    """Tests for whitespace, unicode, and normalization."""

    def test_strips_leading_trailing_whitespace(self):
        result = preprocess("   Hello world!   ")
        assert result == "Hello world!"

    def test_collapses_multiple_spaces(self):
        result = preprocess("Hello    world    test")
        assert result == "Hello world test"

    def test_collapses_tabs_and_newlines(self):
        result = preprocess("Hello\t\tworld\n\ntest input")
        assert result == "Hello world test input"

    def test_unicode_normalization(self):
        # NFKC normalizes ﬁ (U+FB01) → "fi"
        result = preprocess("The ﬁrst test of normalization")
        assert "fi" in result

    def test_preserves_normal_text(self):
        text = "This is a perfectly normal sentence."
        assert preprocess(text) == text


class TestPreprocessLengthValidation:
    """Tests for min/max length constraints."""

    def test_rejects_text_too_short(self):
        with pytest.raises(ValueError, match="too short"):
            preprocess("Hi")

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError, match="too short"):
            preprocess("")

    def test_rejects_whitespace_only(self):
        with pytest.raises(ValueError, match="too short"):
            preprocess("         ")

    def test_accepts_minimum_length(self):
        # MIN_TEXT_LENGTH is 10
        result = preprocess("A" * 10)
        assert len(result) == 10

    def test_rejects_text_too_long(self):
        # MAX_TEXT_LENGTH is 5000
        with pytest.raises(ValueError, match="too long"):
            preprocess("A" * 5001)

    def test_accepts_maximum_length(self):
        text = "A" * 5000
        assert preprocess(text) == text


class TestPreprocessReturnType:
    """Verify return value."""

    def test_returns_string(self):
        result = preprocess("This is a test sentence.")
        assert isinstance(result, str)

    def test_length_shrinks_after_cleaning(self):
        raw = "   Hello     world   test  input  "
        result = preprocess(raw)
        assert len(result) < len(raw)
