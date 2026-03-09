"""
Unit tests for app.services.bias_selector
"""

from app.services.bias_selector import select_biases


class TestSelectBiasesFiltering:
    """Tests for threshold-based filtering."""

    def test_returns_scores_above_threshold(self):
        scores = {"Stereotyping": 0.9, "Overgeneralization": 0.7, "Catastrophizing": 0.3}
        result = select_biases(scores)
        types = [b["type"] for b in result]
        assert "Stereotyping" in types
        assert "Overgeneralization" in types
        assert "Catastrophizing" not in types

    def test_excludes_scores_below_threshold(self):
        scores = {"Stereotyping": 0.2, "Overgeneralization": 0.1}
        result = select_biases(scores)
        assert result == []

    def test_includes_score_exactly_at_threshold(self):
        # Threshold is 0.5 — boundary test
        scores = {"Stereotyping": 0.5}
        result = select_biases(scores)
        assert len(result) == 1
        assert result[0]["type"] == "Stereotyping"

    def test_excludes_score_just_below_threshold(self):
        scores = {"Stereotyping": 0.4999}
        result = select_biases(scores)
        assert result == []

    def test_all_above_threshold(self):
        scores = {"A": 0.9, "B": 0.8, "C": 0.7}
        result = select_biases(scores)
        assert len(result) == 3


class TestSelectBiasesSorting:
    """Tests for descending confidence sort order."""

    def test_sorted_by_confidence_descending(self):
        scores = {"A": 0.6, "B": 0.9, "C": 0.7}
        result = select_biases(scores)
        confidences = [b["confidence"] for b in result]
        assert confidences == sorted(confidences, reverse=True)

    def test_order_with_many_biases(self):
        scores = {
            "Stereotyping": 0.92,
            "Overgeneralization": 0.74,
            "Hasty Generalization": 0.61,
            "Confirmation Bias": 0.55,
            "Catastrophizing": 0.30,
        }
        result = select_biases(scores)
        assert len(result) == 4  # Catastrophizing excluded
        assert result[0]["type"] == "Stereotyping"
        assert result[-1]["type"] == "Confirmation Bias"


class TestSelectBiasesOutputFormat:
    """Tests for output structure."""

    def test_output_has_type_and_confidence_keys(self):
        scores = {"Stereotyping": 0.85}
        result = select_biases(scores)
        assert "type" in result[0]
        assert "confidence" in result[0]

    def test_confidence_is_rounded_to_4_decimals(self):
        scores = {"Stereotyping": 0.123456789}
        # 0.1235 rounds, but below threshold — use above-threshold value
        scores = {"Stereotyping": 0.876543219}
        result = select_biases(scores)
        conf_str = str(result[0]["confidence"])
        # At most 4 decimal places
        if "." in conf_str:
            assert len(conf_str.split(".")[1]) <= 4

    def test_returns_list(self):
        scores = {"A": 0.9}
        result = select_biases(scores)
        assert isinstance(result, list)

    def test_empty_input_returns_empty_list(self):
        result = select_biases({})
        assert result == []
