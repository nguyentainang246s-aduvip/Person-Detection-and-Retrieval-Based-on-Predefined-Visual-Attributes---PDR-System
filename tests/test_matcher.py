"""
tests/test_matcher.py
=====================
Unit tests cho AttributeMatcher (F1).
"""
import pytest
from src.retrieval.matcher import AttributeMatcher


@pytest.fixture
def matcher():
    return AttributeMatcher(default_threshold=0.7)


class TestColorScoring:
    def test_exact_match(self, matcher):
        score = matcher._color_score("Red", "Red")
        assert score == 1.0

    def test_neighbor_match(self, matcher):
        score = matcher._color_score("Black", "Gray")
        assert score == matcher.soft_color_score  # 0.4

    def test_no_match(self, matcher):
        score = matcher._color_score("Red", "Blue")
        assert score == 0.0

    def test_case_insensitive(self, matcher):
        score = matcher._color_score("RED", "red")
        assert score == 1.0


class TestFullMatching:
    def test_perfect_match(self, matcher):
        query = {"gender": "Male", "upper_color": "Black"}
        attrs = {
            "gender": "Male",
            "upper_color": "Black",
            "gender_confidence": 0.9,
            "upper_color_confidence": 0.9,
        }
        is_match, score, breakdown = matcher.is_match(query, attrs)
        assert is_match is True
        assert score >= 0.99

    def test_no_match(self, matcher):
        query = {"gender": "Female", "upper_color": "Red"}
        attrs = {
            "gender": "Male",
            "upper_color": "Blue",
            "gender_confidence": 0.9,
        }
        is_match, score, _ = matcher.is_match(query, attrs)
        assert is_match is False

    def test_low_confidence_skipped(self, matcher):
        """Attribute có confidence < min_confidence phải bị bỏ qua."""
        query = {"gender": "Male", "hat": True}
        attrs = {
            "gender": "Male",
            "gender_confidence": 0.9,
            "hat": False,
            "hat_confidence": 0.3,  # confidence thấp < min_confidence (0.55)
        }
        is_match, score, breakdown = matcher.is_match(query, attrs)
        # hat bị bỏ qua → chỉ tính gender → match
        assert "hat" not in breakdown
        assert is_match is True

    def test_empty_query_always_match(self, matcher):
        is_match, score, _ = matcher.is_match({}, {"gender": "Male"})
        assert is_match is True
        assert score == 1.0
