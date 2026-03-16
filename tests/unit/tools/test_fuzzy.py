"""Unit tests for fuzzy string matching module."""

import pytest
from crystal.tools.kg.fuzzy import fuzzy_match


class TestFuzzyMatch:
    def test_exact_match_returns_high_score(self):
        result = fuzzy_match("remulak", ["remulak", "draveth"])
        assert result is not None
        assert result[0] == "remulak"
        assert result[1] == 100.0

    def test_typo_matches(self):
        result = fuzzy_match("remulack", ["remulak", "draveth", "sulari"])
        assert result is not None
        assert result[0] == "remulak"
        assert result[1] >= 80.0

    def test_word_reorder_matches(self):
        result = fuzzy_match(
            "vizier grand korth",
            ["grand vizier korth", "draveth"],
        )
        assert result is not None
        assert result[0] == "grand vizier korth"

    def test_below_threshold_returns_none(self):
        result = fuzzy_match("completely different", ["remulak", "draveth"])
        assert result is None

    def test_custom_threshold(self):
        result = fuzzy_match("remulck", ["remulak"], threshold=99.0)
        assert result is None
        result = fuzzy_match("remulck", ["remulak"], threshold=70.0)
        assert result is not None

    def test_empty_candidates(self):
        result = fuzzy_match("remulak", [])
        assert result is None

    def test_case_insensitive(self):
        result = fuzzy_match("REMULAK", ["remulak"])
        assert result is not None
        assert result[0] == "remulak"

    def test_predicate_fuzzy(self):
        result = fuzzy_match("capitl", ["capital", "climate", "population"])
        assert result is not None
        assert result[0] == "capital"
