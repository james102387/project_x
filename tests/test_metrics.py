"""Unit tests for the token cost metrics module."""

import pytest
from crystal.metrics import count_tokens, compute_proxy, estimate_metrics, TokenMetrics


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_nonempty_string(self):
        assert count_tokens("hello world") > 0

    def test_returns_int(self):
        assert isinstance(count_tokens("5 plus 3"), int)

    def test_longer_string_more_tokens(self):
        short = count_tokens("hello")
        long = count_tokens("hello world, this is a longer sentence")
        assert long > short


class TestComputeProxy:
    def test_zero(self):
        assert compute_proxy(0) == 0

    def test_one(self):
        assert compute_proxy(1) == 2

    def test_ten(self):
        assert compute_proxy(10) == 110

    def test_formula(self):
        for n in (5, 20, 100):
            assert compute_proxy(n) == n + n ** 2


class TestEstimateMetrics:
    def test_pure_math_full_savings(self):
        m = estimate_metrics("5 + 3", "", "pure_math")
        assert m.savings_pct == 1.0
        assert m.compiled_prompt_tokens == 0
        assert m.compiled_compute == 0
        assert m.raw_prompt_tokens > 0

    def test_math_answerable_full_savings(self):
        m = estimate_metrics("John has 10 apples and buys 5 more", "", "math_answerable")
        assert m.savings_pct == 1.0
        assert m.compiled_prompt_tokens == 0
        assert m.compiled_compute == 0

    def test_no_match_zero_savings(self):
        m = estimate_metrics("hello world", "hello world", "no_match")
        assert m.savings_pct == 0.0
        assert m.compiled_prompt_tokens == m.raw_prompt_tokens
        assert m.compiled_compute == m.raw_compute

    def test_math_augmented_has_valid_savings(self):
        raw = "She earned 500 and spent 300"
        compiled = (
            "The user asked: 'She earned 500 and spent 300'\n\n"
            "The following has been computed with verified precision:\n"
            "  start with 500, - 300 (spent) = 200\n\n"
            "Answer the user's question using the computed result. "
            "Do not recalculate."
        )
        m = estimate_metrics(raw, compiled, "math_augmented")
        assert isinstance(m.savings_pct, float)
        assert m.compiled_prompt_tokens > 0
        assert m.raw_prompt_tokens > 0

    def test_to_dict(self):
        m = estimate_metrics("5 + 3", "", "pure_math")
        d = m.to_dict()
        assert isinstance(d, dict)
        assert d["savings_pct"] == 1.0
        assert d["prompt_type"] == "pure_math"
        assert "raw_prompt_tokens" in d
