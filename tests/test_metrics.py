"""Unit tests for the token cost metrics module."""

import pytest
from crystal.metrics import (
    count_tokens, compute_proxy, marginal_cost,
    estimate_metrics, TokenMetrics, DEFAULT_BASE_CONTEXT,
)


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


class TestMarginalCost:
    def test_zero_tokens(self):
        assert marginal_cost(0, 2000) == 0

    def test_zero_base(self):
        assert marginal_cost(10, 0) == compute_proxy(10)

    def test_formula(self):
        n, b = 10, 2000
        assert marginal_cost(n, b) == n + 2 * b * n + n ** 2

    def test_larger_base_increases_cost(self):
        small = marginal_cost(10, 500)
        large = marginal_cost(10, 5000)
        assert large > small


class TestEstimateMetrics:
    def test_pure_math_full_savings(self):
        m = estimate_metrics("5 + 3", "", "pure_math")
        assert m.savings_pct == 1.0
        assert m.token_savings_pct == 1.0
        assert m.marginal_savings_pct == 1.0
        assert m.compiled_prompt_tokens == 0
        assert m.compiled_compute == 0
        assert m.raw_prompt_tokens > 0

    def test_math_answerable_full_savings(self):
        m = estimate_metrics("John has 10 apples and buys 5 more", "", "math_answerable")
        assert m.savings_pct == 1.0
        assert m.token_savings_pct == 1.0
        assert m.marginal_savings_pct == 1.0
        assert m.compiled_prompt_tokens == 0
        assert m.compiled_compute == 0

    def test_no_match_zero_savings(self):
        m = estimate_metrics("hello world", "hello world", "no_match")
        assert m.savings_pct == 0.0
        assert m.token_savings_pct == 0.0
        assert m.marginal_savings_pct == 0.0
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
        assert isinstance(m.token_savings_pct, float)
        assert isinstance(m.marginal_savings_pct, float)
        assert m.compiled_prompt_tokens > 0
        assert m.raw_prompt_tokens > 0

    def test_math_augmented_token_savings_less_extreme(self):
        """Token savings should be less extreme than isolated N+N² savings."""
        raw = "She earned 500 and spent 300"
        compiled = (
            "The user asked: 'She earned 500 and spent 300'\n\n"
            "The following has been computed with verified precision:\n"
            "  start with 500, - 300 (spent) = 200\n\n"
            "Answer the user's question using the computed result. "
            "Do not recalculate."
        )
        m = estimate_metrics(raw, compiled, "math_augmented")
        assert m.token_savings_pct > m.savings_pct

    def test_math_augmented_marginal_less_extreme_than_isolated(self):
        """Marginal savings should be closer to token savings than isolated."""
        raw = "She earned 500 and spent 300"
        compiled = (
            "The user asked: 'She earned 500 and spent 300'\n\n"
            "The following has been computed with verified precision:\n"
            "  start with 500, - 300 (spent) = 200\n\n"
            "Answer the user's question using the computed result. "
            "Do not recalculate."
        )
        m = estimate_metrics(raw, compiled, "math_augmented")
        assert m.marginal_savings_pct > m.savings_pct

    def test_custom_base_context(self):
        raw = "She earned 500 and spent 300"
        compiled = (
            "The user asked: 'She earned 500 and spent 300'\n\n"
            "Computed: 200\n\nAnswer using this result."
        )
        small_ctx = estimate_metrics(raw, compiled, "math_augmented", base_context=100)
        large_ctx = estimate_metrics(raw, compiled, "math_augmented", base_context=10000)
        assert large_ctx.marginal_savings_pct > small_ctx.marginal_savings_pct

    def test_to_dict(self):
        m = estimate_metrics("5 + 3", "", "pure_math")
        d = m.to_dict()
        assert isinstance(d, dict)
        assert d["savings_pct"] == 1.0
        assert d["token_savings_pct"] == 1.0
        assert d["marginal_savings_pct"] == 1.0
        assert d["prompt_type"] == "pure_math"
        assert "raw_prompt_tokens" in d
