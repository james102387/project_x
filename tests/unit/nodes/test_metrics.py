"""Unit tests for the token cost metrics module."""

import pytest
from crystal.metrics import (
    count_tokens, compute_proxy, marginal_cost,
    estimate_metrics, TokenMetrics, DEFAULT_BASE_CONTEXT,
    ReasoningComparison, summarize_reasoning_comparisons,
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

    def test_kg_answerable_full_savings(self):
        m = estimate_metrics("What is the capital of Remulak?", "", "kg_answerable")
        assert m.savings_pct == 1.0
        assert m.token_savings_pct == 1.0
        assert m.marginal_savings_pct == 1.0

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

    def test_to_dict_includes_reasoning_fields(self):
        m = TokenMetrics(
            raw_prompt_tokens=10,
            compiled_prompt_tokens=5,
            prompt_type="kg_augmented",
            actual_prompt_tokens=50,
            actual_output_tokens=30,
            actual_reasoning_tokens=200,
            actual_total_tokens=280,
            actual_cached_tokens=10,
        )
        d = m.to_dict()
        assert d["actual_reasoning_tokens"] == 200
        assert d["actual_total_tokens"] == 280
        assert d["actual_cached_tokens"] == 10

    def test_reasoning_fields_default_none(self):
        m = TokenMetrics()
        assert m.actual_reasoning_tokens is None
        assert m.actual_total_tokens is None
        assert m.actual_cached_tokens is None


# ── Reasoning comparison ─────────────────────────────────────────────────


class TestReasoningComparison:
    def test_total_token_delta(self):
        comp = ReasoningComparison(
            baseline_total_tokens=500,
            grounded_total_tokens=300,
        )
        assert comp.total_token_delta == -200

    def test_reasoning_token_delta(self):
        comp = ReasoningComparison(
            baseline_reasoning_tokens=400,
            grounded_reasoning_tokens=100,
        )
        assert comp.reasoning_token_delta == -300

    def test_total_token_savings_pct(self):
        comp = ReasoningComparison(
            baseline_total_tokens=1000,
            grounded_total_tokens=600,
        )
        assert comp.total_token_savings_pct == pytest.approx(0.4)

    def test_reasoning_token_savings_pct(self):
        comp = ReasoningComparison(
            baseline_reasoning_tokens=800,
            grounded_reasoning_tokens=200,
        )
        assert comp.reasoning_token_savings_pct == pytest.approx(0.75)

    def test_none_when_missing_data(self):
        comp = ReasoningComparison(baseline_total_tokens=500)
        assert comp.total_token_delta is None
        assert comp.total_token_savings_pct is None

    def test_zero_baseline_tokens(self):
        comp = ReasoningComparison(
            baseline_total_tokens=0,
            grounded_total_tokens=100,
        )
        assert comp.total_token_savings_pct == 0.0

    def test_zero_baseline_reasoning(self):
        comp = ReasoningComparison(
            baseline_reasoning_tokens=0,
            grounded_reasoning_tokens=50,
        )
        assert comp.reasoning_token_savings_pct == 0.0

    def test_llm_bypass_zero_grounded(self):
        comp = ReasoningComparison(
            question="What is the capital?",
            baseline_total_tokens=500,
            baseline_reasoning_tokens=300,
            grounded_total_tokens=0,
            grounded_reasoning_tokens=0,
            grounded_correct=True,
        )
        assert comp.total_token_savings_pct == pytest.approx(1.0)
        assert comp.reasoning_token_savings_pct == pytest.approx(1.0)

    def test_to_dict_includes_computed_fields(self):
        comp = ReasoningComparison(
            question="test",
            baseline_total_tokens=1000,
            grounded_total_tokens=600,
            baseline_reasoning_tokens=500,
            grounded_reasoning_tokens=100,
        )
        d = comp.to_dict()
        assert d["total_token_delta"] == -400
        assert d["reasoning_token_delta"] == -400
        assert d["total_token_savings_pct"] == pytest.approx(0.4)
        assert d["reasoning_token_savings_pct"] == pytest.approx(0.8)


class TestSummarizeReasoningComparisons:
    def test_empty_list(self):
        summary = summarize_reasoning_comparisons([])
        assert summary["count"] == 0

    def test_basic_summary(self):
        comps = [
            ReasoningComparison(
                question="Q1",
                baseline_total_tokens=1000,
                baseline_reasoning_tokens=600,
                grounded_total_tokens=400,
                grounded_reasoning_tokens=100,
                baseline_correct=False,
                grounded_correct=True,
            ),
            ReasoningComparison(
                question="Q2",
                baseline_total_tokens=800,
                baseline_reasoning_tokens=400,
                grounded_total_tokens=300,
                grounded_reasoning_tokens=50,
                baseline_correct=False,
                grounded_correct=True,
            ),
        ]
        summary = summarize_reasoning_comparisons(comps)
        assert summary["count"] == 2
        assert summary["baseline_accuracy"] == 0.0
        assert summary["grounded_accuracy"] == 1.0
        assert summary["accuracy_delta"] == 1.0
        assert summary["avg_baseline_total_tokens"] == 900.0
        assert summary["avg_grounded_total_tokens"] == 350.0
        assert summary["sum_baseline_total_tokens"] == 1800
        assert summary["sum_grounded_total_tokens"] == 700

    def test_total_token_savings(self):
        comps = [
            ReasoningComparison(
                baseline_total_tokens=1000,
                grounded_total_tokens=500,
            ),
        ]
        summary = summarize_reasoning_comparisons(comps)
        assert summary["total_token_savings_pct"] == pytest.approx(0.5)

    def test_reasoning_token_savings(self):
        comps = [
            ReasoningComparison(
                baseline_reasoning_tokens=800,
                grounded_reasoning_tokens=200,
            ),
        ]
        summary = summarize_reasoning_comparisons(comps)
        assert summary["reasoning_token_savings_pct"] == pytest.approx(0.75)

    def test_missing_data_handled(self):
        comps = [
            ReasoningComparison(question="Q1"),
        ]
        summary = summarize_reasoning_comparisons(comps)
        assert summary["count"] == 1
        assert summary["avg_baseline_total_tokens"] is None
        assert summary["total_token_savings_pct"] is None
