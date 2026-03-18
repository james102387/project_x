"""
Token cost metrics — measures compute savings from prompt compilation.

Four savings views:
  1. token_savings_pct    — raw token ratio (reflects API billing)
  2. savings_pct          — N+N² isolated compute proxy (legacy, pessimistic for augmented)
  3. marginal_savings_pct — marginal cost against a realistic base context,
                            using (B+N)² where B is the base context length
  4. total_token_savings_pct — actual total token savings (prompt + output + reasoning)
                               when real API usage data is available

Uses tiktoken (cl100k_base) for token counting.
"""

from dataclasses import dataclass, asdict, field

import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")

DEFAULT_BASE_CONTEXT = 2000


def count_tokens(text: str) -> int:
    """Count tokens using the cl100k_base encoding."""
    if not text:
        return 0
    return len(_enc.encode(text))


def compute_proxy(n_tokens: int) -> int:
    """N + N^2 cost model: linear (FFN) + quadratic (attention)."""
    return n_tokens + n_tokens ** 2


def marginal_cost(n_tokens: int, base_context: int) -> int:
    """Cost of adding n_tokens to an existing base_context.

    Full cost is (B+N) + (B+N)²; the base-only cost is B + B².
    Marginal = full - base = N + 2BN + N².
    """
    return n_tokens + 2 * base_context * n_tokens + n_tokens ** 2


@dataclass
class TokenMetrics:
    raw_prompt_tokens: int = 0
    compiled_prompt_tokens: int = 0
    raw_compute: int = 0
    compiled_compute: int = 0
    savings_pct: float = 0.0
    token_savings_pct: float = 0.0
    marginal_savings_pct: float = 0.0
    prompt_type: str = ""
    actual_prompt_tokens: int | None = None
    actual_output_tokens: int | None = None
    actual_reasoning_tokens: int | None = None
    actual_total_tokens: int | None = None
    actual_cached_tokens: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReasoningComparison:
    """Per-query comparison of grounded vs ungrounded token usage."""
    question: str = ""
    # Ungrounded (baseline)
    baseline_prompt_tokens: int | None = None
    baseline_output_tokens: int | None = None
    baseline_reasoning_tokens: int | None = None
    baseline_total_tokens: int | None = None
    baseline_correct: bool = False
    # Grounded (treatment)
    grounded_prompt_tokens: int | None = None
    grounded_output_tokens: int | None = None
    grounded_reasoning_tokens: int | None = None
    grounded_total_tokens: int | None = None
    grounded_correct: bool = False

    @property
    def total_token_delta(self) -> int | None:
        if self.baseline_total_tokens is None or self.grounded_total_tokens is None:
            return None
        return self.grounded_total_tokens - self.baseline_total_tokens

    @property
    def reasoning_token_delta(self) -> int | None:
        if self.baseline_reasoning_tokens is None or self.grounded_reasoning_tokens is None:
            return None
        return self.grounded_reasoning_tokens - self.baseline_reasoning_tokens

    @property
    def total_token_savings_pct(self) -> float | None:
        if self.baseline_total_tokens is None or self.grounded_total_tokens is None:
            return None
        if self.baseline_total_tokens == 0:
            return 0.0
        return 1.0 - (self.grounded_total_tokens / self.baseline_total_tokens)

    @property
    def reasoning_token_savings_pct(self) -> float | None:
        if self.baseline_reasoning_tokens is None or self.grounded_reasoning_tokens is None:
            return None
        if self.baseline_reasoning_tokens == 0:
            return 0.0
        return 1.0 - (self.grounded_reasoning_tokens / self.baseline_reasoning_tokens)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_token_delta"] = self.total_token_delta
        d["reasoning_token_delta"] = self.reasoning_token_delta
        d["total_token_savings_pct"] = self.total_token_savings_pct
        d["reasoning_token_savings_pct"] = self.reasoning_token_savings_pct
        return d


def summarize_reasoning_comparisons(
    comparisons: list[ReasoningComparison],
) -> dict:
    """Aggregate per-query reasoning comparisons into summary statistics."""
    n = len(comparisons)
    if n == 0:
        return {"count": 0}

    baseline_totals = [c.baseline_total_tokens for c in comparisons if c.baseline_total_tokens is not None]
    grounded_totals = [c.grounded_total_tokens for c in comparisons if c.grounded_total_tokens is not None]
    baseline_reasoning = [c.baseline_reasoning_tokens for c in comparisons if c.baseline_reasoning_tokens is not None]
    grounded_reasoning = [c.grounded_reasoning_tokens for c in comparisons if c.grounded_reasoning_tokens is not None]

    baseline_correct = sum(1 for c in comparisons if c.baseline_correct)
    grounded_correct = sum(1 for c in comparisons if c.grounded_correct)

    def _safe_avg(vals: list) -> float | None:
        return sum(vals) / len(vals) if vals else None

    def _safe_sum(vals: list) -> int | None:
        return sum(vals) if vals else None

    baseline_total_sum = _safe_sum(baseline_totals)
    grounded_total_sum = _safe_sum(grounded_totals)
    baseline_reasoning_sum = _safe_sum(baseline_reasoning)
    grounded_reasoning_sum = _safe_sum(grounded_reasoning)

    total_savings = None
    if baseline_total_sum and grounded_total_sum:
        total_savings = 1.0 - (grounded_total_sum / baseline_total_sum)

    reasoning_savings = None
    if baseline_reasoning_sum and grounded_reasoning_sum:
        reasoning_savings = 1.0 - (grounded_reasoning_sum / baseline_reasoning_sum)

    return {
        "count": n,
        "baseline_accuracy": baseline_correct / n,
        "grounded_accuracy": grounded_correct / n,
        "accuracy_delta": (grounded_correct - baseline_correct) / n,
        "avg_baseline_total_tokens": _safe_avg(baseline_totals),
        "avg_grounded_total_tokens": _safe_avg(grounded_totals),
        "avg_baseline_reasoning_tokens": _safe_avg(baseline_reasoning),
        "avg_grounded_reasoning_tokens": _safe_avg(grounded_reasoning),
        "total_token_savings_pct": total_savings,
        "reasoning_token_savings_pct": reasoning_savings,
        "sum_baseline_total_tokens": baseline_total_sum,
        "sum_grounded_total_tokens": grounded_total_sum,
        "sum_baseline_reasoning_tokens": baseline_reasoning_sum,
        "sum_grounded_reasoning_tokens": grounded_reasoning_sum,
    }


def estimate_metrics(
    raw_prompt: str,
    compiled_prompt: str,
    prompt_type: str,
    base_context: int = DEFAULT_BASE_CONTEXT,
) -> TokenMetrics:
    """
    Estimate token savings for a given prompt compilation.

    For pure_math and math_answerable, the LLM is bypassed entirely (100% savings).
    For math_augmented, computes three savings views (token, isolated, marginal).
    For no_math/fallback, the raw prompt goes to the LLM unchanged (0% savings).
    """
    raw_n = count_tokens(raw_prompt)
    raw_cost = compute_proxy(raw_n)

    if prompt_type in ("pure_math", "math_answerable", "kg_answerable"):
        return TokenMetrics(
            raw_prompt_tokens=raw_n,
            compiled_prompt_tokens=0,
            raw_compute=raw_cost,
            compiled_compute=0,
            savings_pct=1.0,
            token_savings_pct=1.0,
            marginal_savings_pct=1.0,
            prompt_type=prompt_type,
        )

    if prompt_type in ("math_augmented", "kg_augmented"):
        comp_n = count_tokens(compiled_prompt)
        comp_cost = compute_proxy(comp_n)

        iso_savings = 1.0 - (comp_cost / raw_cost) if raw_cost > 0 else 0.0
        tok_savings = 1.0 - (comp_n / raw_n) if raw_n > 0 else 0.0

        raw_marginal = marginal_cost(raw_n, base_context)
        comp_marginal = marginal_cost(comp_n, base_context)
        mar_savings = (
            1.0 - (comp_marginal / raw_marginal) if raw_marginal > 0 else 0.0
        )

        return TokenMetrics(
            raw_prompt_tokens=raw_n,
            compiled_prompt_tokens=comp_n,
            raw_compute=raw_cost,
            compiled_compute=comp_cost,
            savings_pct=iso_savings,
            token_savings_pct=tok_savings,
            marginal_savings_pct=mar_savings,
            prompt_type=prompt_type,
        )

    # no_math / fallback — raw prompt goes to LLM unchanged
    return TokenMetrics(
        raw_prompt_tokens=raw_n,
        compiled_prompt_tokens=raw_n,
        raw_compute=raw_cost,
        compiled_compute=raw_cost,
        savings_pct=0.0,
        token_savings_pct=0.0,
        marginal_savings_pct=0.0,
        prompt_type=prompt_type,
    )
