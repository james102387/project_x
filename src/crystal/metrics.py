"""
Token cost metrics — measures compute savings from prompt compilation.

Three savings views:
  1. token_savings_pct  — raw token ratio (reflects API billing)
  2. savings_pct        — N+N² isolated compute proxy (legacy, pessimistic for augmented)
  3. marginal_savings_pct — marginal cost against a realistic base context,
                            using (B+N)² where B is the base context length

Uses tiktoken (cl100k_base) for token counting.
"""

from dataclasses import dataclass, asdict

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

    def to_dict(self) -> dict:
        return asdict(self)


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
        # LLM bypassed entirely — 100% savings
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
