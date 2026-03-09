"""
Token cost metrics — measures compute savings from prompt compilation.

Uses tiktoken (cl100k_base) for token counting and a simple N + N^2 compute
proxy where N is the input token count. The two dominant costs in transformer
inference are the FFN pass (linear in N) and self-attention (quadratic in N).
"""

from dataclasses import dataclass, asdict

import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens using the cl100k_base encoding."""
    if not text:
        return 0
    return len(_enc.encode(text))


def compute_proxy(n_tokens: int) -> int:
    """N + N^2 cost model: linear (FFN) + quadratic (attention)."""
    return n_tokens + n_tokens ** 2


@dataclass
class TokenMetrics:
    raw_prompt_tokens: int = 0
    compiled_prompt_tokens: int = 0
    raw_compute: int = 0
    compiled_compute: int = 0
    savings_pct: float = 0.0
    prompt_type: str = ""
    actual_prompt_tokens: int | None = None
    actual_output_tokens: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_metrics(
    raw_prompt: str, compiled_prompt: str, prompt_type: str,
) -> TokenMetrics:
    """
    Estimate token savings for a given prompt compilation.

    For pure_math and math_answerable, the LLM is bypassed entirely (100% savings).
    For math_augmented, savings is the delta between raw and compiled compute proxy.
    For no_math/fallback, the raw prompt goes to the LLM unchanged (0% savings).
    """
    raw_n = count_tokens(raw_prompt)
    raw_cost = compute_proxy(raw_n)

    if prompt_type in ("pure_math", "math_answerable"):
        return TokenMetrics(
            raw_prompt_tokens=raw_n,
            compiled_prompt_tokens=0,
            raw_compute=raw_cost,
            compiled_compute=0,
            savings_pct=1.0,
            prompt_type=prompt_type,
        )

    if prompt_type == "math_augmented":
        comp_n = count_tokens(compiled_prompt)
        comp_cost = compute_proxy(comp_n)
        savings = 1.0 - (comp_cost / raw_cost) if raw_cost > 0 else 0.0
        return TokenMetrics(
            raw_prompt_tokens=raw_n,
            compiled_prompt_tokens=comp_n,
            raw_compute=raw_cost,
            compiled_compute=comp_cost,
            savings_pct=savings,
            prompt_type=prompt_type,
        )

    # no_math / fallback — raw prompt goes to LLM unchanged
    return TokenMetrics(
        raw_prompt_tokens=raw_n,
        compiled_prompt_tokens=raw_n,
        raw_compute=raw_cost,
        compiled_compute=raw_cost,
        savings_pct=0.0,
        prompt_type=prompt_type,
    )
