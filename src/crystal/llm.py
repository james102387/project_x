"""LLM client wrapper. Swap providers by changing this file only.

Set LLM_PROVIDER=anthropic to use Claude, or leave unset/gemini for Gemini.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.5-flash")

_llm_client = None


def _get_client():
    """Lazy-init the LLM client so imports work without an API key."""
    global _llm_client
    if _llm_client is not None:
        return _llm_client

    if LLM_PROVIDER == "anthropic":
        import anthropic
        _llm_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    else:
        from google import genai
        _llm_client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    return _llm_client


def _extract_usage_gemini(response) -> dict | None:
    """Extract token usage from a Gemini response."""
    if not hasattr(response, "usage_metadata") or not response.usage_metadata:
        return None
    um = response.usage_metadata
    usage = {
        "prompt_tokens": getattr(um, "prompt_token_count", None),
        "output_tokens": getattr(um, "candidates_token_count", None),
        "reasoning_tokens": getattr(um, "thoughts_token_count", None),
        "cached_tokens": getattr(um, "cached_content_token_count", None),
    }
    total = 0
    for key in ("prompt_tokens", "output_tokens", "reasoning_tokens"):
        if usage[key] is not None:
            total += usage[key]
    usage["total_tokens"] = total if total > 0 else None
    return usage


def _extract_usage_anthropic(response) -> dict | None:
    """Extract token usage from an Anthropic response."""
    if not hasattr(response, "usage") or not response.usage:
        return None
    u = response.usage
    return {
        "prompt_tokens": getattr(u, "input_tokens", None),
        "output_tokens": getattr(u, "output_tokens", None),
        "reasoning_tokens": None,
        "cached_tokens": getattr(u, "cache_read_input_tokens", None),
        "total_tokens": (getattr(u, "input_tokens", 0) or 0) + (getattr(u, "output_tokens", 0) or 0),
    }


THINKING_BUDGET = int(os.environ.get("LLM_THINKING_BUDGET", "0"))


def _call_anthropic(client, prompt: str) -> tuple[str, dict | None]:
    """Call the Anthropic API, optionally with extended thinking."""
    model = LLM_MODEL if "claude" in LLM_MODEL else "claude-haiku-4-5"

    kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    if THINKING_BUDGET > 0:
        kwargs["max_tokens"] = THINKING_BUDGET + 4096
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": THINKING_BUDGET}
    else:
        kwargs["max_tokens"] = 1024

    response = client.messages.create(**kwargs)

    text = ""
    thinking_text = ""
    for block in response.content:
        if block.type == "thinking":
            thinking_text = block.thinking
        elif block.type == "text":
            text = block.text.strip()

    usage = _extract_usage_anthropic(response)
    if usage is not None:
        usage["thinking_chars"] = len(thinking_text)
    return text, usage


def _call_gemini(client, prompt: str) -> tuple[str, dict | None]:
    """Call the Gemini API."""
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
    )
    return response.text.strip(), _extract_usage_gemini(response)


def call_llm(prompt: str, max_retries: int = 6) -> tuple[str, dict | None]:
    """
    Call the LLM with retry logic for rate limits.

    Returns (response_text, usage_dict) where usage_dict contains
    prompt_tokens, output_tokens, reasoning_tokens (thinking models),
    and total_tokens from the API response, or None if unavailable.
    """
    client = _get_client()
    call_fn = _call_anthropic if LLM_PROVIDER == "anthropic" else _call_gemini

    for attempt in range(max_retries):
        try:
            return call_fn(client, prompt)
        except Exception as e:
            err = str(e)
            if "429" in err or "TooManyRequests" in err or "RESOURCE_EXHAUSTED" in err or "rate" in err.lower():
                if attempt < max_retries - 1:
                    wait = min(2 ** attempt * 5, 60)
                    print(
                        f"    Rate limited, waiting {wait}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait)
            else:
                raise
    return "[ERROR: Rate limit exceeded after retries]", None


# ---------------------------------------------------------------------------
# Gemini Batch API
# ---------------------------------------------------------------------------

def submit_batch(
    prompts: list[str],
    *,
    display_name: str = "crystal-benchmark",
) -> str:
    """Submit a batch of prompts to the Gemini Batch API.

    Returns the batch job name (ID) for polling.
    Only works with LLM_PROVIDER=gemini.
    """
    if LLM_PROVIDER != "gemini":
        raise RuntimeError("Batch API only supported for Gemini provider")

    client = _get_client()
    inline_requests = [
        {"contents": [{"parts": [{"text": p}], "role": "user"}]}
        for p in prompts
    ]

    batch_job = client.batches.create(
        model=LLM_MODEL,
        src=inline_requests,
        config={"display_name": display_name},
    )
    return batch_job.name


def poll_batch(
    job_name: str,
    *,
    poll_interval: float = 10.0,
    timeout: float = 3600.0,
) -> list[tuple[str, dict | None]]:
    """Poll a Gemini batch job until completion.

    Returns a list of (response_text, usage_dict) in the same order
    as the prompts submitted.
    """
    client = _get_client()
    start = time.time()

    while True:
        job = client.batches.get(name=job_name)
        state = str(getattr(job, "state", ""))

        if "SUCCEEDED" in state:
            break
        if "FAILED" in state or "CANCELLED" in state:
            raise RuntimeError(f"Batch job {job_name} ended with state: {state}")

        elapsed = time.time() - start
        if elapsed > timeout:
            raise TimeoutError(f"Batch job {job_name} timed out after {timeout}s")

        print(f"    Batch {job_name}: {state} ({elapsed:.0f}s elapsed)")
        time.sleep(poll_interval)

    results = []
    responses = getattr(job, "dest", None)
    if responses and hasattr(responses, "inlined_responses"):
        for resp in responses.inlined_responses:
            if hasattr(resp, "response") and resp.response:
                inner = resp.response
                text = ""
                if hasattr(inner, "candidates") and inner.candidates:
                    parts = inner.candidates[0].content.parts
                    text = parts[0].text.strip() if parts else ""
                usage = _extract_usage_gemini(inner)
                results.append((text, usage))
            else:
                error = getattr(resp, "error", None)
                err_msg = str(error) if error else "Unknown batch error"
                results.append((f"[BATCH_ERROR: {err_msg}]", None))
    else:
        raise RuntimeError(f"No inlined responses found for batch job {job_name}")

    return results


def call_llm_batch(
    prompts: list[str],
    *,
    display_name: str = "crystal-benchmark",
    poll_interval: float = 10.0,
    timeout: float = 3600.0,
) -> list[tuple[str, dict | None]]:
    """Submit prompts as a batch and wait for results.

    Convenience wrapper around submit_batch + poll_batch.
    Falls back to sequential calls for non-Gemini providers.
    """
    if LLM_PROVIDER != "gemini":
        results = []
        for i, p in enumerate(prompts):
            print(f"    [seq {i+1}/{len(prompts)}]")
            results.append(call_llm(p))
            if i < len(prompts) - 1:
                time.sleep(1.0)
        return results

    print(f"    Submitting batch of {len(prompts)} prompts...")
    job_name = submit_batch(prompts, display_name=display_name)
    print(f"    Batch job: {job_name}")
    return poll_batch(job_name, poll_interval=poll_interval, timeout=timeout)
