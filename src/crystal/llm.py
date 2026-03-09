"""LLM client wrapper. Swap providers by changing this file only."""

import os
import time

LLM_MODEL = "gemini-2.0-flash"

_llm_client = None


def _get_client():
    """Lazy-init the Gemini client so imports work without an API key."""
    global _llm_client
    if _llm_client is None:
        from google import genai
        _llm_client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    return _llm_client


def call_llm(prompt: str, max_retries: int = 3) -> tuple[str, dict | None]:
    """
    Call the LLM with retry logic for rate limits.

    Returns (response_text, usage_dict) where usage_dict contains
    prompt_tokens and output_tokens from the API response, or None
    if usage metadata is unavailable.
    """
    client = _get_client()
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
            )
            usage = None
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                um = response.usage_metadata
                usage = {
                    "prompt_tokens": getattr(um, "prompt_token_count", None),
                    "output_tokens": getattr(um, "candidates_token_count", None),
                }
            return response.text.strip(), usage
        except Exception as e:
            if "429" in str(e) or "TooManyRequests" in str(e):
                wait = 2 ** attempt * 5
                print(
                    f"    Rate limited, waiting {wait}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(wait)
            else:
                raise
    return "[ERROR: Rate limit exceeded after retries]", None
