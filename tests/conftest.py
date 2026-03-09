"""Shared test fixtures."""

import hashlib
import json

import pytest
import spacy

nlp = spacy.load("en_core_web_sm")

LLM_CACHE_PATH = __import__("pathlib").Path(__file__).parent / "fixtures" / "llm_cache.json"


# ── CLI option: --run-llm ──────────────────────────────────────────────────


def pytest_addoption(parser):
    parser.addoption(
        "--run-llm",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.llm (hits real LLM API)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-llm"):
        return
    skip_llm = pytest.mark.skip(reason="needs --run-llm flag to run")
    for item in items:
        if "llm" in item.keywords:
            item.add_marker(skip_llm)


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def parse():
    """Fixture that returns a function to parse text into a spaCy Doc."""
    def _parse(text: str):
        return nlp(text)
    return _parse


@pytest.fixture
def cached_llm(monkeypatch, request):
    """
    Swap call_llm for a caching wrapper.

    First run: calls Gemini, saves response to tests/fixtures/llm_cache.json.
    Subsequent runs: replays from cache, zero API calls.

    Delete the cache file to force a refresh.
    """
    cache = {}
    if LLM_CACHE_PATH.exists():
        cache = json.loads(LLM_CACHE_PATH.read_text())

    _real_call = None

    def _ensure_real_call():
        nonlocal _real_call
        if _real_call is None:
            from crystal.llm import call_llm
            _real_call = call_llm

    def cached_call(prompt: str, **kwargs) -> tuple[str, dict | None]:
        key = hashlib.sha256(prompt.encode()).hexdigest()
        if key in cache:
            return cache[key], None

        _ensure_real_call()
        response_text, usage = _real_call(prompt, **kwargs)
        cache[key] = response_text
        LLM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        LLM_CACHE_PATH.write_text(json.dumps(cache, indent=2))
        return response_text, usage

    monkeypatch.setattr("crystal.llm.call_llm", cached_call)
    return cached_call
