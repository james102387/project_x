"""
Integration tests that exercise the full LangGraph pipeline including LLM.

All tests are marked @pytest.mark.llm and skipped by default.
Run with:  pytest --run-llm tests/integration/test_llm.py -v

Uses the cached_llm fixture so only the first run hits the real API;
subsequent runs replay from tests/fixtures/llm_cache.json.
"""

import pytest

from crystal.graph import build_crystal_graph
from crystal.state import make_initial_state


@pytest.fixture(autouse=True)
def _use_cache(cached_llm):
    """Activate the caching LLM wrapper for every test in this module."""


@pytest.mark.llm
class TestAugmentedPath:
    """math_augmented prompts — tool computes, LLM reasons about the result."""

    def test_advisory_question(self):
        app = build_crystal_graph()
        result = app.invoke(
            make_initial_state("She earned 500 and spent 300, is she managing her money wisely?")
        )
        assert result["prompt_type"] == "math_augmented"
        assert len(result["final_response"]) > 0


@pytest.mark.llm
class TestAnswerablePath:
    """math_answerable prompts — tool computes, LLM is bypassed."""

    def test_word_problem_direct_return(self):
        app = build_crystal_graph()
        result = app.invoke(make_initial_state("John has 10 apples and buys 5 more"))
        assert result["prompt_type"] == "math_answerable"
        assert result["final_response"] == "15"
        assert result["llm_response"] == ""

    def test_multi_step_direct_return(self):
        app = build_crystal_graph()
        result = app.invoke(
            make_initial_state("I had 100 dollars, earned 50, and spent 30")
        )
        assert result["prompt_type"] == "math_answerable"
        assert result["final_response"] == "120"
        assert result["llm_response"] == ""


@pytest.mark.llm
class TestFallbackPath:
    """no_math prompts — LLM handles everything."""

    def test_general_question(self):
        app = build_crystal_graph()
        result = app.invoke(make_initial_state("What is the capital of France?"))
        assert result["prompt_type"] == "no_math"
        assert len(result["final_response"]) > 0


@pytest.mark.llm
class TestDirectReturnSkipsLLM:
    """pure_math prompts — LLM should NOT be called."""

    def test_pure_addition(self):
        app = build_crystal_graph()
        result = app.invoke(make_initial_state("5 plus 3"))
        assert result["prompt_type"] == "pure_math"
        assert result["final_response"] == "8"
        assert result["llm_response"] == ""
