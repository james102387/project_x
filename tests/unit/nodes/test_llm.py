"""Unit tests for LLM client and node helpers."""

import pytest
from unittest.mock import MagicMock
from crystal.llm import _extract_usage
from crystal.nodes.llm_nodes import _update_metrics_from_usage


class TestExtractUsage:
    def test_full_usage_with_reasoning(self):
        response = MagicMock()
        response.usage_metadata = MagicMock(
            prompt_token_count=50,
            candidates_token_count=30,
            thoughts_token_count=200,
        )
        usage = _extract_usage(response)
        assert usage["prompt_tokens"] == 50
        assert usage["output_tokens"] == 30
        assert usage["reasoning_tokens"] == 200
        assert usage["total_tokens"] == 280

    def test_usage_without_reasoning(self):
        um = MagicMock(spec=["prompt_token_count", "candidates_token_count"])
        um.prompt_token_count = 40
        um.candidates_token_count = 20
        response = MagicMock()
        response.usage_metadata = um
        usage = _extract_usage(response)
        assert usage["prompt_tokens"] == 40
        assert usage["output_tokens"] == 20
        assert usage["reasoning_tokens"] is None
        assert usage["total_tokens"] == 60

    def test_no_usage_metadata(self):
        response = MagicMock(spec=[])
        usage = _extract_usage(response)
        assert usage is None

    def test_none_usage_metadata(self):
        response = MagicMock()
        response.usage_metadata = None
        usage = _extract_usage(response)
        assert usage is None


class TestUpdateMetricsFromUsage:
    def test_merges_all_fields(self):
        metrics = {"prompt_type": "kg_augmented"}
        usage = {
            "prompt_tokens": 50,
            "output_tokens": 30,
            "reasoning_tokens": 200,
            "total_tokens": 280,
        }
        result = _update_metrics_from_usage(metrics, usage)
        assert result["actual_prompt_tokens"] == 50
        assert result["actual_output_tokens"] == 30
        assert result["actual_reasoning_tokens"] == 200
        assert result["actual_total_tokens"] == 280
        assert result["prompt_type"] == "kg_augmented"

    def test_no_usage_returns_unchanged(self):
        metrics = {"prompt_type": "no_math"}
        result = _update_metrics_from_usage(metrics, None)
        assert result == {"prompt_type": "no_math"}
        assert "actual_reasoning_tokens" not in result

    def test_partial_usage(self):
        metrics = {}
        usage = {"prompt_tokens": 50, "output_tokens": 30}
        result = _update_metrics_from_usage(metrics, usage)
        assert result["actual_prompt_tokens"] == 50
        assert result["actual_output_tokens"] == 30
        assert result.get("actual_reasoning_tokens") is None
