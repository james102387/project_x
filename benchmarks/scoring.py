"""Backward-compatible re-export — use benchmarks.scoring.binary instead."""

from benchmarks.scoring.binary import score_response, score_batch, score_batch_rubric

__all__ = ["score_response", "score_batch", "score_batch_rubric"]
