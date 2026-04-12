"""Backward-compatible re-export — use benchmarks.scoring.rubric instead."""

from benchmarks.scoring.rubric import (
    ABSTENTION_PHRASES,
    RubricResult,
    accuracy_score,
    calibration_score,
    is_abstention,
    score_rubric,
)

__all__ = [
    "ABSTENTION_PHRASES",
    "RubricResult",
    "accuracy_score",
    "calibration_score",
    "is_abstention",
    "score_rubric",
]
