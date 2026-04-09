"""Backward-compatible re-export — use benchmarks.scoring.fitness instead."""

from benchmarks.scoring.fitness import binary_correct, fitness_score, evaluate_cases

__all__ = ["binary_correct", "fitness_score", "evaluate_cases"]
