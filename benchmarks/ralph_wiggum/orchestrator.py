"""Orchestrator — runs all specialized loops in sequence.

Performs a single shared evaluation, then feeds each loop only its
relevant failures. Produces a unified change report.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from benchmarks.ralph_wiggum.base import (
    BaseLoop,
    IterationResult,
    LoopResult,
    build_change_report,
)
from benchmarks.ralph_wiggum.predicate_loop import PredicateLoop
from benchmarks.ralph_wiggum.entity_loop import EntityLoop
from benchmarks.ralph_wiggum.threshold_loop import ThresholdLoop
from benchmarks.ralph_wiggum.extraction_loop import ExtractionLoop

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResult:
    overall_score: float
    loop_results: dict[str, LoopResult] = field(default_factory=dict)
    unified_report: str = ""


class Orchestrator:
    """Coordinates all Ralph Wiggum loops with shared evaluation."""

    LOOP_CLASSES: list[type[BaseLoop]] = [PredicateLoop, EntityLoop, ThresholdLoop]

    def __init__(
        self,
        kg,
        cases: list[tuple[str, str, list[str], bool]],
        nlp=None,
        call_llm_fn: Callable[[str], tuple[str, dict | None]] | None = None,
        use_git: bool = False,
        use_full_pipeline: bool = True,
        mock_llm_fn=None,
    ) -> None:
        self.kg = kg
        self.cases = cases
        self.nlp = nlp
        self.call_llm_fn = call_llm_fn
        self.use_git = use_git
        self.use_full_pipeline = use_full_pipeline
        self.mock_llm_fn = mock_llm_fn

    def run(
        self,
        threshold: float = 0.90,
        max_iterations_per_loop: int = 10,
    ) -> OrchestratorResult:
        loop_results: dict[str, LoopResult] = {}

        for loop_cls in self.LOOP_CLASSES:
            loop = loop_cls(
                kg=self.kg,
                cases=self.cases,
                nlp=self.nlp,
                call_llm_fn=self.call_llm_fn,
                use_git=self.use_git,
                use_full_pipeline=self.use_full_pipeline,
                mock_llm_fn=self.mock_llm_fn,
            )

            logger.info("Running %s...", loop.LOOP_NAME)
            result = loop.run(
                threshold=threshold,
                max_iterations=max_iterations_per_loop,
            )
            loop_results[loop.LOOP_NAME] = result
            logger.info(
                "%s finished: %.1f%% (%d iterations)",
                loop.LOOP_NAME, result.final_score * 100, result.iterations_run,
            )

        overall_score = 0.0
        if loop_results:
            last_result = list(loop_results.values())[-1]
            overall_score = last_result.final_score

        report = self._build_unified_report(loop_results)

        return OrchestratorResult(
            overall_score=overall_score,
            loop_results=loop_results,
            unified_report=report,
        )

    def _build_unified_report(self, loop_results: dict[str, LoopResult]) -> str:
        lines = ["# Ralph Wiggum v3 — Unified Report\n"]
        lines.append("## Loop Summary\n")

        for name, result in loop_results.items():
            status = "converged" if result.converged else "did not converge"
            lines.append(
                f"- **{name}**: {result.final_score:.1%} "
                f"({result.iterations_run} iterations, {status})"
            )

        lines.append("")

        for name, result in loop_results.items():
            if result.change_report:
                lines.append(f"---\n")
                lines.append(result.change_report)

        return "\n".join(lines)
