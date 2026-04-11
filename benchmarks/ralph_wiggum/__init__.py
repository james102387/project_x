"""Ralph Wiggum v3 — multi-loop self-improvement engine.

Each loop owns exactly one failure category and one code area:
  - PredicateLoop:  predicate_mismatch → QUESTION_PREDICATE_MAP + LEGAL_PREDICATE_ALIASES
  - EntityLoop:     entity_mismatch   → entity alias tables
  - ThresholdLoop:  routing_error     → CONFIDENCE_LOW/HIGH numeric thresholds

Shared infrastructure lives in BaseLoop: evaluation, diagnosis, scoring,
reporting, git operations.

The Orchestrator runs all loops in sequence with shared evaluation state
and produces a unified change report.
"""

from benchmarks.ralph_wiggum.base import (
    BaseLoop,
    IterationResult,
    LoopResult,
    FailureCategory,
    diagnose_failure,
    call_ralph_llm,
    parse_llm_proposal,
    insert_dict_entries,
    remove_dict_entries,
    build_change_report,
    run_full_pipeline,
    default_mock_llm,
    git_cmd,
    RALPH_MODEL,
)
from benchmarks.ralph_wiggum.predicate_loop import PredicateLoop
from benchmarks.ralph_wiggum.entity_loop import EntityLoop
from benchmarks.ralph_wiggum.threshold_loop import ThresholdLoop, _update_threshold
from benchmarks.ralph_wiggum.orchestrator import Orchestrator, OrchestratorResult

# Backward-compatible aliases (v2 names → v3 locations)
RalphWiggumLoop = PredicateLoop
_parse_llm_proposal = parse_llm_proposal
_build_change_report = build_change_report
_run_full_pipeline = run_full_pipeline
_default_mock_llm = default_mock_llm
_PLANNER_FILE = ThresholdLoop.TARGET_FILES[0] if ThresholdLoop.TARGET_FILES else None
from benchmarks.ralph_wiggum.predicate_loop import (
    _PREDICATE_MAP_FILE,
    _ALIAS_FILE,
)


def _apply_proposal(
    proposal: dict,
    *,
    predicate_map_file=None,
    alias_file=None,
) -> dict[str, int]:
    """Backward-compatible apply that handles all mutation types."""
    pm_file = predicate_map_file or _PREDICATE_MAP_FILE
    al_file = alias_file or _ALIAS_FILE
    counts: dict[str, int] = {}

    if "predicate_map" in proposal and proposal["predicate_map"]:
        counts["predicate_map"] = insert_dict_entries(
            pm_file, "QUESTION_PREDICATE_MAP", proposal["predicate_map"],
        )
    if "predicate_aliases" in proposal and proposal["predicate_aliases"]:
        counts["predicate_aliases"] = insert_dict_entries(
            al_file, "LEGAL_PREDICATE_ALIASES", proposal["predicate_aliases"],
        )
    if "entity_aliases" in proposal and proposal["entity_aliases"]:
        counts["entity_aliases"] = len(proposal["entity_aliases"])
    threshold = proposal.get("confidence_threshold")
    if threshold is not None:
        if _update_threshold(threshold):
            counts["confidence_threshold"] = 1
    return counts


def _validate_proposal(proposal: dict | None) -> bool:
    """Backward-compatible unified validator (accepts all loop formats)."""
    if not proposal or not isinstance(proposal, dict):
        return False

    allowed_sections = {
        "predicate_map", "predicate_aliases", "entity_aliases",
        "confidence_threshold",
    }
    if not any(k in allowed_sections for k in proposal):
        return False

    for key in proposal:
        if key not in allowed_sections:
            return False
        if key == "confidence_threshold":
            from benchmarks.ralph_wiggum.threshold_loop import MIN_THRESHOLD, MAX_THRESHOLD
            val = proposal[key]
            if val is not None:
                if not isinstance(val, (int, float)):
                    return False
                if val < MIN_THRESHOLD or val > MAX_THRESHOLD:
                    return False
            continue
        section = proposal[key]
        if not isinstance(section, dict):
            return False
        for k, v in section.items():
            if not isinstance(k, str) or not isinstance(v, str):
                return False
            if not k.strip() or not v.strip():
                return False
    return True


__all__ = [
    "BaseLoop",
    "IterationResult",
    "LoopResult",
    "FailureCategory",
    "diagnose_failure",
    "PredicateLoop",
    "EntityLoop",
    "ThresholdLoop",
    "Orchestrator",
    "OrchestratorResult",
    # backward compat
    "RalphWiggumLoop",
    "_validate_proposal",
    "_parse_llm_proposal",
    "_build_change_report",
    "_update_threshold",
    "_run_full_pipeline",
    "_default_mock_llm",
    "_PLANNER_FILE",
    "_apply_proposal",
    "_PREDICATE_MAP_FILE",
    "_ALIAS_FILE",
]
