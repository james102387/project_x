"""
Extraction quality benchmark: can Crystal learn from documents and answer questions?

Runs LLM extraction on cached SCOTUS opinions, builds an ephemeral KG from
ONLY the extracted facts (no CourtListener scaffold), then tests benchmark
questions against the document-only KG.

This directly measures the document-to-KG-to-answer pipeline.

Usage:
    PYTHONPATH=src:. python -m benchmarks.extraction_quality
    PYTHONPATH=src:. python -m benchmarks.extraction_quality --ner-only
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from benchmarks.documents import load_all_opinions
from benchmarks.ground_truth.legal import LEGAL_BENCHMARK_CASES
from benchmarks.scoring.binary import score_batch_rubric
from crystal.data.legal_ontology import normalize_case_name
from crystal.tools.kg.store import SqliteKnowledgeGraph

RESULTS_DIR = Path(__file__).parent / "results"

EXTRACTION_CASES: list[dict] = [
    {"slug": "brown-v-board-of-education", "case_name": "Brown v. Board of Education", "min_chars": 5000},
    {"slug": "gideon-v-wainwright", "case_name": "Gideon v. Wainwright", "min_chars": 5000},
    {"slug": "loving-v-virginia", "case_name": "Loving v. Virginia", "min_chars": 5000},
    {"slug": "terry-v-ohio", "case_name": "Terry v. Ohio", "min_chars": 5000},
    {"slug": "plessy-v-ferguson", "case_name": "Plessy v. Ferguson", "min_chars": 5000},
    {"slug": "marbury-v-madison", "case_name": "Marbury v. Madison", "min_chars": 5000},
    {"slug": "citizens-united-v-federal-election-commission", "case_name": "Citizens United v. Federal Election Commission", "min_chars": 5000},
    {"slug": "dred-scott-v-sandford", "case_name": "Dred Scott v. Sandford", "min_chars": 5000},
    {"slug": "mcculloch-v-maryland", "case_name": "McCulloch v. Maryland", "min_chars": 5000},
    {"slug": "coventry-health-care-of-mo-inc-v-nevils", "case_name": "Coventry Health Care of Mo., Inc. v. Nevils", "min_chars": 5000},
    {"slug": "pennsylvania-railroad-v-st-louis-alton-terre-haute-railroad", "case_name": "Pennsylvania Railroad v. St. Louis, Alton & Terre Haute Railroad", "min_chars": 5000},
    {"slug": "flanders-v-seelye", "case_name": "Flanders v. Seelye", "min_chars": 5000},
    {"slug": "union-insurance-v-hoge", "case_name": "Union Insurance v. Hoge", "min_chars": 5000},
    {"slug": "zubik-v-burwell", "case_name": "Zubik v. Burwell", "min_chars": 5000},
    {"slug": "sturgis-v-clough", "case_name": "Sturgis v. Clough", "min_chars": 5000},
]


def _questions_for_cases(case_names: set[str]) -> list[tuple]:
    """Filter benchmark questions to those targeting the given case names."""
    import re

    matched = []
    for q, gt, ms, neg in LEGAL_BENCHMARK_CASES:
        if neg:
            continue
        m = re.search(
            r"(?:in |for |decided |heard |filed )(.+?)(?:\?|$)", q, re.IGNORECASE
        )
        if m:
            name = normalize_case_name(m.group(1).strip().rstrip("?")).lower()
            if name in case_names:
                matched.append((q, gt, ms, neg))
    return matched


def run_extraction_benchmark(
    *,
    ner_only: bool = False,
    call_llm_fn=None,
    sleep_between: float = 4.0,
) -> dict:
    """Run the full extraction quality benchmark.

    1. Load opinions for the target cases
    2. Run ingest_document() on each (NER-only or NER+LLM)
    3. Build an ephemeral KG from extracted facts
    4. Run Crystal pipeline against this KG
    5. Score and report
    """
    from crystal.ingest import ingest_document
    from crystal.graph import build_crystal_graph
    from crystal.state import make_initial_state

    if call_llm_fn is None and not ner_only:
        from crystal.llm import call_llm
        call_llm_fn = call_llm

    opinions = load_all_opinions()

    print(f"\n{'=' * 60}")
    print(f"  EXTRACTION QUALITY BENCHMARK")
    print(f"  Mode: {'NER-only' if ner_only else 'NER + LLM'}")
    print(f"{'=' * 60}")

    extraction_kg = SqliteKnowledgeGraph()
    ingestion_stats = []
    case_name_set = set()

    for case_info in EXTRACTION_CASES:
        case_name = case_info["case_name"]
        key = normalize_case_name(case_name).lower()
        text = opinions.get(key)

        if not text or len(text) < case_info["min_chars"]:
            print(f"  SKIP: {case_name} (text too short or missing)")
            continue

        print(f"\n  Ingesting: {case_name} ({len(text):,} chars)")
        case_name_set.add(key)

        result = ingest_document(
            text,
            kg=extraction_kg,
            call_llm_fn=call_llm_fn if not ner_only else None,
            auto_accept_threshold=0.50,
            domain="legal",
        )

        result.accept_all_pending()

        total = len(result.auto_accepted) + len(result.pending_review)
        ingestion_stats.append({
            "case_name": case_name,
            "chars": len(text),
            "total_extracted": result.stats.get("total_extracted", 0),
            "auto_accepted": result.stats.get("auto_accepted", 0),
            "pending_accepted": len(result.auto_accepted) - result.stats.get("auto_accepted", 0),
            "rejected": result.stats.get("rejected", 0),
            "ner_triplets": result.stats.get("ner_triplets", 0),
            "llm_triplets": result.stats.get("llm_triplets", 0),
        })

        if sleep_between > 0 and not ner_only:
            time.sleep(sleep_between)

    fact_count = len(extraction_kg)

    print(f"\n  --- Extraction Summary ---")
    print(f"  Cases processed: {len(ingestion_stats)}")
    print(f"  Total KG facts:  {fact_count}")

    for stat in ingestion_stats:
        print(f"    {stat['case_name']}: {stat['total_extracted']} extracted "
              f"({stat['ner_triplets']} NER, {stat['llm_triplets']} LLM)")

    questions = _questions_for_cases(case_name_set)
    print(f"\n  Benchmark questions matching extracted cases: {len(questions)}")

    if not questions:
        print("  No matching questions found. Cannot score.")
        return {"error": "no_matching_questions", "ingestion_stats": ingestion_stats}

    graph = build_crystal_graph()
    results = []

    for i, (question, ground_truth, match_strings, is_negative) in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {question[:70]}")
        try:
            state = make_initial_state(question, kg=extraction_kg)
            final = graph.invoke(state)
            response = final.get("final_response", "")
            prompt_type = final.get("prompt_type", "unknown")
            kg_results = final.get("kg_results", [])
            token_metrics = final.get("token_metrics", {})
        except Exception as e:
            response = f"[ERROR: {e}]"
            prompt_type = "error"
            kg_results = []
            token_metrics = {}

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "match_strings": match_strings,
            "is_negative": is_negative,
            "response": response,
            "prompt_type": prompt_type,
            "kg_results": kg_results,
            "token_metrics": token_metrics,
        })

        if sleep_between > 0:
            time.sleep(sleep_between)

    scored = score_batch_rubric(results)

    report = {
        "timestamp": datetime.now().isoformat(),
        "mode": "ner_only" if ner_only else "ner_llm",
        "cases_processed": len(ingestion_stats),
        "kg_fact_count": fact_count,
        "questions_tested": len(questions),
        "ingestion_stats": ingestion_stats,
        "scored": scored,
    }

    return report


def print_extraction_report(report: dict) -> None:
    """Print a human-readable extraction quality report."""
    scored = report.get("scored", {})

    print(f"\n{'=' * 60}")
    print(f"  EXTRACTION QUALITY RESULTS")
    print(f"  Mode: {report.get('mode', 'unknown')}")
    print(f"{'=' * 60}")
    print(f"  Cases ingested:    {report.get('cases_processed', 0)}")
    print(f"  KG facts built:    {report.get('kg_fact_count', 0)}")
    print(f"  Questions tested:  {report.get('questions_tested', 0)}")
    print()
    print(f"  Accuracy:          {scored.get('accuracy', 0):.1%}")
    print(f"  Hallucination:     {scored.get('hallucination_rate', 1.0):.1%}")
    print(f"  Correct:           {scored.get('correct', 0)}/{scored.get('total', 0)}")
    print()

    incorrect = [d for d in scored.get("details", []) if not d.get("correct")]
    if incorrect:
        print(f"  --- Incorrect ({len(incorrect)}) ---")
        for d in incorrect:
            resp = d.get("response", "")[:80].replace("\n", " ")
            print(f"    Q: {d['question']}")
            print(f"    Expected: {d['ground_truth']}")
            print(f"    Got: {resp}")
            print(f"    Route: {d.get('prompt_type', '?')}")
            print()

    correct = [d for d in scored.get("details", []) if d.get("correct")]
    if correct:
        print(f"  --- Correct ({len(correct)}) ---")
        for d in correct:
            resp = d.get("response", "")[:60].replace("\n", " ")
            print(f"    Q: {d['question'][:60]}")
            print(f"    Route: {d.get('prompt_type', '?')}")


def save_extraction_report(report: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = report.get("mode", "unknown")
    path = RESULTS_DIR / f"extraction_quality_{mode}_{ts}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="Extraction quality benchmark")
    parser.add_argument("--ner-only", action="store_true",
                        help="Skip LLM extraction, NER only")
    parser.add_argument("--sleep", type=float, default=4.0,
                        help="Sleep between LLM calls")
    args = parser.parse_args()

    report = run_extraction_benchmark(
        ner_only=args.ner_only,
        sleep_between=args.sleep,
    )

    print_extraction_report(report)
    path = save_extraction_report(report)
    print(f"\n  Report saved to {path}")


if __name__ == "__main__":
    main()
