"""Human-in-the-loop review consolidation.

Discovers all pending review items across the project:
  - Generated questions (question_gen.py output)
  - LLM-extracted triplet proposals
  - Legal known gaps (detector improvements needed)

All pending items are surfaced in a single place for the user.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

REVIEW_DIR = Path(__file__).parent.parent.parent / "review"


@dataclass
class ReviewItem:
    """A single item requiring human attention."""
    category: str
    source_file: str
    data: dict
    status: str = "pending_review"


@dataclass
class ReviewSummary:
    """Aggregated view of all pending review items."""
    pending_questions: int = 0
    accepted_questions: int = 0
    rejected_questions: int = 0
    pending_triplets: int = 0
    known_gaps: int = 0
    question_files: list[str] = field(default_factory=list)
    triplet_files: list[str] = field(default_factory=list)


def discover_review_files(review_dir: Path | None = None) -> ReviewSummary:
    """Scan the review directory for all pending items."""
    review_dir = review_dir or REVIEW_DIR
    summary = ReviewSummary()

    if not review_dir.exists():
        return summary

    for path in sorted(review_dir.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        if "cases" in data and isinstance(data.get("cases"), list):
            summary.question_files.append(str(path))
            for case in data["cases"]:
                status = case.get("status", "pending_review")
                if status == "pending_review":
                    summary.pending_questions += 1
                elif status == "accepted":
                    summary.accepted_questions += 1
                elif status == "rejected":
                    summary.rejected_questions += 1

        elif "reviewable" in data and isinstance(data.get("reviewable"), list):
            summary.triplet_files.append(str(path))
            for item in data["reviewable"]:
                if item.get("status") == "pending_review":
                    summary.pending_triplets += 1

    from benchmarks.ground_truth.legal import LEGAL_KNOWN_GAPS
    summary.known_gaps = len(LEGAL_KNOWN_GAPS)

    return summary


def load_pending_questions(review_dir: Path | None = None) -> list[dict]:
    """Load all pending questions across all review files."""
    review_dir = review_dir or REVIEW_DIR
    pending = []

    if not review_dir.exists():
        return pending

    for path in sorted(review_dir.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        if "cases" not in data:
            continue

        for case in data.get("cases", []):
            if case.get("status") == "pending_review":
                case["_source_file"] = str(path)
                pending.append(case)

    return pending


def load_pending_triplets(review_dir: Path | None = None) -> list[dict]:
    """Load all pending LLM-extracted triplets across review files."""
    review_dir = review_dir or REVIEW_DIR
    pending = []

    if not review_dir.exists():
        return pending

    for path in sorted(review_dir.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        if "reviewable" not in data:
            continue

        for item in data.get("reviewable", []):
            if item.get("status") == "pending_review":
                item["_source_file"] = str(path)
                pending.append(item)

    return pending


def load_known_gaps() -> list[dict]:
    """Load detector known gaps that need engineering attention."""
    from benchmarks.ground_truth.legal import LEGAL_KNOWN_GAPS
    return [
        {
            "question": q,
            "expected_answer": a,
            "match_strings": m,
            "reason": reason,
        }
        for q, a, m, reason in LEGAL_KNOWN_GAPS
    ]


def format_review_dashboard(review_dir: Path | None = None) -> str:
    """Format a human-readable dashboard of all pending review items."""
    summary = discover_review_files(review_dir)
    gaps = load_known_gaps()

    lines = []

    total_pending = summary.pending_questions + summary.pending_triplets
    if total_pending > 0:
        lines.append(f"**{total_pending} items need your review** — "
                      f"{summary.pending_questions} questions, {summary.pending_triplets} triplets\n")
    else:
        lines.append("**All caught up** — no pending review items.\n")

    lines.append(f"Questions: **{summary.pending_questions}** pending · "
                 f"{summary.accepted_questions} accepted · {summary.rejected_questions} rejected")

    if summary.pending_triplets > 0:
        lines.append(f"  \nTriplets: **{summary.pending_triplets}** pending LLM-extracted triplets")
    lines.append("")

    batches = list_batches(review_dir)
    if batches:
        total_accepted = sum(b["accepted"] for b in batches)
        lines.append(f"**{len(batches)}** batches · **{total_accepted}** accepted golden answers")
        if total_accepted >= 50:
            lines.append(
                f"  \n*Ready for Ralph Wiggum loop — {total_accepted} accepted cases.*"
            )
            lines.append("")

    return "\n".join(lines)


# ── Batch-aware functions ────────────────────────────────────────────────


def list_batches(review_dir: Path | None = None) -> list[dict]:
    """List all ingestion batches with metadata and status counts.

    Discovers both ``batch_*.json`` files (with batch metadata) and any other
    ``*.json`` files that contain a ``cases`` list (e.g. ``pending_questions.json``).
    """
    review_dir = review_dir or REVIEW_DIR
    batches = []

    if not review_dir.exists():
        return batches

    for path in sorted(review_dir.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        if "cases" not in data or not isinstance(data["cases"], list):
            continue

        batch_meta = data.get("batch", {})
        cases = data["cases"]
        pending = sum(1 for c in cases if c.get("status") == "pending_review")
        accepted = sum(1 for c in cases if c.get("status") == "accepted")
        rejected = sum(1 for c in cases if c.get("status") == "rejected")

        batches.append({
            "id": batch_meta.get("id", path.stem),
            "source": batch_meta.get("source", path.name),
            "records_ingested": batch_meta.get("records_ingested", len(cases)),
            "timestamp": batch_meta.get("timestamp", ""),
            "total_cases": len(cases),
            "pending": pending,
            "accepted": accepted,
            "rejected": rejected,
            "file": str(path),
        })

    return batches


def _resolve_batch_path(batch_id: str, review_dir: Path) -> Path | None:
    """Find the JSON file for a batch id, trying batch_ prefix then bare stem."""
    for candidate in [
        review_dir / f"batch_{batch_id}.json",
        review_dir / f"{batch_id}.json",
    ]:
        if candidate.exists():
            return candidate
    return None


def load_batch_questions(batch_id: str, review_dir: Path | None = None) -> list[dict]:
    """Load all questions for a specific batch."""
    review_dir = review_dir or REVIEW_DIR
    path = _resolve_batch_path(batch_id, review_dir)

    if path is None:
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    return data.get("cases", [])


def load_batch_context(batch_id: str, review_dir: Path | None = None) -> list[list[str]]:
    """Load the source triplets for a batch (the underlying data)."""
    review_dir = review_dir or REVIEW_DIR
    path = _resolve_batch_path(batch_id, review_dir)

    if path is None:
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    batch = data.get("batch", {})
    triplets = batch.get("triplets", [])

    if not triplets:
        for case in data.get("cases", []):
            st = case.get("source_triplet")
            if st and isinstance(st, list) and len(st) == 3:
                triplets.append(st)

    return triplets


def save_review_decisions(
    batch_id: str,
    decisions: dict[int, str],
    review_dir: Path | None = None,
) -> None:
    """Save accept/reject decisions for a batch.

    decisions: {case_index: "accepted" | "rejected"}
    """
    review_dir = review_dir or REVIEW_DIR
    path = _resolve_batch_path(batch_id, review_dir)
    if path is None:
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for idx, status in decisions.items():
        if 0 <= idx < len(data["cases"]):
            data["cases"][idx]["status"] = status

    data["pending"] = sum(1 for c in data["cases"] if c.get("status") == "pending_review")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def collect_accepted_cases(
    review_dir: Path | None = None,
    *,
    origin_filter: str | None = None,
    document_filter: str | None = None,
) -> list[tuple[str, str, list[str], bool]]:
    """Collect all accepted cases across all batches as benchmark tuples.

    Returns list of (question, golden_answer, match_strings, is_negative).
    Used by the Ralph Wiggum loop to gather its test corpus.

    Args:
        origin_filter: If set, only return cases with this origin value.
        document_filter: If set, only return cases from this source_document.
    """
    review_dir = review_dir or REVIEW_DIR
    accepted = []

    if not review_dir.exists():
        return accepted

    for path in sorted(review_dir.glob("batch_*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        for case in data.get("cases", []):
            if case.get("status") != "accepted":
                continue
            if origin_filter and case.get("origin", "unknown") != origin_filter:
                continue
            if document_filter and case.get("source_document", "") != document_filter:
                continue
            accepted.append((
                case["question"],
                case.get("golden_answer", ""),
                case.get("match_strings", []),
                case.get("is_negative", False),
            ))

    return accepted


def save_single_review_decision(
    batch_id: str,
    question_idx: int,
    golden_answer: str,
    status: str,
    review_dir: Path | None = None,
) -> bool:
    """Save golden answer and status for a single question.

    Updates the golden_answer text, re-derives match_strings, and sets status.
    Returns True on success.
    """
    review_dir = review_dir or REVIEW_DIR
    path = _resolve_batch_path(batch_id, review_dir)
    if path is None:
        return False

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases = data.get("cases", [])
    if not (0 <= question_idx < len(cases)):
        return False

    cases[question_idx]["golden_answer"] = golden_answer.strip()
    cases[question_idx]["match_strings"] = _derive_match_strings(golden_answer)
    cases[question_idx]["status"] = status

    data["pending"] = sum(1 for c in cases if c.get("status") == "pending_review")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return True


def _normalize_for_match(s: str) -> str:
    """Collapse a case name or slug to a canonical matching key."""
    import re
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]", " ", s.lower())).strip()


def _build_doc_index(docs_dir: Path) -> dict[str, Path]:
    """Build a map of normalized-key -> document path for fuzzy matching."""
    index: dict[str, Path] = {}
    if not docs_dir.exists():
        return index
    for p in docs_dir.glob("*.json"):
        key = _normalize_for_match(p.stem)
        index[key] = p
    return index


def find_document_for_entity(
    entity_name: str, docs_dir: Path | None = None,
) -> Path | None:
    """Find the document file matching a KG entity name.

    Tries exact slug match first, then normalized match, then prefix match
    (for truncated filenames).
    """
    if docs_dir is None:
        docs_dir = Path(__file__).parent.parent.parent / "benchmarks" / "documents"

    slug = (entity_name.lower()
            .replace(".", "").replace(",", "").replace("'", "")
            .replace("&", "").replace(" ", "-"))
    exact = docs_dir / f"{slug}.json"
    if exact.exists():
        return exact

    index = _build_doc_index(docs_dir)
    entity_key = _normalize_for_match(entity_name)
    if entity_key in index:
        return index[entity_key]

    for doc_key, path in index.items():
        if doc_key.startswith(entity_key[:40]) and len(entity_key) > 40:
            return path
        if entity_key.startswith(doc_key[:40]) and len(doc_key) > 40:
            return path

    return None


def load_source_document_text(slug: str, docs_dir: Path | None = None) -> str:
    """Load the opinion text for a document by its slug or entity name.

    Looks in ``benchmarks/documents/`` using normalized matching.
    Returns the text content, or an error message if not found.
    """
    if docs_dir is None:
        docs_dir = Path(__file__).parent.parent.parent / "benchmarks" / "documents"

    path = docs_dir / f"{slug}.json"
    if not path.exists():
        path = find_document_for_entity(slug, docs_dir)
    if path is None or not path.exists():
        return f"Document not found: {slug}"

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ("text", "plain_text", "opinion_text", "content"):
                if key in data and data[key]:
                    text = str(data[key])
                    if len(text) > 60_000:
                        return text[:60_000] + (
                            f"\n\n[...truncated at 60,000 characters, "
                            f"full document is {len(text):,} characters]"
                        )
                    return text
            if "opinions" in data and isinstance(data["opinions"], list):
                parts = [op["plain_text"] for op in data["opinions"]
                         if isinstance(op, dict) and op.get("plain_text")]
                if parts:
                    text = "\n\n".join(parts)
                    if len(text) > 60_000:
                        return text[:60_000] + "\n\n[...truncated]"
                    return text
        return "Could not extract text from document."
    except Exception as e:
        return f"Error loading document: {e}"


def find_batch_document_slugs(batch_id: str, review_dir: Path | None = None) -> list[str]:
    """Find document slugs relevant to a batch.

    Parses the batch source field for known slugs, then scans questions'
    source_triplet subjects to find additional matching documents.
    """
    review_dir = review_dir or REVIEW_DIR
    docs_dir = Path(__file__).parent.parent.parent / "benchmarks" / "documents"
    if not docs_dir.exists():
        return []

    path = _resolve_batch_path(batch_id, review_dir)
    if path is None:
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    slugs: set[str] = set()

    import re
    source = data.get("batch", {}).get("source", "")
    for part in source.split(","):
        part = part.strip()
        if not part or part.startswith("(+"):
            continue
        part = re.sub(r"\s*\(\+\d+ more\)\s*$", "", part).strip()
        if part:
            candidate = docs_dir / f"{part}.json"
            if candidate.exists():
                slugs.add(part)

    for case in data.get("cases", []):
        st = case.get("source_triplet", [])
        if st and len(st) >= 1:
            name = str(st[0])
            match = find_document_for_entity(name, docs_dir)
            if match is not None:
                slugs.add(match.stem)

    return sorted(slugs)


def save_proposed_as_batch(
    proposed_rows: list[dict],
    source: str = "document_extraction",
    review_dir: Path | None = None,
) -> Path | None:
    """Save Crystal's proposed Q&A pairs as a review batch for human verification.

    Each row should have: question, crystal_answer, route, confidence, expected,
    and optionally golden_answer (user-corrected) and source_triplet.

    The user edits golden_answer to provide ground truth before accepting.
    """
    review_dir = review_dir or REVIEW_DIR
    review_dir.mkdir(parents=True, exist_ok=True)

    if not proposed_rows:
        return None

    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc)
    batch_id = f"doc_{ts.strftime('%Y%m%d_%H%M%S')}"

    cases = []
    for row in proposed_rows:
        golden = row.get("golden_answer", "").strip()
        crystal_answer = row.get("crystal_answer", "")
        question = row.get("question", "")

        if not golden:
            golden = crystal_answer

        match_strings = _derive_match_strings(golden)

        status = "pending_review"
        if row.get("status") in ("accepted", "rejected"):
            status = row["status"]

        cases.append({
            "question": question,
            "golden_answer": golden,
            "match_strings": match_strings,
            "is_negative": False,
            "tier": 2,
            "status": status,
            "crystal_proposed": crystal_answer,
            "crystal_route": row.get("route", ""),
            "crystal_confidence": row.get("confidence", ""),
            "source_triplet": row.get("source_triplet", []),
            "source_sentence": row.get("source_sentence", ""),
            "origin": row.get("origin", "unknown"),
            "source_document": row.get("source_document", ""),
        })

    data = {
        "batch": {
            "id": batch_id,
            "source": source,
            "records_ingested": len(cases),
            "timestamp": ts.isoformat(),
            "type": "document_extraction",
        },
        "description": (
            f"Questions generated from document extraction ({source}). "
            "Crystal proposed answers; human must verify golden_answer."
        ),
        "total": len(cases),
        "pending": sum(1 for c in cases if c["status"] == "pending_review"),
        "cases": cases,
    }

    path = review_dir / f"batch_{batch_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return path


def revalidate_pending_questions(
    kg,
    review_dir: Path | None = None,
) -> dict:
    """Re-evaluate pending questions against current KG state.

    For each pending_review case with a source_triplet, checks:
      1. Whether the source triplet still exists in the KG.
      2. Whether the source triplet passes current validation rules.

    Cases that fail either check are auto-rejected. The batch files are
    updated in place and the pending count is recalculated.

    Returns a summary: {rejected: [...], total_checked, total_rejected}.
    """
    from crystal.ingest.validation import validate_triplet

    review_dir = review_dir or REVIEW_DIR
    if not review_dir.exists():
        return {"rejected": [], "total_checked": 0, "total_rejected": 0}

    rejected_details: list[dict] = []
    total_checked = 0

    for path in sorted(review_dir.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        cases = data.get("cases")
        if not cases or not isinstance(cases, list):
            continue

        changed = False
        for case in cases:
            if case.get("status") != "pending_review":
                continue

            st = case.get("source_triplet")
            if not st or not isinstance(st, list) or len(st) < 3:
                continue

            total_checked += 1
            subj, pred, obj = st[0], st[1], st[2]
            reason = None

            vr = validate_triplet(subj, pred, obj)
            if not vr.valid:
                reason = f"fails validation: {'; '.join(vr.reasons)}"

            if reason is None and kg is not None:
                hits = kg.lookup(subject=subj, predicate=pred)
                if not any(h["object"].lower() == obj.lower() for h in hits):
                    reason = "source triplet no longer in KG"

            if reason:
                case["status"] = "rejected"
                case["_rejection_reason"] = reason
                changed = True
                rejected_details.append({
                    "file": path.name,
                    "question": case.get("question", ""),
                    "source_triplet": st,
                    "reason": reason,
                })

        if changed:
            data["pending"] = sum(
                1 for c in cases if c.get("status") == "pending_review"
            )
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    return {
        "rejected": rejected_details,
        "total_checked": total_checked,
        "total_rejected": len(rejected_details),
    }


def _derive_match_strings(golden_answer: str) -> list[str]:
    """Derive match_strings from a golden answer for benchmark scoring."""
    if not golden_answer:
        return []
    answer = golden_answer.strip()
    strings = [answer.lower()]
    for sep in [",", ";", " and "]:
        if sep in answer:
            parts = [p.strip().lower() for p in answer.split(sep) if p.strip()]
            if len(parts) > 1:
                strings.extend(parts)
                break
    return list(dict.fromkeys(strings))
