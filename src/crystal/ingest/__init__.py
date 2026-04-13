"""
Crystal KG ingestion pipeline.

Entry points:
    ingest(path)                — auto-detect format, return IngestResult
    ingest_text(text)           — NER extraction from raw text
    ingest_with_llm(path)       — two-pass: NER first, LLM for gaps (D2 Phase 2)
    ingest_document(...)        — full orchestrator: NER + LLM + scoring + auto-accept
    build_kg(result)            — convert IngestResult to KnowledgeGraph

CLI:
    python -m crystal.ingest <document>
    python -m crystal.ingest <document> --llm-assist
    python -m crystal.ingest data.csv
    python -m crystal.ingest facts.json
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from crystal.ingest.schema import (
    IngestResult,
    LLMExtractionResult,
    ReviewableTriplet,
    Triplet,
)
from crystal.ingest.ner import (
    extract_triplets,
    find_unresolved_sentences,
    ingest_file,
    ingest_text,
)
from crystal.ingest.loader import load_csv, load_json, load_file, load_review
from crystal.ingest.llm_extract import extract_triplets_llm, normalize_predicate
from crystal.ingest.confidence import (
    INGEST_AUTO_ACCEPT,
    ScoredTriplet,
    classify_extraction_source,
    score_ingestion_confidence,
)
from crystal.ingest.validation import validate_triplet
from crystal.tools.kg.graph import KnowledgeGraph

logger = logging.getLogger(__name__)


def ingest(path: str | Path) -> IngestResult:
    """Auto-detect file type and ingest into an IngestResult.

    .csv / .json → hand-curated loader (exact triplets)
    .txt / other → NER extraction from text
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".csv", ".json"):
        return load_file(path)
    else:
        return ingest_file(str(path))


def ingest_with_llm(
    path: str | Path,
    call_llm_fn: Callable[[str], tuple[str, dict | None]] | None = None,
) -> tuple[IngestResult, LLMExtractionResult]:
    """Two-pass ingestion: NER extraction first, then LLM for unresolved sentences.

    Pass 1: Standard NER dep-tree extraction (same as ingest()).
    Pass 2: Sentences where NER found entities but no predicates are sent
             to the LLM for relationship extraction.

    Returns (ner_result, llm_result). The llm_result contains reviewable
    triplets that need human approval before being added to the KG.
    Only works on text files — CSV/JSON bypass NER entirely.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    ner_result = ingest_text(text, source=str(path))

    unresolved = find_unresolved_sentences(text)
    llm_result = extract_triplets_llm(
        unresolved,
        call_llm_fn=call_llm_fn,
    )
    llm_result.source = str(path)

    return ner_result, llm_result


def build_kg(result: IngestResult) -> KnowledgeGraph:
    """Convert an IngestResult into a ready-to-query KnowledgeGraph."""
    return KnowledgeGraph(
        triplets=result.as_tuples(),
        predicate_aliases=result.predicate_aliases or None,
        entity_aliases=result.entity_aliases or None,
    )


# ── Document ingestion orchestrator ──────────────────────────────────


@dataclass
class DocumentIngestionResult:
    """Result of a full document ingestion run with progressive trust."""

    auto_accepted: list[ScoredTriplet] = field(default_factory=list)
    pending_review: list[ScoredTriplet] = field(default_factory=list)
    rejected: list[ScoredTriplet] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    _kg = None
    _source: str = ""

    def accept_pending(self, indices: list[int]) -> int:
        """Accept specific pending triplets and insert them into the KG."""
        accepted = 0
        to_remove = []
        for idx in sorted(indices, reverse=True):
            if 0 <= idx < len(self.pending_review):
                st = self.pending_review[idx]
                st.status = "accepted"
                self.auto_accepted.append(st)
                to_remove.append(idx)
                accepted += 1
        for idx in to_remove:
            self.pending_review.pop(idx)

        if accepted > 0 and self._kg is not None:
            new_triplets = [
                self.auto_accepted[-(accepted - i)].as_tuple()
                for i in range(accepted)
            ]
            self._kg.bulk_insert(new_triplets, source=self._source)

        return accepted

    def accept_all_pending(self) -> int:
        return self.accept_pending(list(range(len(self.pending_review))))

    def reject_pending(self, indices: list[int]) -> int:
        rejected = 0
        for idx in sorted(indices, reverse=True):
            if 0 <= idx < len(self.pending_review):
                st = self.pending_review[idx]
                st.status = "rejected"
                self.rejected.append(st)
                self.pending_review.pop(idx)
                rejected += 1
        return rejected


def _post_insert_validate(accepted: list[ScoredTriplet]) -> None:
    """Safety net: re-validate just-inserted triplets and warn on failures.

    In normal flow this should never fire (we validate before insertion),
    but catches cases where bulk_insert is called directly or validation
    is bypassed.
    """
    hard_fails = 0
    for st in accepted:
        vr = validate_triplet(st.subject, st.predicate, st.object)
        if not vr.valid:
            sev = vr.severity.value if vr.severity else "unknown"
            reason_str = "; ".join(vr.reasons) if vr.reasons else "unknown"
            logger.warning(
                "Post-insert validation failure (%s): (%s, %s, %s) — %s",
                sev, st.subject, st.predicate, st.object[:80], reason_str,
            )
            if vr.severity and vr.severity.value == "hard":
                hard_fails += 1
    if hard_fails:
        logger.error(
            "POST-INSERT ALERT: %d hard validation failures in batch of %d. "
            "KG may contain invalid facts.",
            hard_fails, len(accepted),
        )


def ingest_document(
    path_or_text: str | Path,
    kg=None,
    *,
    call_llm_fn: Callable[[str], tuple[str, dict | None]] | None = None,
    auto_accept_threshold: float = INGEST_AUTO_ACCEPT,
    domain: str = "legal",
    ontology_predicates: set[str] | None = None,
    predicate_aliases: dict[str, str] | None = None,
) -> DocumentIngestionResult:
    """Full document ingestion: NER + LLM + scoring + auto-accept + KG insert.

    Args:
        path_or_text: File path or raw text string.
        kg: KG to insert auto-accepted triplets into (must support bulk_insert).
        call_llm_fn: LLM caller for extraction. If None, LLM pass is skipped.
        auto_accept_threshold: Confidence threshold for auto-accepting.
        domain: "legal" for legal-tuned extraction, "general" for generic.
        ontology_predicates: Set of canonical predicate names for scoring.
        predicate_aliases: Mapping of surface forms to canonical predicates.

    Returns:
        DocumentIngestionResult with auto_accepted, pending_review, rejected lists.
    """
    if ontology_predicates is None and domain == "legal":
        from crystal.data.legal_ontology import LEGAL_PREDICATES, LEGAL_PREDICATE_ALIASES
        ontology_predicates = set(LEGAL_PREDICATES)
        if predicate_aliases is None:
            predicate_aliases = dict(LEGAL_PREDICATE_ALIASES)

    start_time = time.time()

    path_obj = None
    if isinstance(path_or_text, Path):
        path_obj = path_or_text
    elif isinstance(path_or_text, str) and len(path_or_text) < 260 and "\n" not in path_or_text:
        candidate = Path(path_or_text)
        try:
            if candidate.exists():
                path_obj = candidate
        except OSError:
            pass

    if path_obj is not None:
        text = path_obj.read_text(encoding="utf-8")
        source = path_obj.name
    else:
        text = str(path_or_text)
        source = "pasted_text"

    ner_result = ingest_text(text, source=source)

    scored: list[ScoredTriplet] = []
    validation_rejected: list[ScoredTriplet] = []

    for triplet in ner_result.triplets:
        norm_pred = normalize_predicate(
            triplet.predicate, ontology_predicates, predicate_aliases,
        )
        vr = validate_triplet(triplet.subject, norm_pred, triplet.object)
        if not vr.valid:
            validation_rejected.append(ScoredTriplet(
                subject=triplet.subject,
                predicate=norm_pred,
                object=triplet.object,
                source_sentence=triplet.source_sentence,
                extraction_source="ner",
                ingestion_confidence=0.0,
                status="rejected",
            ))
            continue
        conf = score_ingestion_confidence(
            triplet.subject, norm_pred, "ner",
            kg=kg, ontology_predicates=ontology_predicates,
            predicate_aliases=predicate_aliases,
        )
        scored.append(ScoredTriplet(
            subject=triplet.subject,
            predicate=norm_pred,
            object=triplet.object,
            source_sentence=triplet.source_sentence,
            extraction_source="ner",
            ingestion_confidence=conf,
        ))

    if call_llm_fn is not None:
        unresolved = find_unresolved_sentences(text)
        if unresolved:
            llm_result = extract_triplets_llm(
                unresolved, call_llm_fn=call_llm_fn, domain=domain,
            )
            for rt in llm_result.reviewable:
                ext_source = classify_extraction_source(rt.confidence)
                norm_pred = normalize_predicate(
                    rt.predicate, ontology_predicates, predicate_aliases,
                )
                vr = validate_triplet(rt.subject, norm_pred, rt.object)
                if not vr.valid:
                    validation_rejected.append(ScoredTriplet(
                        subject=rt.subject,
                        predicate=norm_pred,
                        object=rt.object,
                        source_sentence=rt.source_sentence,
                        extraction_source=ext_source,
                        ingestion_confidence=0.0,
                        status="rejected",
                    ))
                    continue
                conf = score_ingestion_confidence(
                    rt.subject, norm_pred, ext_source,
                    kg=kg, ontology_predicates=ontology_predicates,
                    predicate_aliases=predicate_aliases,
                )
                scored.append(ScoredTriplet(
                    subject=rt.subject,
                    predicate=norm_pred,
                    object=rt.object,
                    source_sentence=rt.source_sentence,
                    extraction_source=ext_source,
                    ingestion_confidence=conf,
                ))

    seen = set()
    auto_accepted = []
    pending_review = []
    rejected = []

    for st in scored:
        key = (st.subject.lower(), st.predicate.lower(), st.object.lower())
        if key in seen:
            continue
        seen.add(key)

        if kg is not None:
            existing = kg.lookup(subject=st.subject, predicate=st.predicate)
            if any(e["object"].lower() == st.object.lower() for e in existing):
                continue

        if st.ingestion_confidence >= auto_accept_threshold:
            st.status = "accepted"
            auto_accepted.append(st)
        elif st.ingestion_confidence >= 0.25:
            st.status = "pending_review"
            pending_review.append(st)
        else:
            st.status = "rejected"
            rejected.append(st)

    rejected.extend(validation_rejected)

    if auto_accepted and kg is not None:
        try:
            kg.bulk_insert(
                [st.as_tuple_with_sentence() for st in auto_accepted],
                source=source,
            )
            _post_insert_validate(auto_accepted)
        except Exception:
            logger.exception("Failed to insert auto-accepted triplets into KG")

    elapsed = time.time() - start_time

    result = DocumentIngestionResult(
        auto_accepted=auto_accepted,
        pending_review=pending_review,
        rejected=rejected,
        stats={
            "source": source,
            "total_extracted": len(scored) + len(validation_rejected),
            "validation_rejected": len(validation_rejected),
            "auto_accepted": len(auto_accepted),
            "pending_review": len(pending_review),
            "rejected": len(rejected),
            "deduped": len(scored) - len(auto_accepted) - len(pending_review) - (len(rejected) - len(validation_rejected)),
            "ner_triplets": len(ner_result.triplets),
            "llm_triplets": max(0, len(scored) + len(validation_rejected) - len(ner_result.triplets)),
            "elapsed_seconds": round(elapsed, 2),
        },
    )
    result._kg = kg
    result._source = source

    logger.info(
        "Ingested %s: %d extracted, %d auto-accepted, %d pending, %d rejected (%.1fs)",
        source, len(scored), len(auto_accepted), len(pending_review),
        len(rejected), elapsed,
    )

    return result
