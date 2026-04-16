"""
Question generator — auto-generate test questions with golden answers from KG data.

Produces two tiers:
  Tier 1: Factual lookup — "What is the {predicate} of {subject}?"
  Tier 2: Relational traversal — "What cases cite {subject}?"

Plus auto-generated negative cases for abstention testing.

Output format matches benchmarks/ground_truth.py:
  (question, golden_answer, match_strings, is_negative)

Workflow:
  1. generate_all(kg) → list[QuestionCase]  (status="pending_review")
  2. export_for_review(cases, path) → writes JSON for human editing
  3. Human edits: changes status to "accepted"/"rejected", corrects golden_answer
  4. import_reviewed(path) → list[QuestionCase]  (only accepted cases)
  5. Accepted cases feed into the Ralph Wiggum loop
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from crystal.tools.kg.graph import KnowledgeGraph


@dataclass
class QuestionCase:
    """A single test question with golden answer.

    Status lifecycle:
      pending_review → accepted / rejected
    Only "accepted" cases are used as benchmark ground truth.
    """
    question: str
    golden_answer: str
    match_strings: list[str]
    is_negative: bool = False
    tier: int = 1
    source_triplet: tuple[str, str, str] | None = None
    status: str = "pending_review"

    def as_benchmark_tuple(self) -> tuple[str, str, list[str], bool]:
        """Convert to the benchmark format used by ground_truth.py."""
        return (self.question, self.golden_answer, self.match_strings, self.is_negative)

    def to_review_dict(self) -> dict:
        """Serialize for the human review JSON file."""
        d = {
            "question": self.question,
            "golden_answer": self.golden_answer,
            "match_strings": self.match_strings,
            "is_negative": self.is_negative,
            "tier": self.tier,
            "status": self.status,
        }
        if self.source_triplet:
            d["source_triplet"] = list(self.source_triplet)
        return d

    @classmethod
    def from_review_dict(cls, d: dict) -> QuestionCase:
        """Deserialize from a human review JSON file."""
        src = d.get("source_triplet")
        return cls(
            question=d["question"],
            golden_answer=d.get("golden_answer", ""),
            match_strings=d.get("match_strings", []),
            is_negative=d.get("is_negative", False),
            tier=d.get("tier", 1),
            source_triplet=tuple(src) if src else None,
            status=d.get("status", "pending_review"),
        )


# ── Intake bay: export / import for human review ────────────────────────


def export_for_review(cases: list[QuestionCase], path: str | Path) -> int:
    """Write generated questions to a JSON file for human review.

    The reviewer should:
      - Change "status" from "pending_review" to "accepted" or "rejected"
      - Correct "golden_answer" and "match_strings" if needed
      - Add notes in an optional "notes" field (ignored on import)

    Returns number of cases written.
    """
    path = Path(path)
    data = {
        "description": "Auto-generated questions pending human review. "
                       "Change status to 'accepted' or 'rejected'. "
                       "Correct golden_answer and match_strings as needed.",
        "total": len(cases),
        "pending": sum(1 for c in cases if c.status == "pending_review"),
        "cases": [c.to_review_dict() for c in cases],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return len(cases)


def import_reviewed(path: str | Path) -> list[QuestionCase]:
    """Load reviewed questions, returning only accepted cases.

    Cases with status != "accepted" are filtered out.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases_data = data.get("cases", data) if isinstance(data, dict) else data
    if isinstance(cases_data, dict):
        cases_data = cases_data.get("cases", [])

    return [
        QuestionCase.from_review_dict(d)
        for d in cases_data
        if isinstance(d, dict) and d.get("status") == "accepted"
    ]


def review_stats(path: str | Path) -> dict:
    """Get counts of pending/accepted/rejected from a review file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases_data = data.get("cases", []) if isinstance(data, dict) else data
    counts = {"pending_review": 0, "accepted": 0, "rejected": 0}
    for d in cases_data:
        status = d.get("status", "pending_review") if isinstance(d, dict) else "unknown"
        counts[status] = counts.get(status, 0) + 1
    return counts


# ── Tier 1: Factual lookup templates ─────────────────────────────────────

_TIER1_TEMPLATES: list[str] = [
    "What is the {predicate} of {subject}?",
    "What {predicate} does {subject} have?",
    "Tell me the {predicate} of {subject}.",
]

PREDICATE_QUESTION_FORMS: dict[str, list[str]] = {
    "court": [
        "What court decided {subject}?",
        "What court heard {subject}?",
        "Which court decided {subject}?",
    ],
    "date_filed": [
        "When was {subject} decided?",
        "When was {subject} filed?",
        "What date was {subject} filed?",
    ],
    "judges": [
        "Who were the judges in {subject}?",
        "Who decided {subject}?",
        "What judges heard {subject}?",
    ],
    "disposition": [
        "What was the ruling in {subject}?",
        "What was the disposition of {subject}?",
        "How was {subject} decided?",
    ],
    "cited_by_count": [
        "How many times has {subject} been cited?",
        "What is the citation count for {subject}?",
    ],
    "nature_of_suit": [
        "What type of case was {subject}?",
        "What was the nature of suit in {subject}?",
    ],
    "opinion_author": [
        "Who wrote the opinion in {subject}?",
        "Who authored the opinion in {subject}?",
    ],
    "per_curiam": [
        "Was {subject} a per curiam opinion?",
        "Is {subject} a per curiam decision?",
    ],
    "attorneys": [
        "Who were the attorneys in {subject}?",
        "Who represented the parties in {subject}?",
    ],
    "precedential_status": [
        "What is the precedential status of {subject}?",
        "Is {subject} a published opinion?",
    ],
    "cites": [
        "What cases does {subject} cite?",
        "Which cases are cited by {subject}?",
    ],
    "holding": [
        "What did the court hold in {subject}?",
        "What was the holding in {subject}?",
        "What did {subject} decide?",
    ],
    "doctrine": [
        "What legal principle was established in {subject}?",
        "What doctrine did {subject} establish?",
        "What legal rule came from {subject}?",
    ],
    "reasoning": [
        "What was the court's reasoning in {subject}?",
        "How did the court reason in {subject}?",
        "What rationale did the court give in {subject}?",
    ],
    "capital": [
        "What is the capital of {subject}?",
    ],
    "population": [
        "What is the population of {subject}?",
    ],
    "leader": [
        "Who is the leader of {subject}?",
        "Who leads {subject}?",
    ],
}

LONG_PREDICATES: set[str] = {"holding", "doctrine", "reasoning"}

SKIP_PREDICATES: set[str] = {"opinions", "headmatter", "headnotes", "syllabus", "summary"}


def canonical_template(predicate: str) -> str | None:
    """Return the canonical (first) question template for a predicate, or None.

    This is the single-template resolver used by triplet-level generators
    (compare.py) that don't want randomized phrasing variety.
    """
    forms = PREDICATE_QUESTION_FORMS.get(predicate.lower())
    return forms[0] if forms else None


def object_length_limit(predicate: str) -> int:
    """Max object length for question generation (doctrinal preds get more room)."""
    return 2000 if predicate.lower() in LONG_PREDICATES else 200


_PREDICATE_QUESTION_FORMS = PREDICATE_QUESTION_FORMS
_LONG_PREDICATES = LONG_PREDICATES
_SKIP_PREDICATES = SKIP_PREDICATES


def generate_tier1(
    kg: KnowledgeGraph,
    *,
    max_per_subject: int = 3,
    template_variety: bool = True,
) -> list[QuestionCase]:
    """Generate Tier 1 factual lookup questions from KG triplets.

    For each subject, generates questions for up to max_per_subject predicates.
    Uses predicate-specific natural language templates when available.
    """
    cases: list[QuestionCase] = []
    subjects = sorted(kg.subjects)

    for subject in subjects:
        facts = kg.lookup(subject=subject)
        selected = facts[:max_per_subject] if len(facts) > max_per_subject else facts

        for fact in selected:
            pred = fact["predicate"]
            obj = fact["object"]

            if pred.lower() in SKIP_PREDICATES:
                continue
            if not obj:
                continue
            if len(obj) > object_length_limit(pred):
                continue

            templates = PREDICATE_QUESTION_FORMS.get(pred.lower())
            if templates and template_variety:
                template = random.choice(templates)
                question = template.format(subject=subject.title())
            else:
                template = random.choice(_TIER1_TEMPLATES)
                question = template.format(
                    predicate=pred.replace("_", " "),
                    subject=subject.title(),
                )

            match_str = obj.lower()
            match_words = [w.strip() for w in match_str.split(",") if len(w.strip()) > 2]
            if not match_words:
                match_words = [match_str]
            match_words = match_words[:3]

            cases.append(QuestionCase(
                question=question,
                golden_answer=obj,
                match_strings=match_words,
                is_negative=False,
                tier=1,
                source_triplet=(subject, pred, obj),
            ))

    return cases


# ── Tier 2: Relational traversal ────────────────────────────────────────

_TIER2_TEMPLATES: dict[str, list[str]] = {
    "cites": [
        "What cases does {subject} cite?",
        "Which cases are cited by {subject}?",
    ],
}


def generate_tier2(
    kg: KnowledgeGraph,
    *,
    target_predicates: set[str] | None = None,
) -> list[QuestionCase]:
    """Generate Tier 2 relational traversal questions.

    Finds subjects with multiple facts sharing the same predicate
    (e.g., multiple 'cites' relationships) and generates questions
    that require traversal to answer.
    """
    if target_predicates is None:
        target_predicates = {"cites"}

    cases: list[QuestionCase] = []
    subjects = sorted(kg.subjects)

    for subject in subjects:
        facts = kg.lookup(subject=subject)
        pred_groups: dict[str, list[dict]] = {}
        for f in facts:
            pred_groups.setdefault(f["predicate"], []).append(f)

        for pred, group in pred_groups.items():
            if pred.lower() not in target_predicates:
                continue
            if len(group) < 2:
                continue

            templates = _TIER2_TEMPLATES.get(pred.lower(), [
                f"What are the {{subject}}'s {pred.replace('_', ' ')} relationships?",
            ])
            question = random.choice(templates).format(subject=subject.title())

            objects = [f["object"] for f in group]
            match_strings = [o.lower() for o in objects[:5]]

            cases.append(QuestionCase(
                question=question,
                golden_answer=", ".join(objects),
                match_strings=match_strings,
                is_negative=False,
                tier=2,
                source_triplet=(subject, pred, objects[0]),
            ))

    return cases


# ── Negative cases ───────────────────────────────────────────────────────

_NONEXISTENT_PREDICATES: list[str] = [
    "GDP",
    "population density",
    "area in square miles",
    "chief justice",
    "majority opinion author",
    "dissenting opinion",
    "constitutional amendment",
    "appeal status",
]


def generate_negatives(
    kg: KnowledgeGraph,
    *,
    count: int = 5,
) -> list[QuestionCase]:
    """Generate negative cases — questions about predicates that don't exist in the KG.

    Picks real entities but asks about non-existent predicates.
    """
    cases: list[QuestionCase] = []
    subjects = sorted(kg.subjects)

    if not subjects:
        return cases

    for i in range(min(count, len(subjects))):
        subject = subjects[i % len(subjects)]
        fake_pred = _NONEXISTENT_PREDICATES[i % len(_NONEXISTENT_PREDICATES)]
        question = f"What is the {fake_pred.lower()} of {subject.title()}?"

        cases.append(QuestionCase(
            question=question,
            golden_answer="[ABSTAIN]",
            match_strings=[],
            is_negative=True,
            tier=1,
            source_triplet=None,
        ))

    return cases


# ── Orchestrator ─────────────────────────────────────────────────────────


def generate_all(
    kg: KnowledgeGraph,
    *,
    max_tier1_per_subject: int = 3,
    negative_count: int = 5,
    tier2_predicates: set[str] | None = None,
    call_llm_fn=None,
) -> list[QuestionCase]:
    """Generate a complete test suite from a KG.

    When call_llm_fn is provided, uses LLM to generate richer Tier 1
    questions (especially for holdings/doctrines/reasoning). Falls back
    to templates when LLM is unavailable or fails.
    """
    cases: list[QuestionCase] = []

    if call_llm_fn is not None:
        llm_cases = generate_tier1_llm(
            kg, call_llm_fn, max_per_subject=max_tier1_per_subject,
        )
        cases.extend(llm_cases)
    if not cases:
        cases.extend(generate_tier1(kg, max_per_subject=max_tier1_per_subject))

    cases.extend(generate_tier2(kg, target_predicates=tier2_predicates))
    cases.extend(generate_negatives(kg, count=negative_count))
    return cases


def generate_tier1_llm(
    kg: KnowledgeGraph,
    call_llm_fn,
    *,
    max_per_subject: int = 3,
) -> list[QuestionCase]:
    """Generate Tier 1 questions using LLM for natural phrasing.

    Collects all facts per subject, sends them to the LLM in batch,
    and parses out questions with golden answers.
    """
    from crystal.compare import generate_questions_llm

    subjects = sorted(kg.subjects)
    triplets = []
    for subject in subjects:
        for fact in kg.lookup(subject=subject):
            pred = fact["predicate"]
            obj = fact["object"]
            if pred.lower() in SKIP_PREDICATES:
                continue
            if not obj:
                continue
            if len(obj) > object_length_limit(pred):
                continue
            triplets.append((subject, pred, obj))

    results = generate_questions_llm(
        triplets, call_llm_fn,
        max_questions=len(subjects) * max_per_subject,
        max_per_subject=max_per_subject,
    )

    cases = []
    for r in results:
        st = r.get("source_triplet", [])
        source = tuple(st) if st and len(st) == 3 else None
        golden = r["golden_answer"]
        match_str = golden.lower()
        match_words = [w.strip() for w in match_str.split(",") if len(w.strip()) > 2]
        if not match_words:
            match_words = [match_str[:200]]
        cases.append(QuestionCase(
            question=r["question"],
            golden_answer=golden,
            match_strings=match_words[:3],
            is_negative=False,
            tier=1,
            source_triplet=source,
        ))
    return cases
