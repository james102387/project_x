"""Knowledge Graph tool — hash-table lookup over (subject, predicate, object) triplets.

Dataset-agnostic: accepts any list of triplets at construction time.
Optionally accepts alias maps for synonym resolution and a fuzzy threshold.

Resolution cascade (entities and predicates):
  Tier 1: Exact match (O(1) hash lookup)
  Tier 2: Alias table (O(1) dict lookup)
  Tier 3: Fuzzy string match (rapidfuzz, sub-ms)
"""

from __future__ import annotations

from typing import TypeAlias

from crystal.tools.kg.fuzzy import fuzzy_match

Triplet: TypeAlias = tuple[str, str, str]


class KnowledgeGraph:
    """O(1) hash-table KG backed by forward and reverse indexes."""

    def __init__(
        self,
        triplets: list[Triplet],
        predicate_aliases: dict[str, str] | None = None,
        entity_aliases: dict[str, str] | None = None,
        fuzzy_threshold: float = 80.0,
    ) -> None:
        self.triplets = triplets
        self.fuzzy_threshold = fuzzy_threshold
        self._predicate_aliases: dict[str, str] = {
            k.lower(): v.lower() for k, v in (predicate_aliases or {}).items()
        }
        self._entity_aliases: dict[str, str] = {
            k.lower(): v.lower() for k, v in (entity_aliases or {}).items()
        }
        self._forward: dict[str, dict] = {}
        self._reverse: dict[str, dict] = {}
        self._entity_index: set[str] = set()
        self._subject_index: set[str] = set()
        self._build(triplets)

    def _build(self, triplets: list[Triplet]) -> None:
        for subject, predicate, obj in triplets:
            entry = {"subject": subject, "predicate": predicate, "object": obj}
            self._forward[f"{subject}|{predicate}".lower()] = entry
            self._reverse[f"{predicate}|{obj}".lower()] = entry
            self._entity_index.add(subject.lower())
            self._entity_index.add(obj.lower())
            self._subject_index.add(subject.lower())

    # ── Mutation ───────────────────────────────────────────────────────

    def extend(self, triplets: list[Triplet]) -> int:
        """Append new triplets, skipping duplicates. Returns count added."""
        new = [
            (s, p, o) for s, p, o in triplets
            if f"{s}|{p}".lower() not in self._forward
        ]
        self.triplets.extend(new)
        self._build(new)
        return len(new)

    # ── Resolution cascades ───────────────────────────────────────────

    def _resolve_entity(self, text: str) -> tuple[str, str]:
        """Resolve a surface form to a canonical entity via 3-tier cascade.

        Returns (canonical_entity, match_tier) where match_tier is
        "exact", "alias", or "fuzzy".
        """
        low = text.lower()
        if low in self._entity_index:
            return low, "exact"

        if low in self._entity_aliases:
            return self._entity_aliases[low], "alias"

        result = fuzzy_match(low, self._entity_index, self.fuzzy_threshold)
        if result is not None:
            return result[0], "fuzzy"

        return low, "none"

    def _resolve_predicate(self, predicate: str) -> str:
        """Map an alias to its canonical predicate, or return as-is."""
        return self._predicate_aliases.get(predicate.lower(), predicate)

    def _resolve_predicate_cascade(
        self, predicate: str, subject: str | None = None,
    ) -> tuple[str, str]:
        """Resolve a predicate via 3-tier cascade: exact → alias → fuzzy.

        If subject is given, fuzzy matching runs against predicates for that
        subject only (narrower candidate set = better precision).

        Returns (resolved_predicate, match_tier).
        """
        low = predicate.lower()

        # Tier 1: exact — check if the predicate exists in forward index
        if subject:
            key = f"{subject}|{low}".lower()
            if key in self._forward:
                return low, "exact"
        else:
            canonical_preds = {e.split("|", 1)[1] for e in self._forward}
            if low in canonical_preds:
                return low, "exact"

        # Tier 2: alias
        alias_resolved = self._predicate_aliases.get(low)
        if alias_resolved is not None:
            return alias_resolved, "alias"

        # Tier 3: fuzzy — match against predicates for the resolved subject
        if subject:
            prefix = f"{subject}|".lower()
            subject_preds = [
                k.split("|", 1)[1] for k in self._forward if k.startswith(prefix)
            ]
        else:
            subject_preds = list({e.split("|", 1)[1] for e in self._forward})

        if subject_preds:
            result = fuzzy_match(low, subject_preds, self.fuzzy_threshold)
            if result is not None:
                return result[0], "fuzzy"

        return low, "none"

    @property
    def entities(self) -> set[str]:
        """All known entity strings (lowercased) from subjects and objects."""
        return self._entity_index

    @property
    def subjects(self) -> set[str]:
        """All known subject strings (lowercased)."""
        return self._subject_index

    def has_entity(self, text: str) -> bool:
        """Check whether a string matches a known entity (exact or alias)."""
        low = text.lower()
        if low in self._entity_index:
            return True
        if low in self._entity_aliases:
            return True
        return False

    # ── Core lookup ───────────────────────────────────────────────────

    def lookup(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        obj: str | None = None,
    ) -> list[dict]:
        """Query the KG.

        Predicates are resolved through the alias map before lookup.

        Supports:
            lookup(subject="X", predicate="Y")  -> forward lookup
            lookup(predicate="Y", obj="Z")       -> reverse lookup
            lookup(subject="X")                   -> all facts about subject
        """
        if predicate:
            predicate = self._resolve_predicate(predicate)

        results: list[dict] = []

        if subject and predicate:
            key = f"{subject}|{predicate}".lower()
            if key in self._forward:
                results.append(self._forward[key])

        elif predicate and obj:
            key = f"{predicate}|{obj}".lower()
            if key in self._reverse:
                results.append(self._reverse[key])

        elif subject:
            prefix = f"{subject}|".lower()
            for key, value in self._forward.items():
                if key.startswith(prefix):
                    results.append(value)

        return results

    # ── Multi-hop traversal ───────────────────────────────────────────

    def traverse(
        self,
        entity: str,
        max_depth: int = 2,
    ) -> list[dict]:
        """Recursive depth-limited graph traversal from a starting entity.

        Collects all facts reachable within `max_depth` hops. A "hop" follows
        objects of triplets to discover further subjects. Avoids cycles via
        a visited set.

        Returns a flat list of triplet dicts, ordered by discovery (BFS-like).
        """
        visited: set[str] = set()
        result: list[dict] = []
        frontier: list[tuple[str, int]] = [(entity.lower(), 0)]

        while frontier:
            current, depth = frontier.pop(0)
            if current in visited:
                continue
            visited.add(current)

            facts = self.lookup(subject=current)
            for fact in facts:
                if fact not in result:
                    result.append(fact)
                obj_lower = fact["object"].lower()
                if (
                    depth + 1 <= max_depth
                    and obj_lower not in visited
                    and obj_lower in self._subject_index
                ):
                    frontier.append((obj_lower, depth + 1))

        return result

    def __len__(self) -> int:
        return len(self.triplets)

    def __repr__(self) -> str:
        return f"KnowledgeGraph({len(self)} triplets)"
