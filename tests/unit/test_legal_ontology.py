"""Tests for legal ontology — case name normalization, citation parsing, alias generation."""

import pytest

from crystal.data.legal_ontology import (
    LEGAL_PREDICATES,
    LEGAL_PREDICATE_ALIASES,
    COURT_TYPE_MAP,
    _FIRST_PARTY_BLOCKLIST,
    _is_safe_first_party,
    normalize_case_name,
    parse_citation,
    generate_case_aliases,
    deduplicate_aliases,
)


# ── normalize_case_name ──────────────────────────────────────────────────


class TestNormalizeCaseName:
    def test_standard_v_dot(self):
        assert normalize_case_name("Miranda v. Arizona") == "Miranda v. Arizona"

    def test_vs_dot(self):
        assert normalize_case_name("Miranda vs. Arizona") == "Miranda v. Arizona"

    def test_vs_no_dot(self):
        assert normalize_case_name("Miranda vs Arizona") == "Miranda v. Arizona"

    def test_bare_v(self):
        assert normalize_case_name("Miranda v Arizona") == "Miranda v. Arizona"

    def test_strips_trailing_year_paren(self):
        assert normalize_case_name("Roe v. Wade (1973)") == "Roe v. Wade"

    def test_collapses_whitespace(self):
        assert normalize_case_name("Miranda  v.   Arizona") == "Miranda v. Arizona"

    def test_title_case(self):
        result = normalize_case_name("MIRANDA v. ARIZONA")
        assert result == "Miranda v. Arizona"

    def test_title_case_preserves_v_dot(self):
        result = normalize_case_name("brown v. board of education")
        assert result == "Brown v. Board Of Education"

    def test_empty_string(self):
        assert normalize_case_name("") == ""

    def test_no_v_single_party(self):
        result = normalize_case_name("In re Gault")
        assert result == "In Re Gault"

    def test_full_name_with_roles(self):
        result = normalize_case_name(
            "The STATE of Oklahoma, Appellant, v. Frankie HOWERTON, Appellee"
        )
        assert "v." in result
        assert "Howerton" in result


# ── parse_citation ───────────────────────────────────────────────────────


class TestParseCitation:
    def test_us_reports(self):
        assert parse_citation("384 U.S. 436") == "384 U.S. 436"

    def test_federal_reporter_3d(self):
        assert parse_citation("46 P.3d 154") == "46 P.3d 154"

    def test_federal_supplement(self):
        assert parse_citation("933 F. Supp. 781") == "933 F. Supp. 781"

    def test_garbage_returns_none(self):
        assert parse_citation("not a citation") is None

    def test_empty_returns_none(self):
        assert parse_citation("") is None

    def test_none_returns_none(self):
        assert parse_citation(None) is None

    def test_embedded_in_text(self):
        result = parse_citation("See 384 U.S. 436 at 440")
        assert result == "384 U.S. 436"

    def test_supreme_court_reporter(self):
        result = parse_citation("86 S. Ct. 1602")
        assert result is not None
        assert "86" in result
        assert "1602" in result


# ── generate_case_aliases ────────────────────────────────────────────────


class TestGenerateCaseAliases:
    def test_short_name_alias(self):
        aliases = generate_case_aliases(
            case_name="Miranda v. Arizona",
            case_name_short="Miranda",
        )
        canonical = "miranda v. arizona"
        assert aliases.get("miranda") == canonical

    def test_citation_alias(self):
        aliases = generate_case_aliases(
            case_name="Miranda v. Arizona",
            citations=["384 U.S. 436"],
        )
        canonical = "miranda v. arizona"
        assert aliases.get("384 u.s. 436") == canonical

    def test_slug_alias(self):
        aliases = generate_case_aliases(
            case_name="State v. Howerton",
            slug="state-v-howerton",
        )
        canonical = "state v. howerton"
        assert aliases.get("state v howerton") == canonical

    def test_full_name_alias(self):
        aliases = generate_case_aliases(
            case_name="State v. Howerton",
            case_name_full="The STATE of Oklahoma, Appellant, v. Frankie HOWERTON, Appellee",
        )
        canonical = "state v. howerton"
        assert any(v == canonical for v in aliases.values())

    def test_first_party_alias(self):
        aliases = generate_case_aliases(
            case_name="Miranda v. Arizona",
        )
        canonical = "miranda v. arizona"
        assert aliases.get("miranda") == canonical

    def test_no_first_party_alias_for_short_names(self):
        """Don't create aliases for very short first-party names (<=3 chars)."""
        aliases = generate_case_aliases(case_name="In v. Re")
        assert "in" not in aliases

    def test_no_first_party_alias_for_blocked_states(self):
        """US state names should never become first-party aliases."""
        for case_name in ["Texas v. Johnson", "Ohio v. Clark", "Virginia v. Moore"]:
            aliases = generate_case_aliases(case_name=case_name)
            first_party = case_name.split(" v. ")[0].lower()
            assert first_party not in aliases, (
                f"Blocked name '{first_party}' should not be an alias for '{case_name}'"
            )

    def test_no_first_party_alias_for_generic_entities(self):
        """Generic government entities should be blocked."""
        for case_name in [
            "United States v. Nixon",
            "State v. Howerton",
            "People v. Turner",
            "Commonwealth v. Hunt",
        ]:
            aliases = generate_case_aliases(case_name=case_name)
            first_party = case_name.split(" v. ")[0].lower()
            assert first_party not in aliases, (
                f"Generic entity '{first_party}' should not be an alias for '{case_name}'"
            )

    def test_no_first_party_alias_for_short_single_words(self):
        """Single words <=5 chars are rejected even if not blocklisted."""
        aliases = generate_case_aliases(case_name="Smith v. Jones")
        assert "smith" not in aliases

    def test_first_party_alias_for_distinctive_names(self):
        """Multi-word or longer distinctive names should still generate aliases."""
        aliases = generate_case_aliases(case_name="Miranda v. Arizona")
        assert aliases.get("miranda") == "miranda v. arizona"

        aliases2 = generate_case_aliases(case_name="Griswold v. Connecticut")
        assert aliases2.get("griswold") == "griswold v. connecticut"

    def test_empty_case_name(self):
        assert generate_case_aliases(case_name="") == {}

    def test_multiple_citations(self):
        aliases = generate_case_aliases(
            case_name="Miranda v. Arizona",
            citations=["384 U.S. 436", "86 S. Ct. 1602"],
        )
        canonical = "miranda v. arizona"
        assert aliases.get("384 u.s. 436") == canonical
        assert "86 s. ct. 1602" in aliases


# ── _is_safe_first_party ─────────────────────────────────────────────────


class TestIsSafeFirstParty:
    def test_blocklisted_state(self):
        assert _is_safe_first_party("Texas") is False

    def test_blocklisted_entity(self):
        assert _is_safe_first_party("United States") is False
        assert _is_safe_first_party("State") is False

    def test_short_single_word(self):
        assert _is_safe_first_party("Smith") is False
        assert _is_safe_first_party("Roe") is False

    def test_distinctive_name(self):
        assert _is_safe_first_party("Miranda") is True
        assert _is_safe_first_party("Griswold") is True
        assert _is_safe_first_party("Obergefell") is True

    def test_multi_word_specific(self):
        assert _is_safe_first_party("New York Times Co.") is True
        assert _is_safe_first_party("Planned Parenthood") is True

    def test_blocklist_has_all_50_states(self):
        states_50 = {
            "alabama", "alaska", "arizona", "arkansas", "california",
            "colorado", "connecticut", "delaware", "florida", "georgia",
            "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas",
            "kentucky", "louisiana", "maine", "maryland", "massachusetts",
            "michigan", "minnesota", "mississippi", "missouri", "montana",
            "nebraska", "nevada", "new hampshire", "new jersey", "new mexico",
            "new york", "north carolina", "north dakota", "ohio", "oklahoma",
            "oregon", "pennsylvania", "rhode island", "south carolina",
            "south dakota", "tennessee", "texas", "utah", "vermont",
            "virginia", "washington", "west virginia", "wisconsin", "wyoming",
        }
        for state in states_50:
            assert state in _FIRST_PARTY_BLOCKLIST, f"Missing state: {state}"


# ── deduplicate_aliases ──────────────────────────────────────────────────


class TestDeduplicateAliases:
    def test_no_conflict_merges(self):
        acc = {"a": "x"}
        new = {"b": "y"}
        result = deduplicate_aliases(acc, new)
        assert result == {"a": "x", "b": "y"}

    def test_same_canonical_keeps(self):
        acc = {"miranda": "miranda v. arizona"}
        new = {"miranda": "miranda v. arizona"}
        result = deduplicate_aliases(acc, new)
        assert result["miranda"] == "miranda v. arizona"

    def test_conflicting_canonical_removes(self):
        acc = {"brown": "brown v. board of education"}
        new = {"brown": "brown v. allen"}
        result = deduplicate_aliases(acc, new)
        assert "brown" not in result

    def test_mixed_conflict_and_safe(self):
        acc = {"miranda": "miranda v. arizona", "brown": "brown v. board of education"}
        new = {"brown": "brown v. allen", "384 u.s. 436": "miranda v. arizona"}
        result = deduplicate_aliases(acc, new)
        assert "brown" not in result
        assert result["miranda"] == "miranda v. arizona"
        assert result["384 u.s. 436"] == "miranda v. arizona"

    def test_mutates_in_place(self):
        acc = {"a": "x"}
        new = {"b": "y"}
        result = deduplicate_aliases(acc, new)
        assert result is acc


# ── Ontology data structures ─────────────────────────────────────────────


class TestOntologyStructure:
    def test_predicates_are_non_empty(self):
        assert len(LEGAL_PREDICATES) >= 4

    def test_all_alias_targets_are_canonical(self):
        for alias, target in LEGAL_PREDICATE_ALIASES.items():
            assert target in LEGAL_PREDICATES, (
                f"Alias '{alias}' points to '{target}' which is not a canonical predicate"
            )

    def test_court_type_map_has_federal(self):
        assert "F" in COURT_TYPE_MAP
        assert "FD" in COURT_TYPE_MAP

    def test_court_type_map_has_state(self):
        assert "S" in COURT_TYPE_MAP
        assert "SA" in COURT_TYPE_MAP
