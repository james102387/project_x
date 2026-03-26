"""Unit tests for the NER-based triplet extractor."""

import pytest
import spacy

from crystal.ingest.ner import extract_triplets, ingest_text, ingest_file, find_unresolved_sentences


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm")


class TestCopularPattern:
    """Pattern 1: "X is/are Y" """

    def test_simple_copular(self, nlp):
        triplets = extract_triplets("Remulak is a planet.", nlp=nlp)
        assert len(triplets) >= 1
        t = triplets[0]
        assert t.subject == "Remulak"
        assert t.predicate == "is a"
        assert "planet" in t.object

    def test_copular_with_subject_of(self, nlp):
        """'The capital of Remulak is Zelphos' → (Remulak, capital, Zelphos)"""
        triplets = extract_triplets("The capital of Remulak is Zelphos.", nlp=nlp)
        assert len(triplets) >= 1
        t = triplets[0]
        assert t.subject == "Remulak"
        assert t.predicate == "capital"
        assert t.object == "Zelphos"

    def test_copular_with_attr_of(self, nlp):
        """'Zelphos is the capital of Remulak' → (Remulak, capital, Zelphos)"""
        triplets = extract_triplets("Zelphos is the capital of Remulak.", nlp=nlp)
        assert len(triplets) >= 1
        t = triplets[0]
        assert t.subject == "Remulak"
        assert t.predicate == "capital"
        assert t.object == "Zelphos"

    def test_copular_with_prep(self, nlp):
        """'Remulak is a planet in the Veldra-7 star system'"""
        triplets = extract_triplets(
            "Remulak is a planet in the Veldra-7 star system.", nlp=nlp,
        )
        assert len(triplets) >= 1
        assert triplets[0].subject == "Remulak"
        assert triplets[0].predicate == "is a"


class TestPassivePattern:
    """Pattern 4: 'X was Vd by Y' """

    def test_passive_with_agent(self, nlp):
        triplets = extract_triplets(
            "The Veldran Guard is commanded by Marshal Draya Kess.", nlp=nlp,
        )
        assert len(triplets) >= 1
        t = triplets[0]
        assert "Veldran Guard" in t.subject
        assert "commanded by" in t.predicate
        assert "Marshal Draya Kess" in t.object

    def test_passive_born_in(self, nlp):
        triplets = extract_triplets(
            "Grand Vizier Korth was born in Zelphos.", nlp=nlp,
        )
        assert len(triplets) >= 1
        t = triplets[0]
        assert "Grand Vizier Korth" in t.subject
        assert "born in" in t.predicate
        assert t.object == "Zelphos"

    def test_passive_invented_by(self, nlp):
        triplets = extract_triplets(
            "Resonance-fold drives were invented by Physicist Orath Yenn.", nlp=nlp,
        )
        assert len(triplets) >= 1
        t = triplets[0]
        assert "invented by" in t.predicate
        assert "Orath Yenn" in t.object

    def test_passive_found_in(self, nlp):
        triplets = extract_triplets(
            "Resonance crystals are found in Sulari.", nlp=nlp,
        )
        assert len(triplets) >= 1
        t = triplets[0]
        assert "found in" in t.predicate
        assert t.object == "Sulari"

    def test_passive_celebrated_on(self, nlp):
        triplets = extract_triplets(
            "The Festival of Vohn is celebrated on the summer solstice.", nlp=nlp,
        )
        assert len(triplets) >= 1
        t = triplets[0]
        assert "Festival of Vohn" in t.subject
        assert "celebrated on" in t.predicate


class TestActivePattern:
    """Patterns 2/3: 'X verbs Y' / 'X has Y' """

    def test_has_dobj(self, nlp):
        triplets = extract_triplets(
            "Remulak has a population of 4.3 billion.", nlp=nlp,
        )
        assert len(triplets) >= 1
        t = triplets[0]
        assert t.subject == "Remulak"
        assert t.predicate == "have"
        assert "population" in t.object
        assert "4.3 billion" in t.object

    def test_active_transitive(self, nlp):
        triplets = extract_triplets(
            "The Sulari Fracture War lasted 12 standard years.", nlp=nlp,
        )
        assert len(triplets) >= 1
        t = triplets[0]
        assert "Sulari Fracture War" in t.subject
        assert t.predicate == "last"
        assert "12 standard years" in t.object


class TestHyphenatedEntities:
    def test_hyphen_in_subject(self, nlp):
        triplets = extract_triplets(
            "Dark-ore is used for starship hull reinforcement.", nlp=nlp,
        )
        assert len(triplets) >= 1
        assert triplets[0].subject == "Dark-ore"

    def test_hyphen_preserved(self, nlp):
        triplets = extract_triplets(
            "Resonance-fold drives were invented by Physicist Orath Yenn.", nlp=nlp,
        )
        assert len(triplets) >= 1
        assert "Resonance-fold" in triplets[0].subject


class TestMultiSentence:
    def test_extracts_from_multiple_sentences(self, nlp):
        text = (
            "Remulak is a planet. "
            "The capital of Remulak is Zelphos. "
            "Grand Vizier Korth was born in Zelphos."
        )
        triplets = extract_triplets(text, nlp=nlp)
        assert len(triplets) >= 3
        subjects = {t.subject for t in triplets}
        assert "Remulak" in subjects

    def test_empty_text(self, nlp):
        triplets = extract_triplets("", nlp=nlp)
        assert triplets == []


class TestIngestText:
    def test_returns_ingest_result(self, nlp):
        result = ingest_text("Remulak is a planet.", source="test", nlp=nlp)
        assert len(result.triplets) >= 1
        assert result.source == "test"
        assert result.as_tuples()[0][0] == "Remulak"

    def test_source_preserved(self, nlp):
        result = ingest_text("hello", source="my_doc.txt", nlp=nlp)
        assert result.source == "my_doc.txt"


class TestIngestFile:
    def test_reads_text_file(self, nlp, tmp_path):
        p = tmp_path / "test.txt"
        p.write_text("The capital of Remulak is Zelphos.")
        result = ingest_file(str(p), nlp=nlp)
        assert len(result.triplets) >= 1
        assert result.source == str(p)
        assert result.triplets[0].subject == "Remulak"
        assert result.triplets[0].object == "Zelphos"


class TestFindUnresolvedSentences:
    """D2 Phase 2: detect sentences where NER found entities but no predicates."""

    def test_resolved_sentences_excluded(self, nlp):
        text = "Remulak is a planet."
        unresolved = find_unresolved_sentences(text, nlp=nlp)
        assert len(unresolved) == 0

    def test_no_entities_excluded(self, nlp):
        text = "Hello there."
        unresolved = find_unresolved_sentences(text, nlp=nlp)
        assert len(unresolved) == 0

    def test_unresolved_with_entities_found(self, nlp):
        text = "Unlike Remulak, Draveth specializes more in governmental traditions."
        unresolved = find_unresolved_sentences(text, nlp=nlp)
        # This is a sentence structure that dep-tree patterns may struggle with,
        # but it contains entity-like noun chunks. We just verify the function
        # returns sentences with at least 2 chunks that produced no triplets.
        for sent_text, entities in unresolved:
            assert len(entities) >= 2
            assert isinstance(sent_text, str)

    def test_mixed_text(self, nlp):
        text = (
            "Remulak is a planet. "
            "The capital of Remulak is Zelphos. "
            "Between Draveth and Sulari, trade relationships remain complex."
        )
        unresolved = find_unresolved_sentences(text, nlp=nlp)
        # The first two sentences should resolve. The third might not.
        # All resolved sentences should be excluded.
        resolved_texts = {"Remulak is a planet.", "The capital of Remulak is Zelphos."}
        for sent_text, _ in unresolved:
            assert sent_text not in resolved_texts

    def test_empty_text(self, nlp):
        unresolved = find_unresolved_sentences("", nlp=nlp)
        assert unresolved == []

    def test_single_entity_excluded(self, nlp):
        text = "Remulak alone."
        unresolved = find_unresolved_sentences(text, nlp=nlp)
        assert len(unresolved) == 0
