"""Unit tests for explicit math pattern detectors."""

import pytest
from crystal.detectors.math.explicit import (
    match_verb_pattern,
    match_conjunction_pattern,
    match_noun_pattern,
    match_symbol_pattern,
)


class TestVerbPattern:
    def test_basic(self, parse):
        assert match_verb_pattern(parse("add 5 and 3")) == [5, 3]

    def test_with_filler(self, parse):
        assert match_verb_pattern(parse("can you add 12 to 8")) == [12, 8]

    def test_multi_operand(self, parse):
        assert match_verb_pattern(parse("add 1 and 2 and 3")) == [1, 2, 3]

    def test_no_numbers(self, parse):
        assert match_verb_pattern(parse("add me to the list")) is None


class TestConjunctionPattern:
    def test_basic(self, parse):
        assert match_conjunction_pattern(parse("5 plus 3")) == [5, 3]

    def test_with_question(self, parse):
        result = match_conjunction_pattern(parse("what's 5 plus 3"))
        assert result is not None
        assert 5 in result and 3 in result

    def test_multi_operand(self, parse):
        result = match_conjunction_pattern(parse("100 plus 200 plus 50"))
        assert result is not None
        assert sum(result) == 350


class TestNounPattern:
    def test_sum(self, parse):
        assert match_noun_pattern(parse("the sum of 5 and 3")) == [5, 3]

    def test_total(self, parse):
        assert match_noun_pattern(parse("find the total of 7 and 8")) == [7, 8]

    def test_figurative(self, parse):
        assert match_noun_pattern(parse("the sum of all fears")) is None


class TestSymbolPattern:
    def test_basic(self, parse):
        assert match_symbol_pattern(parse("5 + 3")) == [5, 3]

    def test_multi(self, parse):
        result = match_symbol_pattern(parse("10 + 20 + 30"))
        assert result is not None
        assert sum(result) == 60

    def test_with_question(self, parse):
        result = match_symbol_pattern(parse("what's 5 + 3"))
        assert result is not None
