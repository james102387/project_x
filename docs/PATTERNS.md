# spaCy Pattern Reference

Documented patterns with actual dependency trees from `en_core_web_sm`.

## Explicit Addition Patterns

### Verb-led: "add 5 and 3"
```
add (VERB, ROOT)
├── 5 (NUM, dobj)
│   ├── and (CCONJ, cc)
│   └── 3 (NUM, conj)
```
**Detection:** VERB with lemma "add" → check subtree for NUM children.

### Conjunction: "what is 10 plus 20"
```
is (AUX, ROOT)
└── 20 (NUM, attr)
    └── 10 (NUM, nummod)
        └── plus (CCONJ, cc)
```
**Note:** "what is" and "what's" parse identically. CCONJ "plus" has NUM head.
Detection: find CCONJ "plus" with NUM head, then collect all NUM tokens in doc.

### Noun-led: "the sum of 5 and 3"
```
sum (NOUN, ROOT)
├── the (DET, det)
└── of (ADP, prep)
    └── 5 (NUM, pobj)
        ├── and (CCONJ, cc)
        └── 3 (NUM, conj)
```
**Detection:** NOUN with lemma "sum"/"total" → find "of" prep child → collect NUM in subtree.

### Symbol: "5 + 3"
```
5 (NUM, ROOT)
├── + (SYM, cc)
└── 3 (NUM, conj)
```
**Quirk:** For "10 + 20 + 30", spaCy tags `+` as NUM, not SYM.
Detection: find any token with text "+" regardless of POS.

## Semantic Verb Patterns

### "Adam has 10 chairs, sells 6, and then makes 7 more"
```
has (VERB, ROOT)
├── Adam (PROPN, nsubj)
├── chairs (NOUN, dobj)
│   └── 10 (NUM, nummod)
├── sells (VERB, conj)         ← skip when scanning 'has'
│   ├── 6 (NUM, dobj)
│   ├── makes (VERB, conj)     ← conjoined to 'sells'
│   │   ├── 7 (NUM, dobj)
│   │   │   └── more (ADJ, amod)
```
**Critical:** Must skip `conj` VERB children when collecting NUM tokens.

## Known Gotchas

1. **"what is" vs "what's"**: Identical dep trees, but test both.
2. **Multi-symbol `+`**: spaCy may tag as NUM instead of SYM. Match on text, not POS.
3. **Conjoined verbs**: A ROOT verb's subtree includes all conjoined clauses.
   Always use direct children + one level, never full subtree for semantic matching.
4. **"add" in word problems**: "add 3 more" may not have NUM in explicit subtree
   because "3" attaches to "more". Falls through to semantic matcher.
