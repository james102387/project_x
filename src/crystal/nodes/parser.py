"""spaCy parser node — shared parsing foundation for all detectors."""

import spacy

nlp = spacy.load("en_core_web_sm")


def spacy_parser_node(state: dict) -> dict:
    """Parse the raw prompt into a spaCy Doc."""
    doc = nlp(state["raw_prompt"])
    return {"spacy_doc": doc}


def show_parse(prompt: str):
    """Print the full spaCy parse tree for debugging."""
    doc = nlp(prompt)
    print(f'\nParsing: "{prompt}"')
    print(f"{'Token':<12} {'POS':<8} {'Dep':<12} {'Head':<12} {'Children'}")
    print("-" * 70)
    for token in doc:
        children = [f"{c.text}({c.dep_})" for c in token.children]
        print(
            f"{token.text:<12} {token.pos_:<8} {token.dep_:<12} "
            f"{token.head.text:<12} {children}"
        )
