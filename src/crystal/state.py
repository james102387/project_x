"""CrystalState — shared state schema that flows through all LangGraph nodes."""

from typing import TypedDict, Any


class CrystalState(TypedDict):
    raw_prompt: str
    spacy_doc: Any
    tool_detections: list[dict]
    plan: list[dict]
    preprocessed: list[dict]
    tool_results: list[dict]
    compiled_prompt: str
    prompt_type: str                # pure_math | math_answerable | math_augmented | kg_answerable | no_math
    llm_response: str
    final_response: str             # the actual output to the user
    fallback_to_llm: bool
    token_metrics: dict


def make_initial_state(prompt: str) -> CrystalState:
    """Create a fresh state dict for a given prompt."""
    return {
        "raw_prompt": prompt,
        "spacy_doc": None,
        "tool_detections": [],
        "plan": [],
        "preprocessed": [],
        "tool_results": [],
        "compiled_prompt": "",
        "prompt_type": "",
        "llm_response": "",
        "final_response": "",
        "fallback_to_llm": False,
        "token_metrics": {},
    }
