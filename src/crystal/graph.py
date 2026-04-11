"""Crystal graph — wires all nodes into the LangGraph state machine.

Includes graceful error degradation: if any node before the LLM call
throws an unexpected exception, the pipeline falls through to the LLM
fallback rather than crashing. This ensures Crystal is never worse
than the naked LLM — at minimum it returns the LLM's answer.
"""

import logging
from functools import wraps

from langgraph.graph import StateGraph, END

from crystal.state import CrystalState
from crystal.nodes.parser import spacy_parser_node
from crystal.nodes.math import math_detection_node, calculator_node
from crystal.nodes.kg import kg_detection_node, kg_node
from crystal.nodes.planner import plan_builder_node
from crystal.nodes.preprocessor import preprocessor_node
from crystal.nodes.compiler import prompt_compiler_node
from crystal.nodes.llm_nodes import direct_return_node, llm_augmented_node, llm_fallback_node

logger = logging.getLogger(__name__)


def _safe_node(node_fn):
    """Wrap a graph node so exceptions trigger LLM fallback instead of crash.

    Sets state fields that ensure downstream routing degrades gracefully:
    prompt_type="no_math" and compiled_prompt=raw_prompt so the LLM
    augmented node can still produce a reasonable answer.
    """
    @wraps(node_fn)
    def wrapper(state):
        try:
            return node_fn(state)
        except Exception:
            logger.exception(
                "Node %s failed — falling back to LLM", node_fn.__name__,
            )
            return {
                "fallback_to_llm": True,
                "plan": [],
                "prompt_type": "no_math",
                "compiled_prompt": state.get("raw_prompt", ""),
                "final_response": "",
            }
    return wrapper


def should_execute_tools(state: CrystalState) -> str:
    if state.get("fallback_to_llm", True):
        return "fallback"
    return "execute"


def should_call_llm_after_compile(state: CrystalState) -> str:
    if state.get("prompt_type") in ("pure_math", "math_answerable", "kg_answerable"):
        return "direct_return"
    return "send_to_llm"


def build_crystal_graph():
    """
    Six paths:
    1. Calculator → Pure math → Direct return (no LLM)
    2. Calculator → Math answerable → Direct return (no LLM)
    3. Calculator → Math augmented → Simplified prompt to LLM
    4. KG → KG answerable → Direct return (no LLM)
    5. KG → KG augmented → Grounded facts to LLM for reasoning
    6. No tools matched → Raw prompt to LLM

    All pre-LLM nodes are wrapped in _safe_node: any exception triggers
    fallback_to_llm=True so the pipeline degrades to naked LLM instead
    of crashing.
    """
    graph = StateGraph(CrystalState)

    graph.add_node("spacy_parser", _safe_node(spacy_parser_node))
    graph.add_node("math_detection", _safe_node(math_detection_node))
    graph.add_node("kg_detection", _safe_node(kg_detection_node))
    graph.add_node("plan_builder", _safe_node(plan_builder_node))
    graph.add_node("preprocessor", _safe_node(preprocessor_node))
    graph.add_node("calculator", _safe_node(calculator_node))
    graph.add_node("kg", _safe_node(kg_node))
    graph.add_node("prompt_compiler", _safe_node(prompt_compiler_node))
    graph.add_node("direct_return", direct_return_node)
    graph.add_node("llm_augmented", llm_augmented_node)
    graph.add_node("llm_fallback", llm_fallback_node)

    graph.set_entry_point("spacy_parser")
    graph.add_edge("spacy_parser", "math_detection")
    graph.add_edge("math_detection", "kg_detection")
    graph.add_edge("kg_detection", "plan_builder")

    graph.add_conditional_edges(
        "plan_builder",
        should_execute_tools,
        {"execute": "preprocessor", "fallback": "llm_fallback"},
    )

    graph.add_edge("preprocessor", "calculator")
    graph.add_edge("calculator", "kg")
    graph.add_edge("kg", "prompt_compiler")

    graph.add_conditional_edges(
        "prompt_compiler",
        should_call_llm_after_compile,
        {"direct_return": "direct_return", "send_to_llm": "llm_augmented"},
    )

    graph.add_edge("direct_return", END)
    graph.add_edge("llm_augmented", END)
    graph.add_edge("llm_fallback", END)

    return graph.compile()
