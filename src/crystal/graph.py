"""Crystal graph — wires all nodes into the LangGraph state machine."""

from langgraph.graph import StateGraph, END

from crystal.state import CrystalState
from crystal.nodes.parser import spacy_parser_node
from crystal.nodes.math_detection import math_detection_node
from crystal.nodes.kg_detection import kg_detection_node
from crystal.nodes.planner import plan_builder_node
from crystal.nodes.preprocessor import preprocessor_node
from crystal.nodes.calculator import calculator_node
from crystal.nodes.kg import kg_node
from crystal.nodes.compiler import prompt_compiler_node
from crystal.nodes.llm_nodes import direct_return_node, llm_augmented_node, llm_fallback_node


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
    Five paths:
    1. Calculator → Pure math → Direct return (no LLM)
    2. Calculator → Math answerable → Direct return (no LLM)
    3. Calculator → Math augmented → Simplified prompt to LLM
    4. KG → KG answerable → Direct return (no LLM)
    5. No tools matched → Raw prompt to LLM
    """
    graph = StateGraph(CrystalState)

    graph.add_node("spacy_parser", spacy_parser_node)
    graph.add_node("math_detection", math_detection_node)
    graph.add_node("kg_detection", kg_detection_node)
    graph.add_node("plan_builder", plan_builder_node)
    graph.add_node("preprocessor", preprocessor_node)
    graph.add_node("calculator", calculator_node)
    graph.add_node("kg", kg_node)
    graph.add_node("prompt_compiler", prompt_compiler_node)
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
