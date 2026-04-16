"""Ask tab — side-by-side Crystal vs naked LLM comparison."""

from __future__ import annotations

import time
from dataclasses import dataclass

import gradio as gr

import crystal.llm
from crystal.state import make_initial_state

from crystal.ui.formatting import _CONFIDENCE_BADGES, _ROUTE_LABELS
from crystal.ui.state import _graph, _legal_kg


def ask_question(question, kg_state):
    """Run a question through Crystal and naked LLM, return side-by-side results."""
    if not question or not question.strip():
        return "", "", "", "", ""

    question = question.strip()

    state = make_initial_state(question, kg=kg_state)
    grounding_info = ""
    try:
        final = _graph.invoke(state)
        crystal_response = final.get("final_response", "")
        prompt_type = final.get("prompt_type", "unknown")
        metrics = final.get("token_metrics", {})

        route_label = _ROUTE_LABELS.get(prompt_type, prompt_type)
        confidence = _CONFIDENCE_BADGES.get(prompt_type, "UNKNOWN")
        crystal_meta = f"**Route:** {route_label}\n**Confidence:** {confidence}"

        if metrics:
            if metrics.get("actual_prompt_tokens"):
                crystal_meta += f"\n**Tokens:** {metrics['actual_prompt_tokens']} prompt, {metrics.get('actual_output_tokens', '?')} output"
            if metrics.get("actual_reasoning_tokens"):
                crystal_meta += f", {metrics['actual_reasoning_tokens']} reasoning"
            if prompt_type in ("pure_math", "math_answerable", "kg_answerable"):
                crystal_meta += "\n**LLM called:** No (direct return)"

        tool_results = final.get("tool_results", [])
        kg_results = [r for r in tool_results if r.get("tool") == "kg" and r.get("success")]
        if kg_results:
            facts = []
            for r in kg_results:
                for triplet in r.get("results", []):
                    s, p, o = triplet.get("subject", ""), triplet.get("predicate", ""), triplet.get("object", "")
                    facts.append(f"- **{s.title()}** — {p}: {o}")
            if facts:
                grounding_info = "### Grounding Facts\n" + "\n".join(facts[:20])
                entity_spans = final.get("tool_detections", [])
                for det in entity_spans:
                    if det.get("tool") == "kg":
                        tier = det.get("match_tier", "?")
                        lookup = det.get("lookup_type", "?")
                        grounding_info += f"\n\n*Match: {tier} | Lookup: {lookup}*"
                        break
        elif prompt_type in ("no_math",):
            grounding_info = "*No KG facts found for this query — answer is ungrounded.*"

    except Exception as e:
        crystal_response = f"Error: {e}"
        crystal_meta = "**Route:** error"

    time.sleep(1)
    try:
        llm_response, llm_usage = crystal.llm.call_llm(question)
        llm_meta = ""
        if llm_usage:
            llm_meta = f"**Tokens:** {llm_usage.get('prompt_tokens', '?')} prompt, {llm_usage.get('output_tokens', '?')} output"
            if llm_usage.get("reasoning_tokens"):
                llm_meta += f", {llm_usage['reasoning_tokens']} reasoning"
    except Exception as e:
        llm_response = f"Error: {e}"
        llm_meta = ""

    return crystal_response, crystal_meta, llm_response, llm_meta, grounding_info


@dataclass
class AskTab:
    question_input: gr.Textbox
    ask_btn: gr.Button
    crystal_output: gr.Textbox
    crystal_meta_output: gr.Markdown
    llm_output: gr.Textbox
    llm_meta_output: gr.Markdown
    grounding_output: gr.Markdown


def build_ask_tab(kg_state: gr.State) -> AskTab:
    gr.Markdown(
        "Ask a question and compare Crystal's grounded answer against the naked LLM."
    )

    placeholder = (
        "e.g. What court decided Miranda v. Arizona?"
        if _legal_kg is not None
        else "e.g. What is the capital of Remulak?"
    )
    with gr.Row():
        question_input = gr.Textbox(
            label="Question",
            placeholder=placeholder,
            scale=4,
        )
        ask_btn = gr.Button("Ask", variant="primary", scale=1)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Crystal (Grounded)")
            crystal_output = gr.Textbox(label="Response", lines=6, interactive=False)
            crystal_meta_output = gr.Markdown("")
        with gr.Column():
            gr.Markdown("### Naked LLM (Baseline)")
            llm_output = gr.Textbox(label="Response", lines=6, interactive=False)
            llm_meta_output = gr.Markdown("")

    grounding_output = gr.Markdown("")

    outputs = [
        crystal_output, crystal_meta_output,
        llm_output, llm_meta_output,
        grounding_output,
    ]
    ask_btn.click(fn=ask_question, inputs=[question_input, kg_state], outputs=outputs)
    question_input.submit(
        fn=ask_question, inputs=[question_input, kg_state], outputs=outputs,
    )

    return AskTab(
        question_input=question_input,
        ask_btn=ask_btn,
        crystal_output=crystal_output,
        crystal_meta_output=crystal_meta_output,
        llm_output=llm_output,
        llm_meta_output=llm_meta_output,
        grounding_output=grounding_output,
    )
