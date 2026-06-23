"""
app.py

Task 4: Interactive Chat Interface.

A Gradio web UI for the CrediTrust complaint-analysis RAG pipeline built in
Task 3 (src/rag_pipeline.py). Features:
    - Text input + Ask button
    - Streamed answer (token-by-token as Qwen generates, or chunk-by-chunk
      for the Anthropic backend) for better perceived responsiveness, since
      generation can take 30-70+ seconds on CPU
    - Retrieved source excerpts displayed below the answer, so the user can
      verify the model's claims against the original complaint text -- this
      directly addresses the hallucination risk documented in eval_report_final.md
    - A Clear button that resets both the answer and the question box

Usage (from the project root):
    python app.py

Then open the printed local URL (usually http://127.0.0.1:7860) in a browser.

Note: the RAGPipeline is loaded ONCE at module import time, not per-request --
loading the embedding model, FAISS index (1.37M vectors), and the generator
model all take real time, and re-loading them on every question would make
the UI unusably slow.
"""

import logging

import gradio as gr

from src.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logger.info("Loading RAG pipeline (embedding model, FAISS index, generator)... this runs once at startup.")
rag = RAGPipeline()
logger.info("RAG pipeline ready.")


def format_sources_markdown(sources) -> str:
    """Render retrieved source chunks as a readable Markdown block."""
    if not sources:
        return "_No sources were retrieved for this question._"

    blocks = []
    for i, s in enumerate(sources, start=1):
        meta = s.get("metadata", {}) or {}
        complaint_id = meta.get("complaint_id", "N/A")
        product = meta.get("product_category", meta.get("product", "N/A"))
        issue = meta.get("issue", "N/A")
        company = meta.get("company", "N/A")
        score = s.get("score")
        score_str = f"{score:.3f}" if isinstance(score, float) else "N/A"

        text = s.get("content", "").strip()

        block = (
            f"**Source {i}** — Complaint `{complaint_id}` · {product} · {issue} · {company} "
            f"<sub>(similarity: {score_str})</sub>\n\n"
            f"> {text}"
        )
        blocks.append(block)

    return "\n\n---\n\n".join(blocks)


def respond(question: str, history):
    """
    Generator function for Gradio's streaming output. Yields updated
    (answer_markdown, sources_markdown) pairs as the LLM generates, so the
    UI updates live instead of waiting for the full answer.
    """
    question = (question or "").strip()
    if not question:
        yield "Please enter a question above.", ""
        return

    sources_md = ""
    try:
        for partial_answer, sources in rag.answer_stream(question):
            sources_md = format_sources_markdown(sources)
            yield partial_answer, sources_md
    except Exception as e:
        logger.exception("Error while answering question")
        yield f"Something went wrong while generating the answer: {e}", sources_md


def clear_all():
    """Reset the question box, answer, and sources panel."""
    return "", "", ""


CUSTOM_CSS = """
#header-row { padding-bottom: 0.5rem; }
#answer-box { min-height: 180px; }
#sources-box { font-size: 0.92rem; }
"""

with gr.Blocks(title="CrediTrust Complaint Analyst", css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
    with gr.Row(elem_id="header-row"):
        gr.Markdown(
            "# CrediTrust Complaint Analyst\n"
            "Ask a question about customer complaints. Answers are generated "
            "from real retrieved complaint excerpts — check the **Sources** "
            "panel below to verify what the answer is based on."
        )

    with gr.Row():
        question_box = gr.Textbox(
            label="Your question",
            placeholder="e.g. Why are customers unhappy with credit card billing disputes?",
            lines=2,
            scale=4,
        )
        with gr.Column(scale=1, min_width=120):
            ask_btn = gr.Button("Ask", variant="primary")
            clear_btn = gr.Button("Clear")

    answer_box = gr.Markdown(label="Answer", elem_id="answer-box", value="")

    with gr.Accordion("Sources used for this answer", open=True):
        sources_box = gr.Markdown(elem_id="sources-box", value="")

    # Submitting via Enter in the textbox or clicking "Ask" both trigger the
    # same streaming response function.
    question_box.submit(respond, inputs=[question_box, gr.State([])], outputs=[answer_box, sources_box])
    ask_btn.click(respond, inputs=[question_box, gr.State([])], outputs=[answer_box, sources_box])

    clear_btn.click(clear_all, inputs=[], outputs=[question_box, answer_box, sources_box])

    gr.Examples(
        examples=[
            "Why are customers unhappy with credit card billing disputes?",
            "What kinds of fraud do customers report with money transfers?",
            "Do customers report unauthorized transactions on their accounts?",
        ],
        inputs=question_box,
        label="Example questions",
    )


if __name__ == "__main__":
    demo.queue().launch()