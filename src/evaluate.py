"""
evaluate.py

Task 3: Qualitative evaluation.

Runs a fixed set of representative questions through the RAG pipeline and
writes a Markdown table (Question | Generated Answer | Retrieved Sources |
Quality Score | Comments). Quality Score and Comments are left as TODO
placeholders for manual review -- scoring a RAG system's answer quality is
a judgment call that should be made by a human reading the actual output,
not auto-filled.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...   # if using the anthropic backend
    python evaluate.py

Output:
    eval_report.md  -- Markdown table ready to paste into the final report
"""

import logging

from src.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Edit this list to fit the products/issues actually present in your dataset.
EVAL_QUESTIONS = [
    "Why are customers unhappy with credit card billing disputes?",
    "What kinds of fraud do customers report with money transfers or virtual currency?",
    "What complaints come up most about checking or savings account fees?",
    "Are customers reporting issues getting personal loan applications approved?",
    "What problems do people have with customer service responsiveness?",
    "Do customers report unauthorized transactions on their accounts?",
    "What issues are reported around loan repayment or collections?",
    "Are there complaints about incorrect information being reported to credit bureaus?",
]


def truncate(text: str, n: int = 200) -> str:
    text = " ".join(text.split())  # collapse whitespace/newlines for table cells
    return text if len(text) <= n else text[: n - 3] + "..."


def md_escape(text: str) -> str:
    return text.replace("|", "\\|")


def run_evaluation(questions=None, k: int = 5, out_path: str = "eval_report.md"):
    questions = questions or EVAL_QUESTIONS
    rag = RAGPipeline(top_k=k)

    rows = []
    for q in questions:
        logger.info(f"Running: {q}")
        result = rag.answer(q, k=k)

        sources_str = "<br>".join(
            f"[{s['metadata'].get('complaint_id', 'N/A')} | "
            f"{s['metadata'].get('product_category', 'N/A')}] "
            f"{truncate(s['content'], 120)}"
            for s in result["sources"][:2]
        ) or "_No sources retrieved_"

        rows.append(
            {
                "question": q,
                "answer": truncate(result["answer"], 300),
                "sources": sources_str,
                "score": "TODO (1-5)",
                "comments": "TODO: review answer against sources for faithfulness, relevance, completeness.",
            }
        )

    lines = [
        "| Question | Generated Answer | Retrieved Sources (top 2) | Quality Score | Comments/Analysis |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {md_escape(r['question'])} | {md_escape(r['answer'])} | "
            f"{md_escape(r['sources'])} | {r['score']} | {md_escape(r['comments'])} |"
        )

    table_md = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write("# RAG Qualitative Evaluation\n\n")
        f.write(table_md)
        f.write("\n")

    logger.info(f"Wrote evaluation table to {out_path}")
    return rows


if __name__ == "__main__":
    run_evaluation()
