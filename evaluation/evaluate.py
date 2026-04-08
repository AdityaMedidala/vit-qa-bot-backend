"""
NOVA — RAGAS Evaluation Script
================================
Compares:
  A) NOVA pipeline  — hybrid BM25 + vector search → Cohere rerank → GPT-4o-mini
  B) Baseline       — vanilla GPT-4o-mini with no retrieval context

Metrics (via RAGAS):
  - Faithfulness        : does the answer stay faithful to retrieved context?
  - Answer Relevancy    : is the answer relevant to the question?
  - Context Precision   : are retrieved chunks actually useful?
  - Context Recall      : does retrieval cover what the ground truth needs?

Usage:
    python evaluation/evaluate.py
    python evaluation/evaluate.py --limit 10        # run on first N questions only
    python evaluation/evaluate.py --output results/my_run.json

Requirements:
    pip install ragas datasets openai cohere psycopg2-binary python-dotenv
"""

import os
import sys
import json
import argparse
import datetime
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Add project root to path so app.* imports work ───────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from openai import OpenAI
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

from app.final_retreval import retrieve_sql

_openai = OpenAI()

DATASET_PATH = Path(__file__).parent / "dataset.json"
RESULTS_DIR  = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Baseline: vanilla GPT-4o-mini, no retrieval ───────────────────────────────

def baseline_answer(question: str) -> str:
    """Raw GPT-4o-mini with no context — the comparison target."""
    response = _openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the user's question "
                    "about VIT Vellore university as accurately as you can."
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


# ── NOVA pipeline runner ──────────────────────────────────────────────────────

def nova_run(question: str) -> tuple[str, list[str]]:
    """
    Runs the full NOVA retrieval + generation pipeline.
    Returns (answer, list_of_context_strings).
    """
    retrieval = retrieve_sql(question)
    contexts  = [item["text"] for _, item in retrieval.get("results", [])]

    if retrieval.get("mode") == "none" or not contexts:
        return "The document does not clearly specify this.", contexts

    # Build context string
    context_block = "\n\n".join(
        f"[{item['chunk_id']}]\n{item['text']}"
        for _, item in retrieval.get("results", [])
    )

    response = _openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant for university queries. "
                    "Answer the user's question using only the provided context."
                ),
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nContext:\n{context_block}",
            },
        ],
        temperature=0,
    )
    answer = response.choices[0].message.content.strip()
    return answer, contexts


# ── Build RAGAS Dataset ───────────────────────────────────────────────────────

def build_ragas_dataset(qa_pairs: list[dict], mode: str) -> Dataset:
    """
    mode: "nova" | "baseline"
    RAGAS expects columns: question, answer, contexts, ground_truth
    """
    questions     = []
    answers       = []
    contexts_list = []
    ground_truths = []

    total = len(qa_pairs)
    for i, pair in enumerate(qa_pairs, 1):
        q  = pair["question"]
        gt = pair["ground_truth"]
        print(f"  [{i}/{total}] {q[:70]}...")

        if mode == "nova":
            ans, ctxs = nova_run(q)
            time.sleep(7)  # Cohere trial: 10 req/min → 1 req/6s, sleep 7s to be safe
        else:
            ans  = baseline_answer(q)
            ctxs = []  # baseline has no context

        questions.append(q)
        answers.append(ans)
        contexts_list.append(ctxs if ctxs else [""])  # RAGAS requires non-empty list
        ground_truths.append(gt)

    return Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts_list,
        "ground_truth": ground_truths,
    })


# ── Run RAGAS evaluation ──────────────────────────────────────────────────────

def run_evaluation(dataset: Dataset, mode: str) -> dict:
    if mode == "nova":
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    else:
        metrics = [answer_relevancy]
    result = evaluate(dataset, metrics=metrics)
    df = result.to_pandas()
    scores = {}
    for col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if col in df.columns:
            scores[col] = round(float(df[col].dropna().mean()), 4)
    return scores


# ── Pretty print ─────────────────────────────────────────────────────────────

def print_results(nova_scores: dict, baseline_scores: dict):
    sep = "─" * 62
    print(f"\n{sep}")
    print("  NOVA EVALUATION RESULTS")
    print(sep)

    metric_labels = {
        "faithfulness":      "Faithfulness       ",
        "answer_relevancy":  "Answer Relevancy   ",
        "context_precision": "Context Precision  ",
        "context_recall":    "Context Recall     ",
    }

    for key, label in metric_labels.items():
        nova_val     = nova_scores.get(key)
        baseline_val = baseline_scores.get(key)

        nova_str     = f"{nova_val:.4f}" if nova_val is not None else "  N/A "
        baseline_str = f"{baseline_val:.4f}" if baseline_val is not None else "  N/A "

        if nova_val is not None and baseline_val is not None:
            delta = nova_val - baseline_val
            arrow = "▲" if delta >= 0 else "▼"
            delta_str = f"  {arrow} {abs(delta):.4f}"
        else:
            delta_str = ""

        print(f"  {label}  NOVA: {nova_str}   Baseline: {baseline_str}{delta_str}")

    print(sep)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NOVA RAGAS Evaluation")
    parser.add_argument("--limit",  type=int, default=None,
                        help="Only evaluate first N questions (default: all 25)")
    parser.add_argument("--output", type=str, default=None,
                        help="Custom output JSON path")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline evaluation (faster)")
    args = parser.parse_args()

    # Load dataset
    with open(DATASET_PATH) as f:
        qa_pairs = json.load(f)

    if args.limit:
        qa_pairs = qa_pairs[: args.limit]

    print(f"\n{'='*62}")
    print(f"  NOVA — RAGAS Evaluation")
    print(f"  Questions: {len(qa_pairs)}")
    print(f"{'='*62}")

    # ── NOVA pipeline ─────────────────────────────────────────────
    print("\n[1/2] Running NOVA pipeline...")
    nova_dataset = build_ragas_dataset(qa_pairs, mode="nova")

    print("\n  Computing RAGAS metrics for NOVA...")
    nova_scores = run_evaluation(nova_dataset, mode="nova")

    # ── Baseline ──────────────────────────────────────────────────
    baseline_scores = {}
    if not args.skip_baseline:
        print("\n[2/2] Running baseline (vanilla GPT-4o-mini, no RAG)...")
        baseline_dataset = build_ragas_dataset(qa_pairs, mode="baseline")

        print("\n  Computing RAGAS metrics for baseline...")
        baseline_scores = run_evaluation(baseline_dataset, mode="baseline")
    else:
        print("\n[2/2] Skipping baseline.")

    # ── Print summary ─────────────────────────────────────────────
    print_results(nova_scores, baseline_scores)

    # ── Save results ──────────────────────────────────────────────
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = args.output or str(RESULTS_DIR / f"eval_{timestamp}.json")

    output = {
        "timestamp":       timestamp,
        "question_count":  len(qa_pairs),
        "nova":            nova_scores,
        "baseline":        baseline_scores,
        "questions":       [p["question"] for p in qa_pairs],
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved → {out_path}\n")


if __name__ == "__main__":
    main()