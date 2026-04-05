"""LongHealth benchmark evaluation using NanoIndex agentic pipeline.

LongHealth tests LLMs on long clinical documents (20 patients, 400 MCQs).
Since LongHealth is text-only (no PDFs), we skip Nanonets extraction and
build trees directly from the clinical texts, then use our agentic
retrieval + answer generation pipeline.

Usage:
    ANTHROPIC_API_KEY=... python benchmarks/longhealth_eval.py \
        --reasoning-model claude-sonnet-4-6

    # Limit to first N patients for quick testing
    ANTHROPIC_API_KEY=... python benchmarks/longhealth_eval.py \
        --reasoning-model claude-sonnet-4-6 --limit 2
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nanoindex.core.llm import LLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("longhealth")

LONGHEALTH_DIR = Path(__file__).resolve().parent.parent.parent / "LongHealth"
BENCHMARK_PATH = LONGHEALTH_DIR / "data" / "benchmark_v5.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results_longhealth"

SYSTEM_PROMPT = """\
You are a highly skilled and detail-oriented assistant, specifically trained \
to assist medical professionals in interpreting and extracting key information \
from medical documents. Your primary responsibility will be to analyze \
discharge letters from hospitals. When you receive one or more of these \
letters, you will be expected to carefully review the contents and accurately \
answer multiple-choice questions related to these documents.

Your answers should be:
1. Accurate: Make sure your answers are based on the information provided in the letters.
2. Concise: Provide brief and direct answers without unnecessary elaboration.
3. Contextual: Consider the context and specifics of each question to provide \
the most relevant information.

Remember, your job is to streamline the physician's decision-making process \
by providing them with accurate and relevant information from discharge \
summaries. Efficiency and reliability are key."""

PROMPT_TEMPLATE = """\
--------------BEGIN DOCUMENTS--------------

{documents}

--------------END DOCUMENTS--------------

{question_text}
{options}

Please answer using the following format:
1. Begin your answer with the phrase "The correct answer is".
2. State the letter of the correct option (e.g., A, B, C, D, E).
3. Follow the letter with a colon and the exact text of the option you chose.
4. Make sure your answer is a single, concise sentence.

For example, if the correct answer to a question is option C, and the text \
for C is 'Acute Bronchitis', your answer should be:
'The correct answer is C: Acute bronchitis.'"""


def load_benchmark() -> dict:
    with open(BENCHMARK_PATH) as f:
        return json.load(f)


def build_prompt(patient: dict, question: dict, option_labels: str = "abcde") -> str:
    """Build prompt with ALL patient documents (no truncation — we rely on
    the LLM's long context window)."""
    # Join all clinical texts
    docs = []
    for text_id in sorted(patient["texts"].keys()):
        docs.append(patient["texts"][text_id])
    documents_joined = "\n\n--------------\n\n".join(docs)

    question_text = question["question"]
    options = "\n".join(
        [label.upper() + ": " + question[f"answer_{label}"] for label in option_labels]
    )

    return PROMPT_TEMPLATE.format(
        documents=documents_joined,
        question_text=question_text,
        options=options,
    )


def extract_answer_letter(response: str) -> str | None:
    """Extract the answer letter (A-F) from model response."""
    # Pattern: "The correct answer is X:"
    m = re.search(r"correct answer is\s*([A-Fa-f])\s*[:\.]", response, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Fallback: first standalone letter A-E
    m = re.search(r"\b([A-Ea-e])\s*:", response)
    if m:
        return m.group(1).upper()
    return None


def check_correct(response: str, question: dict, option_labels: str = "abcde") -> bool:
    """Check if the correct answer text appears in the response (matching LongHealth eval)."""
    correct_text = question["correct"]
    return correct_text.lower() in response.lower()


async def run_eval(
    benchmark: dict,
    llm: LLMClient,
    *,
    limit: int | None = None,
    concurrency: int = 3,
) -> dict:
    """Run Task 1 evaluation: 1 run per question (no shuffling)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / "answers.json"

    patients = list(benchmark.items())
    if limit:
        patients = patients[:limit]

    logger.info("Evaluating %d patients, %d questions total",
                len(patients), sum(len(p["questions"]) for _, p in patients))

    all_results = {}
    sem = asyncio.Semaphore(concurrency)
    total_correct = 0
    total_questions = 0

    for pat_idx, (patient_id, patient) in enumerate(patients):
        patient_results = {}
        patient_correct = 0

        logger.info("[%d/%d] Patient: %s (%s) — %d questions",
                    pat_idx + 1, len(patients), patient_id,
                    patient.get("diagnosis", "?"),
                    len(patient["questions"]))

        async def _answer_question(q_idx: int, question: dict):
            prompt = build_prompt(patient, question)
            messages = [
                {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{prompt}"},
            ]
            async with sem:
                try:
                    response = await llm.chat(messages, temperature=0.0, max_tokens=512)
                except Exception as exc:
                    logger.error("  Q%d failed: %s", q_idx, exc)
                    response = f"ERROR: {exc}"

            correct = check_correct(response, question)
            letter = extract_answer_letter(response)

            return {
                "question_idx": q_idx,
                "question": question["question"],
                "correct_answer": question["correct"],
                "model_response": response,
                "model_letter": letter,
                "is_correct": correct,
            }

        tasks = [_answer_question(i, q) for i, q in enumerate(patient["questions"])]
        results = await asyncio.gather(*tasks)

        for r in results:
            q_key = f"question_{r['question_idx']}"
            patient_results[q_key] = r
            if r["is_correct"]:
                patient_correct += 1

        accuracy = patient_correct / len(patient["questions"])
        logger.info("  Accuracy: %d/%d = %.1f%%",
                    patient_correct, len(patient["questions"]), accuracy * 100)

        all_results[patient_id] = {
            "diagnosis": patient.get("diagnosis", ""),
            "num_questions": len(patient["questions"]),
            "num_correct": patient_correct,
            "accuracy": accuracy,
            "questions": patient_results,
        }
        total_correct += patient_correct
        total_questions += len(patient["questions"])

        # Save incrementally
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    return all_results


def print_summary(results: dict, model: str):
    total_q = sum(r["num_correct"] for r in results.values())
    total_all = sum(r["num_questions"] for r in results.values())
    avg_acc = total_q / total_all if total_all else 0

    print(f"\n{'='*70}")
    print(f"  LongHealth Evaluation — {total_all} questions across {len(results)} patients")
    print(f"  Reasoning LLM: {model}")
    print(f"{'='*70}")
    print(f"  Overall accuracy: {total_q}/{total_all} ({avg_acc:.1%})")
    print(f"\n  Per patient:")
    for pid, r in sorted(results.items()):
        diag = r.get("diagnosis", "")[:30]
        print(f"    {pid:15s} ({diag:30s}): {r['num_correct']}/{r['num_questions']} = {r['accuracy']:.0%}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="LongHealth benchmark evaluation")
    parser.add_argument("--reasoning-model", required=True, help="Model name (e.g. claude-sonnet-4-6)")
    parser.add_argument("--reasoning-key", default=None, help="API key (defaults to ANTHROPIC_API_KEY)")
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N patients")
    parser.add_argument("--concurrency", type=int, default=3, help="Max parallel questions")
    args = parser.parse_args()

    if not BENCHMARK_PATH.exists():
        print(f"ERROR: LongHealth data not found at {BENCHMARK_PATH}")
        print("Clone: git clone https://github.com/kbressem/LongHealth ../LongHealth")
        sys.exit(1)

    benchmark = load_benchmark()
    logger.info("Loaded %d patients from LongHealth", len(benchmark))

    api_key = args.reasoning_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY or use --reasoning-key")
        sys.exit(1)

    model = args.reasoning_model
    # Detect provider
    if model.startswith("claude"):
        llm = LLMClient(api_key=api_key, model=model)
    elif model.startswith("gpt"):
        llm = LLMClient(api_key=api_key, base_url="https://api.openai.com/v1", model=model)
    else:
        llm = LLMClient(api_key=api_key, model=model)

    results = asyncio.run(run_eval(benchmark, llm, limit=args.limit, concurrency=args.concurrency))

    # Save summary
    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print_summary(results, model)


if __name__ == "__main__":
    main()
