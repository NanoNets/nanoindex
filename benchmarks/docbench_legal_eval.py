"""DocBench Legal evaluation using NanoIndex + Claude Sonnet 4.6.

Usage:
    ANTHROPIC_API_KEY=... python benchmarks/docbench_legal_eval.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nanoindex import NanoIndex
from nanoindex.core.llm import LLMClient
from nanoindex.utils.tree_ops import load_tree

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("docbench_legal")

DATA_DIR = Path(__file__).resolve().parent / "data"
TREE_CACHE = Path(__file__).resolve().parent / "docbench_legal_tree_cache"
RESULTS_DIR = Path(__file__).resolve().parent / "results_docbench_legal"

_JUDGE_PROMPT = """\
You are an expert evaluator. Determine whether the AI answer correctly \
answers the query based on the golden answer.

Be lenient on wording — the AI answer is correct if it conveys the same \
meaning, conclusion, or key facts as the golden answer. Minor differences \
in phrasing, additional context, or more detail are acceptable.

Query: {query}
AI Answer: {predicted}
Golden Answer: {gold}

Output ONLY: True or False"""


async def llm_judge(query: str, predicted: str, gold: str, llm: LLMClient) -> bool | None:
    prompt = _JUDGE_PROMPT.format(query=query, predicted=predicted[:2000], gold=gold)
    try:
        resp = await llm.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=64)
        return "true" in resp.strip().lower()
    except Exception as exc:
        logger.warning("Judge failed: %s", exc)
        return None


def load_legal_questions() -> list[dict]:
    """Load all QA pairs for legal PDFs."""
    with open(Path(__file__).resolve().parent / "docbench_legal_pdfs.txt") as f:
        pdfs = [Path(p.strip()) for p in f if p.strip()]

    questions = []
    for pdf_path in pdfs:
        folder = pdf_path.parent
        qa_files = list(folder.glob("*_qa.jsonl"))
        if not qa_files:
            continue

        doc_name = pdf_path.stem
        with open(qa_files[0]) as f:
            for i, line in enumerate(f):
                q = json.loads(line)
                q["doc_name"] = doc_name
                q["pdf_path"] = str(pdf_path)
                q["folder_id"] = folder.name
                q["q_id"] = f"{folder.name}_{i}"
                questions.append(q)

    return questions


async def run_eval(questions: list[dict], ni: NanoIndex, judge_llm: LLMClient):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / "answers.jsonl"
    if results_file.exists():
        results_file.unlink()

    # Group by document
    doc_questions: dict[str, list[dict]] = {}
    for q in questions:
        doc_questions.setdefault(q["doc_name"], []).append(q)

    logger.info("Evaluating %d questions across %d legal documents",
                len(questions), len(doc_questions))

    sem = asyncio.Semaphore(2)
    results = []

    for doc_idx, (doc_name, doc_qs) in enumerate(sorted(doc_questions.items())):
        pdf_path = Path(doc_qs[0]["pdf_path"])
        tree_path = TREE_CACHE / f"{doc_name}.json"

        if not tree_path.exists():
            logger.warning("[%d/%d] %s — no tree, skipping",
                          doc_idx + 1, len(doc_questions), doc_name)
            continue

        tree = load_tree(tree_path)
        logger.info("[%d/%d] %s (%d questions)",
                    doc_idx + 1, len(doc_questions), doc_name[:50], len(doc_qs))

        for q in doc_qs:
            async with sem:
                logger.info("  Q: %s", q["question"][:80])
                t0 = time.time()
                try:
                    ans = await ni.async_ask(
                        q["question"], tree,
                        mode="agentic_vision",
                        pdf_path=pdf_path,
                    )
                    predicted = ans.content
                except Exception as exc:
                    logger.error("    ERROR: %s", str(exc)[:100])
                    predicted = f"ERROR: {exc}"

                elapsed = time.time() - t0

                judge_result = None
                if not predicted.startswith("ERROR"):
                    judge_result = await llm_judge(q["question"], predicted, q["answer"], judge_llm)

                result = {
                    "q_id": q["q_id"],
                    "doc_name": doc_name,
                    "folder_id": q["folder_id"],
                    "question": q["question"],
                    "gold_answer": q["answer"],
                    "predicted": predicted[:3000],
                    "llm_judge": judge_result,
                    "question_type": q.get("type", "unknown"),
                    "evidence": q.get("evidence", ""),
                    "elapsed": round(elapsed, 1),
                }

                with open(results_file, "a") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

                status = "✅" if judge_result else "❌" if judge_result is False else "⚠️"
                logger.info("    %s (%.1fs) | Gold: %s | Pred: %s",
                            status, elapsed, q["answer"][:60], predicted[:60])
                results.append(result)

    return results


def print_summary(results: list[dict]):
    total = len(results)
    if not total:
        print("No results.")
        return

    judged = [r for r in results if r.get("llm_judge") is not None]
    correct = sum(1 for r in judged if r["llm_judge"])
    errors = sum(1 for r in results if r["predicted"].startswith("ERROR"))

    # By question type
    type_stats: dict[str, dict] = {}
    for r in results:
        qt = r.get("question_type", "unknown")
        type_stats.setdefault(qt, {"total": 0, "judged": 0, "correct": 0})
        type_stats[qt]["total"] += 1
        if r.get("llm_judge") is not None:
            type_stats[qt]["judged"] += 1
            if r["llm_judge"]:
                type_stats[qt]["correct"] += 1

    print(f"\n{'='*70}")
    print(f"  DocBench Legal — {total} questions, {len(set(r['doc_name'] for r in results))} documents")
    print(f"{'='*70}")
    if judged:
        print(f"  LLM Judge accuracy:  {correct}/{len(judged)} ({correct/len(judged):.1%})")
    print(f"  Errors/skipped:      {errors}")

    print(f"\n  By question type:")
    for qt, stats in sorted(type_stats.items()):
        if stats["judged"]:
            print(f"    {qt:20s}: {stats['correct']}/{stats['judged']} ({stats['correct']/stats['judged']:.0%})")

    print(f"{'='*70}\n")

    # Save summary
    summary = {
        "total": total,
        "judged": len(judged),
        "correct": correct,
        "accuracy": correct / len(judged) if judged else 0,
        "errors": errors,
        "by_type": {qt: {k: v for k, v in s.items()} for qt, s in type_stats.items()},
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY")
        sys.exit(1)

    questions = load_legal_questions()
    logger.info("Loaded %d legal questions", len(questions))

    ni = NanoIndex(
        nanonets_api_key=os.environ.get("NANONETS_API_KEY"),
        reasoning_llm_model="claude-sonnet-4-6",
        reasoning_llm_api_key=api_key,
    )

    judge_llm = LLMClient(api_key=api_key, model="claude-sonnet-4-6")

    results = asyncio.run(run_eval(questions, ni, judge_llm))
    print_summary(results)


if __name__ == "__main__":
    main()
