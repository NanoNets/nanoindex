"""Test only the questions that failed in the previous eval run."""
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
logger = logging.getLogger("test_failures")

FINANCEBENCH_DIR = Path(__file__).resolve().parent.parent.parent / "financebench"
QUESTIONS_PATH = FINANCEBENCH_DIR / "data" / "financebench_open_source.jsonl"
PDFS_DIR = FINANCEBENCH_DIR / "pdfs"
TREE_CACHE = Path(__file__).resolve().parent / "cache_v3"

# The 17 questions that failed + 3 errors
FAILED_IDS = [
    "financebench_id_00807",  # 3M quick ratio
    "financebench_id_10420",  # AES ROA
    "financebench_id_01930",  # AMCOR real sales change
    "financebench_id_00799",  # AMCOR quick ratio
    "financebench_id_00460",  # Best Buy store count
    "financebench_id_01902",  # Best Buy product category
    "financebench_id_00005",  # Corning working capital
    "financebench_id_00790",  # CVS capital-intensive
    "financebench_id_00651",  # JnJ EPS acceleration
    "financebench_id_00711",  # JnJ inventory turnover
    "financebench_id_02119",  # JPMorgan liquidation
    "financebench_id_00382",  # MGM EBITDAR region
    "financebench_id_01911",  # MGM interest coverage
    "financebench_id_00080",  # PayPal working capital
    "financebench_id_02416",  # Pfizer acquisitions
    "financebench_id_02419",  # Pfizer spin-off
    "financebench_id_00606",  # Ulta Beauty wages
    # Errors (AMCOR vision too large)
    "financebench_id_00168",  # AMCOR restructuring
    "financebench_id_04981",  # AMCOR industry
    "financebench_id_01898",  # AMCOR acquisitions
]


async def _judge(query: str, predicted: str, gold: str, llm: LLMClient) -> bool | None:
    prompt = (
        "You are an expert evaluator. Determine if the AI answer correctly answers "
        "the query based on the golden answer.\n\n"
        "Numerical Accuracy: Rounding differences should be ignored. "
        "Fractions/percentages/numerics are considered similar (11/14 ≈ 79%).\n"
        "The AI answer is correct if it conveys the same meaning, conclusion, or "
        "rationale as the golden answer, or is a superset of it.\n\n"
        f"Query: {query}\nAI Answer: {predicted}\nGolden Answer: {gold}\n\n"
        "Output ONLY: True or False"
    )
    try:
        resp = await llm.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=64)
        return "true" in resp.strip().lower()
    except Exception as exc:
        logger.warning("Judge failed: %s", exc)
        return None


async def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY")
        sys.exit(1)

    # Load questions
    with open(QUESTIONS_PATH) as f:
        all_questions = [json.loads(line) for line in f]

    failed_set = set(FAILED_IDS)
    questions = [q for q in all_questions if q["financebench_id"] in failed_set]
    logger.info("Testing %d previously-failed questions", len(questions))

    ni = NanoIndex(
        nanonets_api_key=os.environ.get("NANONETS_API_KEY"),
        reasoning_llm_model="claude-sonnet-4-6",
        reasoning_llm_api_key=api_key,
    )

    judge_llm = LLMClient(api_key=api_key, model="claude-sonnet-4-6")

    results = []
    for idx, q in enumerate(questions):
        qid = q["financebench_id"]
        doc_name = q["doc_name"]
        question = q["question"]
        gold = q["answer"]

        pdf_path = PDFS_DIR / f"{doc_name}.pdf"
        tree_path = TREE_CACHE / f"{doc_name}.json"

        if not tree_path.exists():
            logger.warning("[%d/%d] %s — tree not cached, skipping", idx+1, len(questions), doc_name)
            continue
        if not pdf_path.exists():
            logger.warning("[%d/%d] %s — PDF not found, skipping", idx+1, len(questions), doc_name)
            continue

        tree = load_tree(tree_path)
        logger.info("[%d/%d] %s | Q: %s", idx+1, len(questions), doc_name, question[:80])

        t0 = time.time()
        try:
            ans = await ni.async_ask(
                question, tree,
                mode="agentic_vision",
                pdf_path=pdf_path,
                include_metadata=True,
            )
            predicted = ans.content
        except Exception as exc:
            logger.error("  ERROR: %s", exc)
            predicted = f"ERROR: {exc}"

        elapsed = time.time() - t0

        # Judge
        judge_result = None
        if not predicted.startswith("ERROR"):
            judge_result = await _judge(question, predicted, gold, judge_llm)

        status = "✅" if judge_result else "❌"
        logger.info("  %s judge=%s (%.1fs)", status, judge_result, elapsed)
        logger.info("  Gold: %s", gold[:100])
        logger.info("  Pred: %s", predicted[:100])

        results.append({
            "id": qid,
            "doc_name": doc_name,
            "question": question[:100],
            "gold": gold[:150],
            "predicted": predicted[:300],
            "judge": judge_result,
            "time": round(elapsed, 1),
        })

    # Summary
    judged = [r for r in results if r["judge"] is not None]
    correct = sum(1 for r in judged if r["judge"])
    errors = sum(1 for r in results if r["predicted"].startswith("ERROR"))

    print(f"\n{'='*70}")
    print(f"  Previously-Failed Questions Re-test: {len(results)} questions")
    print(f"{'='*70}")
    print(f"  Judge correct: {correct}/{len(judged)} ({correct/len(judged)*100:.1f}%)" if judged else "  No judged results")
    print(f"  Errors: {errors}")
    print(f"  Previously: 0/17 correct (+ 3 errors)")
    print(f"  Now fixed:  {correct}/{len(judged)} correct")
    print(f"{'='*70}")

    # Show remaining failures
    still_failing = [r for r in results if r["judge"] == False or r["predicted"].startswith("ERROR")]
    if still_failing:
        print(f"\n  Still failing ({len(still_failing)}):")
        for r in still_failing:
            print(f"    {r['id']}: {r['question'][:60]}")
            print(f"      Gold: {r['gold'][:80]}")
            print(f"      Pred: {r['predicted'][:80]}")
            print()


if __name__ == "__main__":
    asyncio.run(main())
