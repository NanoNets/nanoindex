"""MMLongBench-Doc evaluation using NanoIndex + Claude Sonnet 4.6.

Usage:
    ANTHROPIC_API_KEY=... python benchmarks/mmlongbench_eval.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from collections import Counter
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
logger = logging.getLogger("mmlongbench")

MMLONGBENCH_DIR = Path(__file__).resolve().parent.parent.parent / "MMLongBench-Doc"
DATA_FILE = MMLONGBENCH_DIR / "data" / "samples.json"
PDF_DIR = MMLONGBENCH_DIR / "data" / "documents"
TREE_CACHE = Path(__file__).resolve().parent / "mmlongbench_tree_cache"
RESULTS_DIR = Path(__file__).resolve().parent / "results_mmlongbench"

_JUDGE_PROMPT = """\
You are an expert evaluator. Determine whether the AI answer correctly \
answers the query based on the golden answer.

Rules:
- Be lenient on wording — correct if same meaning/facts conveyed.
- For numbers: rounding differences are OK (e.g., 3.6 vs 3.57).
- For lists: order doesn't matter, partial overlap is OK if key items match.
- If gold answer is "None" or "Unanswerable": AI must indicate it cannot answer.
- If gold is a number and AI gives same number in different format, that's correct.

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


def load_questions() -> list[dict]:
    with open(DATA_FILE) as f:
        data = json.load(f)
    # Add index
    for i, d in enumerate(data):
        d["q_idx"] = i
    return data


async def run_eval(questions: list[dict], ni: NanoIndex, judge_llm: LLMClient):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / "answers.jsonl"
    if results_file.exists():
        results_file.unlink()

    # Group by document
    doc_questions: dict[str, list[dict]] = {}
    for q in questions:
        doc_questions.setdefault(q["doc_id"], []).append(q)

    logger.info("Evaluating %d questions across %d documents", len(questions), len(doc_questions))

    sem = asyncio.Semaphore(2)
    results = []

    for doc_idx, (doc_id, doc_qs) in enumerate(sorted(doc_questions.items())):
        doc_name = doc_id.replace(".pdf", "")
        pdf_path = PDF_DIR / doc_id
        tree_path = TREE_CACHE / f"{doc_name}.json"

        if not tree_path.exists():
            logger.warning("[%d/%d] %s — no tree, skipping %d Qs",
                          doc_idx + 1, len(doc_questions), doc_name[:50], len(doc_qs))
            continue
        if not pdf_path.exists():
            logger.warning("[%d/%d] %s — no PDF, skipping", doc_idx + 1, len(doc_questions), doc_name[:50])
            continue

        tree = load_tree(tree_path)
        logger.info("[%d/%d] %s (%d Qs)",
                    doc_idx + 1, len(doc_questions), doc_name[:50], len(doc_qs))

        for q in doc_qs:
            async with sem:
                gold = q["answer"]
                is_unanswerable = q.get("answer_format") == "None"

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

                # Judge
                judge_result = None
                if not predicted.startswith("ERROR"):
                    judge_gold = "Unanswerable / None — the document does not contain this information" if is_unanswerable else str(gold)
                    judge_result = await llm_judge(q["question"], predicted, judge_gold, judge_llm)

                # Evidence page hit
                try:
                    evidence_pages = set(eval(q.get("evidence_pages", "[]")))
                except:
                    evidence_pages = set()

                result = {
                    "q_idx": q["q_idx"],
                    "doc_id": doc_id,
                    "doc_type": q.get("doc_type", ""),
                    "question": q["question"],
                    "gold_answer": str(gold),
                    "predicted": predicted[:3000],
                    "llm_judge": judge_result,
                    "answer_format": q.get("answer_format", ""),
                    "evidence_pages": list(evidence_pages),
                    "evidence_sources": q.get("evidence_sources", ""),
                    "elapsed": round(elapsed, 1),
                    "is_unanswerable": is_unanswerable,
                }

                with open(results_file, "a") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

                status = "✅" if judge_result else "❌" if judge_result is False else "⚠️"
                logger.info("    %s (%.1fs) | Gold: %s | Pred: %s",
                            status, elapsed, str(gold)[:60], predicted[:60])
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

    # By answer format
    format_stats: dict[str, dict] = {}
    for r in results:
        af = r.get("answer_format", "?")
        format_stats.setdefault(af, {"total": 0, "judged": 0, "correct": 0})
        format_stats[af]["total"] += 1
        if r.get("llm_judge") is not None:
            format_stats[af]["judged"] += 1
            if r["llm_judge"]:
                format_stats[af]["correct"] += 1

    # By doc type
    type_stats: dict[str, dict] = {}
    for r in results:
        dt = r.get("doc_type", "?")
        type_stats.setdefault(dt, {"total": 0, "judged": 0, "correct": 0})
        type_stats[dt]["total"] += 1
        if r.get("llm_judge") is not None:
            type_stats[dt]["judged"] += 1
            if r["llm_judge"]:
                type_stats[dt]["correct"] += 1

    # By evidence source
    source_stats: dict[str, dict] = {}
    for r in results:
        try:
            sources = eval(r.get("evidence_sources", "[]"))
        except:
            sources = []
        for s in sources:
            source_stats.setdefault(s, {"total": 0, "judged": 0, "correct": 0})
            source_stats[s]["total"] += 1
            if r.get("llm_judge") is not None:
                source_stats[s]["judged"] += 1
                if r["llm_judge"]:
                    source_stats[s]["correct"] += 1

    print(f"\n{'='*70}")
    print(f"  MMLongBench-Doc — {total} questions, {len(set(r['doc_id'] for r in results))} documents")
    print(f"{'='*70}")
    if judged:
        print(f"  LLM Judge accuracy:  {correct}/{len(judged)} ({correct/len(judged):.1%})")
    print(f"  Errors/skipped:      {errors}")

    print(f"\n  By answer format:")
    for af, stats in sorted(format_stats.items()):
        if stats["judged"]:
            print(f"    {af:10s}: {stats['correct']}/{stats['judged']} ({stats['correct']/stats['judged']:.0%}) [{stats['total']} Qs]")

    print(f"\n  By document type:")
    for dt, stats in sorted(type_stats.items(), key=lambda x: -x[1]["total"]):
        if stats["judged"]:
            print(f"    {dt:35s}: {stats['correct']}/{stats['judged']} ({stats['correct']/stats['judged']:.0%}) [{stats['total']} Qs]")

    print(f"\n  By evidence source:")
    for s, stats in sorted(source_stats.items(), key=lambda x: -x[1]["total"]):
        if stats["judged"]:
            print(f"    {s:30s}: {stats['correct']}/{stats['judged']} ({stats['correct']/stats['judged']:.0%}) [{stats['total']} Qs]")

    print(f"{'='*70}\n")

    summary = {
        "total": total, "judged": len(judged), "correct": correct,
        "accuracy": correct / len(judged) if judged else 0, "errors": errors,
        "by_format": format_stats, "by_doc_type": type_stats, "by_source": source_stats,
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY")
        sys.exit(1)

    questions = load_questions()
    logger.info("Loaded %d questions across %d documents",
                len(questions), len(set(q["doc_id"] for q in questions)))

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
