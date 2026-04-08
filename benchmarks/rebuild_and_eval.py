"""Rebuild trees and run evals for FinanceBench + MMLongBench using Gemini 3.1 Pro.

Usage:
    NANONETS_API_KEY=... GOOGLE_API_KEY=... python benchmarks/rebuild_and_eval.py --benchmark financebench
    NANONETS_API_KEY=... GOOGLE_API_KEY=... python benchmarks/rebuild_and_eval.py --benchmark mmlongbench
    NANONETS_API_KEY=... GOOGLE_API_KEY=... python benchmarks/rebuild_and_eval.py --skip-rebuild --benchmark financebench
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
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rebuild_eval")

# Paths
FINANCEBENCH_DIR = Path(__file__).resolve().parent.parent.parent / "financebench"
FB_QUESTIONS = FINANCEBENCH_DIR / "data" / "financebench_open_source.jsonl"
FB_PDFS = FINANCEBENCH_DIR / "pdfs"
FB_TREE_CACHE = Path("benchmarks/cache_v5")
FB_RESULTS = Path("benchmarks/results_gemini_v1")

MMLB_DIR = Path(__file__).resolve().parent.parent.parent / "MMLongBench-Doc"
MMLB_DATA = MMLB_DIR / "data" / "samples.json"
MMLB_PDFS = MMLB_DIR / "data" / "documents"
MMLB_TREE_CACHE = Path("benchmarks/mmlongbench_cache_v2")
MMLB_RESULTS = Path("benchmarks/results_mmlongbench_gemini")

LLM = "gemini:gemini-3.1-pro-preview"


# ------------------------------------------------------------------
# Tree rebuild
# ------------------------------------------------------------------

async def rebuild_trees(benchmark: str, ni, limit: int | None = None):
    """Rebuild trees from PDFs using latest pipeline."""
    if benchmark == "financebench":
        pdf_dir, cache_dir = FB_PDFS, FB_TREE_CACHE
        # Get unique doc names from questions
        questions = [json.loads(l) for l in open(FB_QUESTIONS)]
        doc_names = sorted(set(q["doc_name"] for q in questions))
    else:
        pdf_dir, cache_dir = MMLB_PDFS, MMLB_TREE_CACHE
        samples = json.load(open(MMLB_DATA))
        doc_names = sorted(set(s["doc_id"].replace(".pdf", "") for s in samples))

    cache_dir.mkdir(parents=True, exist_ok=True)

    if limit:
        doc_names = doc_names[:limit]

    logger.info("Rebuilding %d trees for %s...", len(doc_names), benchmark)

    for i, doc_name in enumerate(doc_names):
        cache_file = cache_dir / f"{doc_name}.json"
        if cache_file.exists():
            logger.info("[%d/%d] Cached: %s", i + 1, len(doc_names), doc_name)
            continue

        # Find PDF
        if benchmark == "financebench":
            pdf_path = pdf_dir / f"{doc_name}.pdf"
        else:
            pdf_path = pdf_dir / f"{doc_name}.pdf"
            if not pdf_path.exists():
                pdf_path = pdf_dir / doc_name  # some have .pdf in doc_id

        if not pdf_path.exists():
            logger.warning("[%d/%d] PDF not found: %s", i + 1, len(doc_names), pdf_path)
            continue

        logger.info("[%d/%d] Indexing: %s", i + 1, len(doc_names), doc_name)
        t0 = time.time()
        try:
            tree = await ni.async_index(str(pdf_path))
            from nanoindex.utils.tree_ops import save_tree, iter_nodes
            save_tree(tree, cache_file)
            nodes = len(list(iter_nodes(tree.structure)))
            logger.info("  Done in %.1fs — %d nodes", time.time() - t0, nodes)
        except Exception as exc:
            logger.error("  FAILED: %s", exc)


# ------------------------------------------------------------------
# Eval helpers
# ------------------------------------------------------------------

def heuristic_score(predicted: str, gold: str) -> dict:
    pred_norm = predicted.lower().strip().replace(",", "")
    gold_norm = gold.lower().strip().replace(",", "")
    exact = gold_norm in pred_norm
    pred_nums = set(re.findall(r"[\d]+\.?\d*", pred_norm))
    gold_nums = set(re.findall(r"[\d]+\.?\d*", gold_norm))
    overlap = bool(pred_nums & gold_nums) if gold_nums else False
    gold_key = re.findall(r"\$?([\d]+\.?\d*)", gold_norm)
    gold_clean = [n.rstrip("0").rstrip(".") if "." in n else n for n in gold_key]
    pred_clean = {n.rstrip("0").rstrip(".") if "." in n else n for n in pred_nums}
    key_found = any(n in pred_norm or n in pred_clean for n in gold_clean) if gold_clean else False
    return {
        "exact_match": exact,
        "key_number_found": key_found,
        "heuristic_score": 1.0 if exact else (0.75 if key_found else (0.5 if overlap else 0.0)),
    }


async def llm_judge(query, gold, predicted, llm):
    prompt = f"""Compare. Respond ONLY {{"correct": true/false, "reasoning": "one sentence"}}
Question: {query}
Gold: {gold}
Predicted: {predicted[:2000]}"""
    try:
        resp = await llm.chat([{"role": "user", "content": prompt}], temperature=0.0)
        resp = re.sub(r"^```(?:json)?\s*", "", resp.strip())
        resp = re.sub(r"\s*```$", "", resp)
        data = json.loads(resp)
        return data.get("correct"), data.get("reasoning", "")
    except Exception:
        return None, "judge failed"


def append_result(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def load_completed(path):
    if not path.exists():
        return set()
    return {json.loads(l)["id"] for l in open(path) if l.strip()}


# ------------------------------------------------------------------
# FinanceBench eval
# ------------------------------------------------------------------

async def run_financebench(ni, limit=None):
    from nanoindex.utils.tree_ops import load_tree
    from nanoindex.core.title_disambiguator import disambiguate_titles

    questions = [json.loads(l) for l in open(FB_QUESTIONS)]
    if limit:
        questions = questions[:limit]

    results_file = FB_RESULTS / "answers.jsonl"
    completed = load_completed(results_file)
    llm = ni._get_reasoning_llm()

    doc_questions = defaultdict(list)
    for q in questions:
        doc_questions[q["doc_name"]].append(q)

    logger.info("FinanceBench: %d questions, %d docs, %d already done",
                len(questions), len(doc_questions), len(completed))

    correct = 0
    total = 0

    for doc_idx, (doc_name, doc_qs) in enumerate(doc_questions.items()):
        tree_file = FB_TREE_CACHE / f"{doc_name}.json"
        pdf_path = FB_PDFS / f"{doc_name}.pdf"

        if not tree_file.exists():
            logger.warning("No tree for %s, skipping", doc_name)
            continue

        tree = load_tree(tree_file)
        tree = disambiguate_titles(tree)

        logger.info("[%d/%d] %s: %d questions",
                    doc_idx + 1, len(doc_questions), doc_name, len(doc_qs))

        for q in doc_qs:
            qid = q["financebench_id"]
            if qid in completed:
                continue

            t0 = time.time()
            try:
                ans = await ni.async_ask(
                    q["question"], tree, mode="agentic_vision",
                    pdf_path=str(pdf_path) if pdf_path.exists() else None,
                )
                predicted = ans.content
            except Exception as exc:
                predicted = f"ERROR: {exc}"
            elapsed = time.time() - t0

            judge_correct, judge_reason = await llm_judge(
                q["question"], q["answer"], predicted, llm
            )

            h = heuristic_score(predicted, q["answer"])
            score = 1.0 if judge_correct else (0.0 if judge_correct is False else h["heuristic_score"])

            result = {
                "id": qid,
                "question": q["question"],
                "gold_answer": q["answer"],
                "predicted": predicted,
                "llm_judge": judge_correct,
                "llm_reasoning": judge_reason,
                "heuristic_score": h["heuristic_score"],
                "score": score,
                "doc_name": doc_name,
                "time_s": round(elapsed, 1),
            }
            append_result(results_file, result)
            total += 1
            if judge_correct:
                correct += 1

            status = "+" if judge_correct else "-"
            logger.info("  %s %.1fs | %s | Gold: %s",
                        status, elapsed, qid, q["answer"][:50])

            if total % 10 == 0:
                logger.info("  Progress: %d/%d correct (%d%%)", correct, total, correct * 100 // total if total else 0)

    logger.info("\nFinanceBench FINAL: %d/%d (%d%%)", correct, total, correct * 100 // total if total else 0)


# ------------------------------------------------------------------
# MMLongBench eval
# ------------------------------------------------------------------

async def run_mmlongbench(ni, limit=None):
    from nanoindex.utils.tree_ops import load_tree
    from nanoindex.core.title_disambiguator import disambiguate_titles

    samples = json.load(open(MMLB_DATA))
    if limit:
        samples = samples[:limit]

    results_file = MMLB_RESULTS / "answers.jsonl"
    completed = load_completed(results_file)
    llm = ni._get_reasoning_llm()

    # Group by doc
    doc_questions = defaultdict(list)
    for i, s in enumerate(samples):
        s["_id"] = f"{s['doc_id']}_{i}"
        doc_questions[s["doc_id"]].append(s)

    logger.info("MMLongBench: %d questions, %d docs, %d already done",
                len(samples), len(doc_questions), len(completed))

    correct = 0
    total = 0

    for doc_idx, (doc_id, doc_qs) in enumerate(doc_questions.items()):
        doc_name = doc_id.replace(".pdf", "")
        tree_file = MMLB_TREE_CACHE / f"{doc_name}.json"
        pdf_path = MMLB_PDFS / doc_id

        if not tree_file.exists():
            logger.warning("No tree for %s, skipping", doc_name)
            continue

        tree = load_tree(tree_file)
        tree = disambiguate_titles(tree)

        logger.info("[%d/%d] %s: %d questions", doc_idx + 1, len(doc_questions), doc_name[:40], len(doc_qs))

        for q in doc_qs:
            if q["_id"] in completed:
                continue

            t0 = time.time()
            try:
                ans = await ni.async_ask(
                    q["question"], tree, mode="agentic_vision",
                    pdf_path=str(pdf_path) if pdf_path.exists() else None,
                )
                predicted = ans.content
            except Exception as exc:
                predicted = f"ERROR: {exc}"
            elapsed = time.time() - t0

            judge_correct, judge_reason = await llm_judge(
                q["question"], q["answer"], predicted, llm
            )

            score = 1.0 if judge_correct else 0.0

            result = {
                "id": q["_id"],
                "question": q["question"],
                "gold_answer": q["answer"],
                "predicted": predicted,
                "llm_judge": judge_correct,
                "llm_reasoning": judge_reason,
                "score": score,
                "doc_id": doc_id,
                "doc_type": q.get("doc_type", ""),
                "answer_format": q.get("answer_format", ""),
                "evidence_sources": q.get("evidence_sources", ""),
                "time_s": round(elapsed, 1),
            }
            append_result(results_file, result)
            total += 1
            if judge_correct:
                correct += 1

            status = "+" if judge_correct else "-"
            logger.info("  %s %.1fs | Gold: %s", status, elapsed, q["answer"][:50])

            if total % 10 == 0:
                logger.info("  Progress: %d/%d correct (%d%%)", correct, total, correct * 100 // total if total else 0)

    logger.info("\nMMLongBench FINAL: %d/%d (%d%%)", correct, total, correct * 100 // total if total else 0)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=["financebench", "mmlongbench"])
    parser.add_argument("--skip-rebuild", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    from nanoindex import NanoIndex
    ni = NanoIndex(llm=LLM)
    # Skip graph building during indexing (saves ~5 min per doc on CPU)
    ni.config.build_graph = False

    async def run():
        if not args.skip_rebuild:
            await rebuild_trees(args.benchmark, ni, args.limit)

        if args.benchmark == "financebench":
            await run_financebench(ni, args.limit)
        else:
            await run_mmlongbench(ni, args.limit)

    asyncio.run(run())


if __name__ == "__main__":
    main()
