"""FinanceBench evaluation script for NanoIndex.

Usage:
    # Run on a small sample (default 10 questions)
    python benchmarks/financebench_eval.py --limit 10

    # Run the full benchmark
    python benchmarks/financebench_eval.py

    # Resume from cached trees (skip re-indexing)
    python benchmarks/financebench_eval.py --cache-dir benchmarks/cache

Requires:
    - FinanceBench repo cloned alongside NanoIndex:
      ../financebench/pdfs/  and  ../financebench/data/
    - NANONETS_API_KEY environment variable set
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nanoindex import NanoIndex
from nanoindex.utils.tree_ops import save_tree, load_tree, iter_nodes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("financebench_eval")

FINANCEBENCH_DIR = Path(__file__).resolve().parent.parent.parent / "financebench"
QUESTIONS_PATH = FINANCEBENCH_DIR / "data" / "financebench_open_source.jsonl"
PDFS_DIR = FINANCEBENCH_DIR / "pdfs"


def load_questions(limit: int | None = None) -> list[dict]:
    with open(QUESTIONS_PATH) as f:
        questions = [json.loads(line) for line in f]
    if limit:
        questions = questions[:limit]
    return questions


def answer_matches(predicted: str, gold: str) -> dict:
    """Simple matching heuristics. Returns a dict with match info."""
    import re

    pred_norm = predicted.lower().strip().replace(",", "")
    gold_norm = gold.lower().strip().replace(",", "")

    exact_match = gold_norm in pred_norm

    pred_nums = set(re.findall(r"[\d]+\.?\d*", pred_norm))
    gold_nums = set(re.findall(r"[\d]+\.?\d*", gold_norm))

    number_overlap = bool(pred_nums & gold_nums) if gold_nums else False

    def _normalize_num(n: str) -> str:
        """Strip trailing .00 / .0 so 1577.00 matches 1577."""
        if "." in n:
            n = n.rstrip("0").rstrip(".")
        return n

    gold_key_nums = re.findall(r"\$?([\d]+\.?\d*)", gold_norm)
    gold_key_nums_clean = [_normalize_num(n) for n in gold_key_nums]
    pred_nums_clean = {_normalize_num(n) for n in pred_nums}
    key_num_found = any(n in pred_norm or n in pred_nums_clean for n in gold_key_nums_clean) if gold_key_nums_clean else False

    return {
        "exact_match": exact_match,
        "number_overlap": number_overlap,
        "key_number_found": key_num_found,
        "score": 1.0 if exact_match else (0.75 if key_num_found else (0.5 if number_overlap else 0.0)),
    }


async def run_eval(
    questions: list[dict],
    cache_dir: Path,
    ni: NanoIndex,
) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    results = []

    # Group questions by document
    doc_questions: dict[str, list[dict]] = {}
    for q in questions:
        doc_questions.setdefault(q["doc_name"], []).append(q)

    logger.info("Evaluating %d questions across %d documents", len(questions), len(doc_questions))

    for doc_idx, (doc_name, doc_qs) in enumerate(doc_questions.items()):
        pdf_path = PDFS_DIR / f"{doc_name}.pdf"
        tree_cache = cache_dir / f"{doc_name}.json"

        if not pdf_path.exists():
            logger.warning("PDF not found: %s — skipping %d questions", pdf_path, len(doc_qs))
            for q in doc_qs:
                results.append({
                    "id": q["financebench_id"],
                    "question": q["question"],
                    "gold_answer": q["answer"],
                    "predicted": "SKIPPED — PDF not found",
                    "score": 0.0,
                    "doc_name": doc_name,
                })
            continue

        # Index document (or load from cache)
        if tree_cache.exists():
            logger.info("[%d/%d] Loading cached tree: %s", doc_idx + 1, len(doc_questions), doc_name)
            tree = load_tree(tree_cache)
        else:
            logger.info("[%d/%d] Indexing: %s (%s)", doc_idx + 1, len(doc_questions), doc_name, pdf_path.name)
            t0 = time.time()
            try:
                tree = await ni.async_index(str(pdf_path))
                elapsed = time.time() - t0
                node_count = len(list(iter_nodes(tree.structure)))
                logger.info("  Indexed in %.1fs — %d nodes", elapsed, node_count)
                save_tree(tree, tree_cache)
            except Exception as exc:
                logger.error("  Indexing failed: %s", exc)
                for q in doc_qs:
                    results.append({
                        "id": q["financebench_id"],
                        "question": q["question"],
                        "gold_answer": q["answer"],
                        "predicted": f"INDEXING FAILED: {exc}",
                        "score": 0.0,
                        "doc_name": doc_name,
                    })
                continue

        # Answer each question for this document
        for q in doc_qs:
            logger.info("  Q: %s", q["question"][:80])
            try:
                ans = await ni.async_ask(q["question"], tree, include_metadata=True)
                predicted = ans.content
                citations = [
                    {"title": c.title, "pages": c.pages}
                    for c in ans.citations
                ]
            except Exception as exc:
                logger.error("    Answer failed: %s", exc)
                predicted = f"ERROR: {exc}"
                citations = []

            match = answer_matches(predicted, q["answer"])

            # Check if evidence page was found
            evidence_pages = set()
            for ev in q.get("evidence", []):
                pg = ev.get("evidence_page_num")
                if pg is not None:
                    evidence_pages.add(pg + 1)  # FinanceBench is 0-indexed

            cited_pages = set()
            for c in citations:
                cited_pages.update(c.get("pages", []))

            page_hit = bool(evidence_pages & cited_pages) if evidence_pages else None

            result = {
                "id": q["financebench_id"],
                "question": q["question"],
                "gold_answer": q["answer"],
                "predicted": predicted[:500],
                "score": match["score"],
                "exact_match": match["exact_match"],
                "key_number_found": match["key_number_found"],
                "evidence_page_hit": page_hit,
                "question_type": q.get("question_type"),
                "question_reasoning": q.get("question_reasoning"),
                "doc_name": doc_name,
                "citations": citations,
            }
            results.append(result)

            status = "✅" if match["score"] >= 0.75 else "⚠️" if match["score"] > 0 else "❌"
            logger.info("    %s score=%.2f | Gold: %s | Pred: %s",
                        status, match["score"], q["answer"][:80], predicted[:80])

    return results


def print_summary(results: list[dict]):
    total = len(results)
    if total == 0:
        print("No results.")
        return

    avg_score = sum(r["score"] for r in results) / total
    exact = sum(1 for r in results if r.get("exact_match"))
    key_num = sum(1 for r in results if r.get("key_number_found"))
    page_hits = [r for r in results if r.get("evidence_page_hit") is not None]
    page_hit_rate = sum(1 for r in page_hits if r["evidence_page_hit"]) / len(page_hits) if page_hits else 0

    print(f"\n{'='*60}")
    print(f"  FinanceBench Evaluation — {total} questions")
    print(f"{'='*60}")
    print(f"  Average score:       {avg_score:.1%}")
    print(f"  Exact match:         {exact}/{total} ({exact/total:.1%})")
    print(f"  Key number found:    {key_num}/{total} ({key_num/total:.1%})")
    print(f"  Evidence page hit:   {sum(1 for r in page_hits if r['evidence_page_hit'])}/{len(page_hits)} ({page_hit_rate:.1%})")

    # By question type
    from collections import defaultdict
    by_type = defaultdict(list)
    for r in results:
        by_type[r.get("question_type", "unknown")].append(r)

    print(f"\n  By question type:")
    for qtype, rs in sorted(by_type.items()):
        avg = sum(r["score"] for r in rs) / len(rs)
        print(f"    {qtype:25s}: {avg:.1%} ({len(rs)} questions)")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="FinanceBench evaluation for NanoIndex")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--cache-dir", type=str, default="benchmarks/cache", help="Directory for cached trees")
    parser.add_argument("--output", type=str, default="benchmarks/results.jsonl", help="Output results file")
    args = parser.parse_args()

    if not QUESTIONS_PATH.exists():
        print(f"ERROR: FinanceBench data not found at {QUESTIONS_PATH}")
        print("Clone it: git clone https://github.com/patronus-ai/financebench.git")
        sys.exit(1)

    questions = load_questions(args.limit)
    logger.info("Loaded %d questions", len(questions))

    ni = NanoIndex(nanonets_api_key=os.environ.get("NANONETS_API_KEY"))
    ni.config.confidence_threshold = 0.0

    results = asyncio.run(run_eval(questions, Path(args.cache_dir), ni))

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Results saved to %s", output_path)

    print_summary(results)


if __name__ == "__main__":
    main()
