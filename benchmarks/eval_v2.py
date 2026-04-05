"""FinanceBench evaluation with the improved tree builder.

Caches extraction results separately from trees so tree-builder iterations
don't require re-extracting.

Usage:
    NANONETS_API_KEY=... python benchmarks/eval_v2.py --limit 15
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
from nanoindex.config import load_config
from nanoindex.core.tree_builder import build_document_tree
from nanoindex.core.refiner import refine_tree
from nanoindex.core.enricher import enrich_tree
from nanoindex.models import ExtractionResult
from nanoindex.utils.tree_ops import iter_nodes, save_tree

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_v2")

FINANCEBENCH_DIR = Path(__file__).resolve().parent.parent.parent / "financebench"
QUESTIONS_PATH = FINANCEBENCH_DIR / "data" / "financebench_open_source.jsonl"
PDFS_DIR = FINANCEBENCH_DIR / "pdfs"
EXTRACTION_CACHE = Path(__file__).resolve().parent / "extraction_cache"
TREE_CACHE = Path(__file__).resolve().parent / "cache_v2"


def load_questions(limit: int | None = None) -> list[dict]:
    with open(QUESTIONS_PATH) as f:
        questions = [json.loads(line) for line in f]
    if limit:
        questions = questions[:limit]
    return questions


def answer_matches(predicted: str, gold: str) -> dict:
    import re
    pred_norm = predicted.lower().strip().replace(",", "")
    gold_norm = gold.lower().strip().replace(",", "")
    exact_match = gold_norm in pred_norm
    pred_nums = set(re.findall(r"[\d]+\.?\d*", pred_norm))
    gold_nums = set(re.findall(r"[\d]+\.?\d*", gold_norm))
    number_overlap = bool(pred_nums & gold_nums) if gold_nums else False

    def _normalize_num(n: str) -> str:
        if "." in n:
            n = n.rstrip("0").rstrip(".")
        return n

    gold_key_nums = re.findall(r"\$?([\d]+\.?\d*)", gold_norm)
    gold_key_nums_clean = [_normalize_num(n) for n in gold_key_nums]
    pred_nums_clean = {_normalize_num(n) for n in pred_nums}
    key_num_found = any(
        n in pred_norm or n in pred_nums_clean for n in gold_key_nums_clean
    ) if gold_key_nums_clean else False

    return {
        "exact_match": exact_match,
        "number_overlap": number_overlap,
        "key_number_found": key_num_found,
        "score": 1.0 if exact_match else (0.75 if key_num_found else (0.5 if number_overlap else 0.0)),
    }


async def get_extraction(doc_name: str, ni: NanoIndex) -> ExtractionResult:
    """Get extraction from cache or Nanonets API."""
    EXTRACTION_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = EXTRACTION_CACHE / f"{doc_name}.json"

    if cache_path.exists():
        logger.info("  Loading cached extraction")
        with open(cache_path) as f:
            return ExtractionResult.model_validate(json.load(f))

    pdf_path = PDFS_DIR / f"{doc_name}.pdf"
    from nanoindex.core.extractor import extract_document
    client = ni._get_client()

    t0 = time.time()
    extraction = await extract_document(pdf_path, client)
    logger.info("  Extracted in %.1fs (%d pages)", time.time() - t0, extraction.page_count)

    with open(cache_path, "w") as f:
        json.dump(extraction.model_dump(), f, ensure_ascii=False)
    return extraction


async def index_document(doc_name: str, ni: NanoIndex):
    """Build tree with new pipeline: extract → tree build → refine → enrich."""
    extraction = await get_extraction(doc_name, ni)

    config = ni.config
    t0 = time.time()
    tree = build_document_tree(extraction, doc_name, config)
    llm = ni._get_llm()
    tree = await refine_tree(tree, llm, config)
    tree = await enrich_tree(tree, llm, config)
    elapsed = time.time() - t0

    all_nodes = list(iter_nodes(tree.structure))
    leaves = [n for n in all_nodes if not n.nodes]
    max_depth = 0

    def _d(n, d=0):
        nonlocal max_depth
        max_depth = max(max_depth, d)
        for c in n.nodes:
            _d(c, d + 1)
    for n in tree.structure:
        _d(n)

    spans = [max(n.end_index - n.start_index + 1, 1) for n in leaves if n.start_index]

    logger.info(
        "  Tree: %d nodes, %d top-level, depth=%d, max_leaf_pages=%d (%.1fs)",
        len(all_nodes), len(tree.structure), max_depth,
        max(spans) if spans else 0, elapsed,
    )

    TREE_CACHE.mkdir(parents=True, exist_ok=True)
    save_tree(tree, TREE_CACHE / f"{doc_name}.json")
    return tree


async def run_eval(questions: list[dict], ni: NanoIndex) -> list[dict]:
    doc_questions: dict[str, list[dict]] = {}
    for q in questions:
        doc_questions.setdefault(q["doc_name"], []).append(q)

    logger.info(
        "Evaluating %d questions across %d documents",
        len(questions), len(doc_questions),
    )

    results = []

    for doc_idx, (doc_name, doc_qs) in enumerate(doc_questions.items()):
        pdf_path = PDFS_DIR / f"{doc_name}.pdf"
        if not pdf_path.exists():
            logger.warning("PDF not found: %s", pdf_path)
            for q in doc_qs:
                results.append({
                    "id": q["financebench_id"], "doc_name": doc_name,
                    "question": q["question"], "gold_answer": q["answer"],
                    "predicted": "SKIPPED", "score": 0.0,
                })
            continue

        logger.info(
            "[%d/%d] %s (%d questions)",
            doc_idx + 1, len(doc_questions), doc_name, len(doc_qs),
        )

        try:
            tree = await index_document(doc_name, ni)
        except Exception as exc:
            logger.error("  Indexing failed: %s", exc, exc_info=True)
            for q in doc_qs:
                results.append({
                    "id": q["financebench_id"], "doc_name": doc_name,
                    "question": q["question"], "gold_answer": q["answer"],
                    "predicted": f"INDEX_FAIL: {exc}", "score": 0.0,
                })
            continue

        for q in doc_qs:
            logger.info("  Q: %s", q["question"][:80])
            try:
                ans = await ni.async_ask(q["question"], tree, include_metadata=True)
                predicted = ans.content
                citations = [{"title": c.title, "pages": c.pages} for c in ans.citations]
            except Exception as exc:
                logger.error("    Answer failed: %s", exc)
                predicted = f"ERROR: {exc}"
                citations = []

            match = answer_matches(predicted, q["answer"])

            evidence_pages = set()
            for ev in q.get("evidence", []):
                pg = ev.get("evidence_page_num")
                if pg is not None:
                    evidence_pages.add(pg + 1)

            cited_pages = set()
            for c in citations:
                cited_pages.update(c.get("pages", []))

            page_hit = bool(evidence_pages & cited_pages) if evidence_pages else None

            result = {
                "id": q["financebench_id"],
                "doc_name": doc_name,
                "question": q["question"],
                "gold_answer": q["answer"],
                "predicted": predicted[:500],
                "score": match["score"],
                "exact_match": match["exact_match"],
                "key_number_found": match["key_number_found"],
                "evidence_page_hit": page_hit,
            }
            results.append(result)

            status = "✅" if match["score"] >= 0.75 else "⚠️" if match["score"] > 0 else "❌"
            logger.info(
                "    %s score=%.2f | Gold: %s | Pred: %s",
                status, match["score"], q["answer"][:70], predicted[:70],
            )

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
    page_hit_count = sum(1 for r in page_hits if r["evidence_page_hit"])

    print(f"\n{'='*70}")
    print(f"  FinanceBench v2 Evaluation — {total} questions")
    print(f"{'='*70}")
    print(f"  Average score:       {avg_score:.1%}")
    print(f"  Exact match:         {exact}/{total} ({exact/total:.1%})")
    print(f"  Key number found:    {key_num}/{total} ({key_num/total:.1%})")
    if page_hits:
        print(f"  Evidence page hit:   {page_hit_count}/{len(page_hits)} ({page_hit_count/len(page_hits):.1%})")

    print(f"\n  Per-question breakdown:")
    for r in results:
        status = "✅" if r["score"] >= 0.75 else "⚠️" if r["score"] > 0 else "❌"
        pg = "📄" if r.get("evidence_page_hit") else "  "
        print(f"    {status} {pg} [{r['doc_name'][:25]:25s}] score={r['score']:.2f}")
        print(f"         Q: {r['question'][:65]}")
        print(f"         Gold: {r['gold_answer'][:65]}")
        print(f"         Pred: {r['predicted'][:65]}")

    print(f"{'='*70}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=15)
    args = parser.parse_args()

    if not QUESTIONS_PATH.exists():
        print(f"ERROR: FinanceBench data not found at {QUESTIONS_PATH}")
        sys.exit(1)

    questions = load_questions(args.limit)
    logger.info("Loaded %d questions", len(questions))

    ni = NanoIndex(nanonets_api_key=os.environ.get("NANONETS_API_KEY"))
    ni.config.confidence_threshold = 0.0

    results = asyncio.run(run_eval(questions, ni))

    output_path = Path(__file__).resolve().parent / "results_v2.jsonl"
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print_summary(results)


if __name__ == "__main__":
    main()
