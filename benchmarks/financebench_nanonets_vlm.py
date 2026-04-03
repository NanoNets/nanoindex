"""FinanceBench evaluation using Nanonets-OCR-3 as the reasoning VLM.

Uses NanoIndex tree retrieval to find relevant pages, then sends page images
to nanonets-ocr-3 VLM for answer generation. Tests the full NanoIndex +
Nanonets-OCR-3 pipeline without any external LLM (Claude/GPT).

Usage:
    NANONETS_API_KEY=... python benchmarks/financebench_nanonets_vlm.py
    NANONETS_API_KEY=... python benchmarks/financebench_nanonets_vlm.py --limit 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
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
logger = logging.getLogger("financebench_vlm")

FINANCEBENCH_DIR = Path(__file__).resolve().parent.parent.parent / "financebench"
QUESTIONS_PATH = FINANCEBENCH_DIR / "data" / "financebench_open_source.jsonl"
PDFS_DIR = FINANCEBENCH_DIR / "pdfs"

BENCH_DIR = Path(__file__).resolve().parent
TREE_CACHE = BENCH_DIR / "cache_v3"
RESULTS_DIR = BENCH_DIR / "results_nanonets_vlm"

NANONETS_VLM_URL = "https://extraction-api.nanonets.com/v1"
NANONETS_VLM_MODEL = "nanonets-ocr-3"
MAX_CONTEXT_TOKENS = 45000
TOKENS_PER_PAGE = 2000  # ~1921 measured, round up for safety
MAX_PAGES = 10  # Conservative: 10 pages * 2K = 20K tokens for images, leaving 25K for text + prompt


# ------------------------------------------------------------------
# Heuristic scoring
# ------------------------------------------------------------------

def heuristic_score(predicted: str, gold: str) -> dict:
    pred_norm = predicted.lower().strip().replace(",", "")
    gold_norm = gold.lower().strip().replace(",", "")
    exact_match = gold_norm in pred_norm
    pred_nums = set(re.findall(r"[\d]+\.?\d*", pred_norm))
    gold_nums = set(re.findall(r"[\d]+\.?\d*", gold_norm))
    number_overlap = bool(pred_nums & gold_nums) if gold_nums else False

    def _norm(n):
        if "." in n:
            n = n.rstrip("0").rstrip(".")
        return n

    gold_key = re.findall(r"\$?([\d]+\.?\d*)", gold_norm)
    gold_clean = [_norm(n) for n in gold_key]
    pred_clean = {_norm(n) for n in pred_nums}
    key_found = any(n in pred_norm or n in pred_clean for n in gold_clean) if gold_clean else False

    return {
        "exact_match": exact_match,
        "key_number_found": key_found,
        "number_overlap": number_overlap,
        "heuristic_score": 1.0 if exact_match else (0.75 if key_found else (0.5 if number_overlap else 0.0)),
    }


# ------------------------------------------------------------------
# LLM-as-judge (using nanonets-ocr-3 itself)
# ------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are an expert evaluator. Determine whether the AI answer correctly \
answers the query based on the golden answer.

Numerical Accuracy: Rounding differences should be ignored. \
Fractions/percentages/numerics are considered similar (11/14 ≈ 79%).

The AI answer is correct if it conveys the same meaning, conclusion, or \
rationale as the golden answer, or is a superset of it.

Query: {query}
AI Answer: {predicted}
Golden Answer: {gold}

Output ONLY: True or False"""


async def llm_judge(query: str, predicted: str, gold: str, llm: LLMClient) -> bool | None:
    prompt = _JUDGE_PROMPT.format(query=query, predicted=predicted, gold=gold)
    try:
        resp = await llm.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=64)
        return "true" in resp.strip().lower()
    except Exception as exc:
        logger.warning("Judge failed: %s", exc)
        return None


# ------------------------------------------------------------------
# Main evaluation
# ------------------------------------------------------------------

def load_questions(limit: int | None = None) -> list[dict]:
    with open(QUESTIONS_PATH) as f:
        questions = [json.loads(line) for line in f]
    if limit:
        questions = questions[:limit]
    return questions


async def run_eval(questions: list[dict], ni: NanoIndex, judge_llm: LLMClient, *, concurrency: int = 3):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / "answers.jsonl"
    if results_file.exists():
        results_file.unlink()

    doc_questions: dict[str, list[dict]] = {}
    for q in questions:
        doc_questions.setdefault(q["doc_name"], []).append(q)

    logger.info("Evaluating %d questions across %d documents",
                len(questions), len(doc_questions))
    logger.info("Reasoning VLM: %s (max %d pages per query)", NANONETS_VLM_MODEL, MAX_PAGES)

    sem = asyncio.Semaphore(concurrency)
    results = []

    for doc_idx, (doc_name, doc_qs) in enumerate(sorted(doc_questions.items())):
        pdf_path = PDFS_DIR / f"{doc_name}.pdf"
        tree_path = TREE_CACHE / f"{doc_name}.json"

        if not pdf_path.exists() or not tree_path.exists():
            logger.warning("[%d/%d] %s — missing PDF or tree, skipping",
                          doc_idx + 1, len(doc_questions), doc_name)
            continue

        tree = load_tree(tree_path)
        logger.info("[%d/%d] %s (%d questions)",
                    doc_idx + 1, len(doc_questions), doc_name, len(doc_qs))

        async def _answer_question(q, tree=tree, doc_name=doc_name, pdf_path=pdf_path):
            qid = q["financebench_id"]

            async with sem:
                logger.info("  Q: %s", q["question"][:80])
                try:
                    ans = await ni.async_ask(
                        q["question"], tree,
                        mode="text",
                        pdf_path=pdf_path,
                        include_metadata=True,
                    )
                    predicted = ans.content
                    citations = [
                        {"title": c.title, "doc_name": c.doc_name, "pages": c.pages}
                        for c in ans.citations
                    ]
                except Exception as exc:
                    logger.error("    Answer failed: %s", exc)
                    predicted = f"ERROR: {exc}"
                    citations = []

                h = heuristic_score(predicted, q["answer"])

                judge_result = None
                if not predicted.startswith("ERROR"):
                    judge_result = await llm_judge(q["question"], predicted, q["answer"], judge_llm)

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
                    "id": qid,
                    "doc_name": doc_name,
                    "question": q["question"],
                    "gold_answer": q["answer"],
                    "predicted": predicted,
                    "heuristic_score": h["heuristic_score"],
                    "exact_match": h["exact_match"],
                    "key_number_found": h["key_number_found"],
                    "llm_judge": judge_result,
                    "evidence_page_hit": page_hit,
                    "citations": citations,
                    "reasoning_model": NANONETS_VLM_MODEL,
                    "mode": "text",
                }

                with open(results_file, "a") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

                jstr = f" | judge={'✅' if judge_result else '❌'}" if judge_result is not None else ""
                status = "✅" if h["heuristic_score"] >= 0.75 else "⚠️" if h["heuristic_score"] > 0 else "❌"
                logger.info("    %s h=%.2f%s | Gold: %s | Pred: %s",
                            status, h["heuristic_score"], jstr,
                            q["answer"][:60], predicted[:60])
                return result

        tasks = [_answer_question(q) for q in doc_qs]
        doc_results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in doc_results:
            if isinstance(r, Exception):
                logger.error("  Unexpected error: %s", r)
                continue
            results.append(r)

    return results


def print_summary(results: list[dict]):
    total = len(results)
    if not total:
        print("No results.")
        return

    h_score = sum(r["heuristic_score"] for r in results) / total
    exact = sum(1 for r in results if r.get("exact_match"))
    key_num = sum(1 for r in results if r.get("key_number_found"))
    judged = [r for r in results if r.get("llm_judge") is not None]
    judge_correct = sum(1 for r in judged if r["llm_judge"])
    page_hits = [r for r in results if r.get("evidence_page_hit") is not None]
    page_correct = sum(1 for r in page_hits if r["evidence_page_hit"])

    print(f"\n{'='*70}")
    print(f"  FinanceBench — Nanonets-OCR-3 VLM — {total} questions")
    print(f"  Model: {NANONETS_VLM_MODEL} (agentic_vision, max {MAX_PAGES} pages)")
    print(f"{'='*70}")
    print(f"  Heuristic avg score: {h_score:.1%}")
    print(f"  Exact match:         {exact}/{total} ({exact/total:.1%})")
    print(f"  Key number found:    {key_num}/{total} ({key_num/total:.1%})")
    if judged:
        print(f"  LLM Judge accuracy:  {judge_correct}/{len(judged)} ({judge_correct/len(judged):.1%})")
    if page_hits:
        print(f"  Evidence page hit:   {page_correct}/{len(page_hits)} ({page_correct/len(page_hits):.1%})")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="FinanceBench eval with Nanonets-OCR-3 VLM")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()

    if not QUESTIONS_PATH.exists():
        print(f"ERROR: FinanceBench data not found at {QUESTIONS_PATH}")
        sys.exit(1)

    questions = load_questions(args.limit)
    logger.info("Loaded %d questions", len(questions))

    nanonets_key = os.environ.get("NANONETS_API_KEY")
    if not nanonets_key:
        print("ERROR: Set NANONETS_API_KEY")
        sys.exit(1)

    # Use nanonets-ocr-3 for BOTH retrieval reasoning AND answer generation
    ni = NanoIndex(
        nanonets_api_key=nanonets_key,
        reasoning_llm_base_url=NANONETS_VLM_URL,
        reasoning_llm_api_key=nanonets_key,
        reasoning_llm_model=NANONETS_VLM_MODEL,
    )
    # Limit context for 45K token model
    ni.config.max_node_tokens = 8000

    # Monkey-patch context limits for 45K token model
    # 1. Reduce retriever context budget so tree outline fits
    import nanoindex.core.retriever as _retriever
    _retriever._CONTEXT_BUDGET = 30000  # was 120K, reduced for 45K model

    # 2. Reduce generator text context
    import nanoindex.core.generator as _generator
    _orig_build_text = _generator._build_text_context

    def _limited_build_text(nodes, **kwargs):
        return _orig_build_text(nodes, max_tokens=30000)

    _generator._build_text_context = _limited_build_text

    # Also use nanonets-ocr-3 as judge
    judge_llm = LLMClient(
        api_key=nanonets_key,
        base_url=NANONETS_VLM_URL,
        model=NANONETS_VLM_MODEL,
    )

    results = asyncio.run(run_eval(questions, ni, judge_llm, concurrency=args.concurrency))

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print_summary(results)


if __name__ == "__main__":
    main()
