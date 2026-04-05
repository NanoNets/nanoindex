"""CUAD (Contract Understanding Atticus Dataset) evaluation using NanoIndex.

Full pipeline: contract text → PDF → Nanonets extraction → tree building →
agentic retrieval → answer extraction → SQuAD-style F1/EM evaluation.

Usage:
    # Phase 1: Convert contracts to PDFs and build tree caches
    NANONETS_API_KEY=... python benchmarks/cuad_eval.py --build-cache-only

    # Phase 2: Evaluate with Claude
    NANONETS_API_KEY=... ANTHROPIC_API_KEY=... python benchmarks/cuad_eval.py \
        --reasoning-model claude-sonnet-4-6

    # Quick test on first N contracts
    NANONETS_API_KEY=... ANTHROPIC_API_KEY=... python benchmarks/cuad_eval.py \
        --reasoning-model claude-sonnet-4-6 --limit 5
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import logging
import os
import re
import string
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nanoindex import NanoIndex
from nanoindex.core.extractor import extract_document
from nanoindex.core.tree_builder import build_document_tree
from nanoindex.core.refiner import refine_tree
from nanoindex.core.enricher import enrich_tree
from nanoindex.utils.tree_ops import iter_nodes, save_tree, load_tree

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cuad_eval")

CUAD_DIR = Path(__file__).resolve().parent.parent.parent / "cuad"
TEST_PATH = CUAD_DIR / "data" / "test.json"

BENCH_DIR = Path(__file__).resolve().parent
PDF_DIR = BENCH_DIR / "cuad_pdfs"
EXTRACTION_CACHE = BENCH_DIR / "cuad_extraction_cache"
TREE_CACHE = BENCH_DIR / "cuad_tree_cache"
RESULTS_DIR = BENCH_DIR / "results_cuad"


# ------------------------------------------------------------------
# PDF conversion (text → PDF via reportlab or fpdf)
# ------------------------------------------------------------------

def text_to_pdf(text: str, output_path: Path) -> None:
    """Convert plain text to a simple PDF."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open()
        # Split text into pages (~4000 chars each)
        page_size = 4000
        for i in range(0, len(text), page_size):
            page = doc.new_page(width=612, height=792)
            chunk = text[i:i + page_size]
            # Insert text with wrapping
            rect = fitz.Rect(50, 50, 562, 742)
            page.insert_textbox(rect, chunk, fontsize=9, fontname="Courier")
        doc.save(str(output_path))
        doc.close()
    except ImportError:
        # Fallback: use fpdf2
        from fpdf import FPDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Courier", size=8)
        for line in text.split("\n"):
            pdf.multi_cell(0, 4, line)
        pdf.output(str(output_path))


# ------------------------------------------------------------------
# SQuAD-style metrics
# ------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Lower text, remove punctuation/articles/extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_em(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: list[str]) -> float:
    """Take max metric over all ground truth answers."""
    if not ground_truths:
        return float(normalize_answer(prediction) == "")
    return max(metric_fn(prediction, gt) for gt in ground_truths)


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_test_data(limit: int | None = None) -> list[dict]:
    """Load CUAD test set, return flat list of (contract, question, answers)."""
    with open(TEST_PATH) as f:
        data = json.load(f)

    contracts = data["data"]
    if limit:
        contracts = contracts[:limit]

    entries = []
    for contract in contracts:
        title = contract["title"]
        for para in contract["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                gold_answers = [a["text"] for a in qa.get("answers", [])]
                entries.append({
                    "contract_title": title,
                    "context": context,
                    "question": qa["question"],
                    "question_id": qa["id"],
                    "is_impossible": qa.get("is_impossible", False),
                    "gold_answers": gold_answers,
                })
    return entries


# ------------------------------------------------------------------
# Tree building with caching
# ------------------------------------------------------------------

def _safe_filename(title: str) -> str:
    """Convert contract title to safe filename."""
    return re.sub(r'[^\w\-.]', '_', title)[:100]


async def get_tree(contract_title: str, context: str, ni: NanoIndex):
    """Build or load cached tree for a contract."""
    safe_name = _safe_filename(contract_title)
    tree_path = TREE_CACHE / f"{safe_name}.json"

    if tree_path.exists():
        return load_tree(tree_path)

    # Step 1: Convert text to PDF
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = PDF_DIR / f"{safe_name}.pdf"
    if not pdf_path.exists():
        text_to_pdf(context, pdf_path)
        logger.info("  Created PDF: %s (%d chars)", safe_name, len(context))

    # Step 2: Extract via Nanonets
    EXTRACTION_CACHE.mkdir(parents=True, exist_ok=True)
    ext_path = EXTRACTION_CACHE / f"{safe_name}.json"

    if ext_path.exists():
        from nanoindex.models import ExtractionResult
        with open(ext_path) as f:
            extraction = ExtractionResult.model_validate(json.load(f))
    else:
        client = ni._get_client()
        extraction = await extract_document(pdf_path, client)
        with open(ext_path, "w") as f:
            json.dump(extraction.model_dump(), f, ensure_ascii=False)
        logger.info("  Extracted %s: %d pages", safe_name, extraction.page_count)

    # Step 3: Build tree
    TREE_CACHE.mkdir(parents=True, exist_ok=True)
    tree = build_document_tree(extraction, safe_name, ni.config)
    llm = ni._get_llm()
    tree = await refine_tree(tree, llm, ni.config)
    tree = await enrich_tree(tree, llm, ni.config)

    save_tree(tree, tree_path)
    n_nodes = sum(1 for _ in iter_nodes(tree.structure))
    logger.info("  Built tree: %d nodes", n_nodes)
    return tree


# ------------------------------------------------------------------
# Answer extraction prompt
# ------------------------------------------------------------------

_EXTRACT_PROMPT = """\
You are a legal contract analyst. Given the question and document context below, \
extract the EXACT text span from the context that answers the question.

Question: {question}

Context:
{context}

RULES:
1. If the answer exists in the context, return ONLY the exact text span — \
copy it character-for-character from the context. Do not paraphrase.
2. If the answer does NOT exist in the context, respond with exactly: \
"NO_ANSWER"
3. If there are multiple answer spans, return the most specific/complete one.
4. Be precise — include only the relevant clause text, not surrounding boilerplate.

Answer:"""


# ------------------------------------------------------------------
# Evaluation loop
# ------------------------------------------------------------------

async def run_eval(
    entries: list[dict],
    ni: NanoIndex,
    *,
    concurrency: int = 3,
) -> list[dict]:
    """Run evaluation: build trees, retrieve, extract answers, score."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / "answers.jsonl"

    # Group by contract
    by_contract: dict[str, list[dict]] = {}
    for e in entries:
        by_contract.setdefault(e["contract_title"], []).append(e)

    logger.info("Evaluating %d questions across %d contracts",
                len(entries), len(by_contract))

    llm = ni._get_reasoning_llm()
    sem = asyncio.Semaphore(concurrency)
    results = []

    for c_idx, (title, contract_entries) in enumerate(sorted(by_contract.items())):
        context = contract_entries[0]["context"]
        logger.info("[%d/%d] %s (%d questions)",
                    c_idx + 1, len(by_contract), title[:60], len(contract_entries))

        # Build tree (cached)
        try:
            tree = await get_tree(title, context, ni)
        except Exception as exc:
            logger.error("  Tree build failed: %s", exc)
            for e in contract_entries:
                results.append({
                    "id": e["question_id"],
                    "contract": title,
                    "question": e["question"],
                    "gold_answers": e["gold_answers"],
                    "is_impossible": e["is_impossible"],
                    "predicted": "TREE_BUILD_FAILED",
                    "f1": 0.0,
                    "em": 0.0,
                })
            continue

        # Answer each question
        async def _answer(entry):
            async with sem:
                if entry["is_impossible"]:
                    # For impossible questions, ideal answer is empty
                    # We still run to check if model correctly says "no answer"
                    pass

                try:
                    ans = await ni.async_ask(
                        entry["question"], tree,
                        mode="agentic",
                        include_metadata=True,
                    )
                    predicted = ans.content
                except Exception as exc:
                    logger.error("    Q failed: %s", str(exc)[:80])
                    predicted = f"ERROR: {exc}"

                # For extractive QA, try to pull out the actual span
                # The model may return a full explanation; extract the key text
                pred_clean = predicted.strip()

                # Compute metrics
                if entry["is_impossible"]:
                    # Correct if model says no answer / refuses
                    no_answer_patterns = [
                        "no_answer", "not found", "not present",
                        "does not contain", "no relevant", "not mentioned",
                        "cannot find", "not included", "no information",
                    ]
                    is_correct_refusal = any(p in pred_clean.lower() for p in no_answer_patterns)
                    f1 = 1.0 if is_correct_refusal else 0.0
                    em = f1
                else:
                    f1 = metric_max_over_ground_truths(compute_f1, pred_clean, entry["gold_answers"])
                    em = metric_max_over_ground_truths(compute_em, pred_clean, entry["gold_answers"])

                result = {
                    "id": entry["question_id"],
                    "contract": title,
                    "question": entry["question"][:100],
                    "gold_answers": entry["gold_answers"],
                    "is_impossible": entry["is_impossible"],
                    "predicted": pred_clean[:1000],
                    "f1": f1,
                    "em": em,
                }

                # Append incrementally
                with open(results_file, "a") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

                return result

        tasks = [_answer(e) for e in contract_entries]
        contract_results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in contract_results:
            if isinstance(r, Exception):
                logger.error("  Unexpected error: %s", r)
                continue
            results.append(r)

        # Per-contract summary
        answerable = [r for r in contract_results if isinstance(r, dict) and not r["is_impossible"]]
        if answerable:
            avg_f1 = sum(r["f1"] for r in answerable) / len(answerable)
            logger.info("  Answerable F1: %.1f%% (%d questions)", avg_f1 * 100, len(answerable))

    return results


# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------

def print_summary(results: list[dict], model: str):
    total = len(results)
    answerable = [r for r in results if not r["is_impossible"]]
    impossible = [r for r in results if r["is_impossible"]]

    ans_f1 = sum(r["f1"] for r in answerable) / len(answerable) if answerable else 0
    ans_em = sum(r["em"] for r in answerable) / len(answerable) if answerable else 0
    imp_acc = sum(r["f1"] for r in impossible) / len(impossible) if impossible else 0
    overall_f1 = sum(r["f1"] for r in results) / total if total else 0

    print(f"\n{'='*70}")
    print(f"  CUAD Evaluation — {total} questions across {len(set(r['contract'] for r in results))} contracts")
    print(f"  Reasoning LLM: {model}")
    print(f"{'='*70}")
    print(f"  Answerable ({len(answerable)} questions):")
    print(f"    F1:           {ans_f1:.1%}")
    print(f"    Exact Match:  {ans_em:.1%}")
    print(f"  Impossible ({len(impossible)} questions):")
    print(f"    Accuracy:     {imp_acc:.1%}")
    print(f"  Overall F1:     {overall_f1:.1%}")

    # By clause type
    by_clause: dict[str, list[dict]] = {}
    for r in results:
        clause = r["id"].split("__")[-1] if "__" in r["id"] else "unknown"
        by_clause.setdefault(clause, []).append(r)

    print(f"\n  Top clause types (by answerable F1):")
    clause_scores = []
    for clause, rs in sorted(by_clause.items()):
        ans_rs = [r for r in rs if not r["is_impossible"]]
        if ans_rs:
            cf1 = sum(r["f1"] for r in ans_rs) / len(ans_rs)
            clause_scores.append((clause, cf1, len(ans_rs)))

    for clause, cf1, n in sorted(clause_scores, key=lambda x: -x[1])[:15]:
        print(f"    {clause:45s}: F1={cf1:.0%} ({n} Qs)")

    print(f"{'='*70}\n")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _resolve_key(args) -> str | None:
    if args.reasoning_key:
        return args.reasoning_key
    model = args.reasoning_model or ""
    if model.startswith("claude"):
        return os.environ.get("ANTHROPIC_API_KEY")
    if model.startswith("gpt"):
        return os.environ.get("OPENAI_API_KEY")
    return None


def main():
    parser = argparse.ArgumentParser(description="CUAD evaluation with NanoIndex")
    parser.add_argument("--build-cache-only", action="store_true",
                        help="Only build tree caches, skip evaluation")
    parser.add_argument("--reasoning-model", default=None,
                        help="Model name (e.g. claude-sonnet-4-6)")
    parser.add_argument("--reasoning-key", default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N contracts")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--clear-results", action="store_true")
    args = parser.parse_args()

    if not TEST_PATH.exists():
        print(f"ERROR: CUAD test data not found at {TEST_PATH}")
        print("Run: cd ../cuad && unzip data.zip -d data/")
        sys.exit(1)

    entries = load_test_data(args.limit)
    logger.info("Loaded %d QA entries from %d contracts",
                len(entries), len(set(e["contract_title"] for e in entries)))

    # Build NanoIndex
    config_kwargs: dict = {"nanonets_api_key": os.environ.get("NANONETS_API_KEY")}
    if args.reasoning_model:
        rkey = _resolve_key(args)
        config_kwargs["reasoning_llm_model"] = args.reasoning_model
        if rkey:
            config_kwargs["reasoning_llm_api_key"] = rkey

    ni = NanoIndex(**config_kwargs)

    if args.clear_results:
        results_file = RESULTS_DIR / "answers.jsonl"
        if results_file.exists():
            results_file.unlink()

    if args.build_cache_only:
        logger.info("Phase 1: Building tree caches...")
        by_contract = {}
        for e in entries:
            by_contract.setdefault(e["contract_title"], e["context"])

        async def _build_all():
            for idx, (title, context) in enumerate(sorted(by_contract.items())):
                logger.info("[%d/%d] %s", idx + 1, len(by_contract), title[:60])
                try:
                    await get_tree(title, context, ni)
                except Exception as exc:
                    logger.error("  FAILED: %s", exc)

        asyncio.run(_build_all())
        return

    # Phase 2: Evaluate
    results = asyncio.run(run_eval(entries, ni, concurrency=args.concurrency))

    # Save summary
    summary_path = RESULTS_DIR / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    model = args.reasoning_model or "default"
    print_summary(results, model)


if __name__ == "__main__":
    main()
