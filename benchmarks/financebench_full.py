"""FinanceBench full evaluation — PageIndex-compatible methodology.

Two-phase workflow:
  Phase 1: Build tree caches for all documents using Nanonets extraction
           + default LLM for enrichment (cheap).
  Phase 2: Evaluate using a powerful reasoning LLM (Claude, Gemini, GPT-4o)
           for retrieval + answer generation, then LLM-as-judge scoring.

Usage:
    # ── Phase 1: build tree caches (Nanonets only, no reasoning LLM needed) ──
    NANONETS_API_KEY=... python benchmarks/financebench_full.py --build-cache-only

    # ── Phase 2: evaluate with Claude ──
    NANONETS_API_KEY=... ANTHROPIC_API_KEY=... python benchmarks/financebench_full.py \\
        --reasoning-model claude-sonnet-4-20250514

    # ── Phase 2: evaluate with Gemini ──
    NANONETS_API_KEY=... GOOGLE_API_KEY=... python benchmarks/financebench_full.py \\
        --reasoning-model gemini-2.5-flash \\
        --reasoning-url https://generativelanguage.googleapis.com/v1beta/openai/ \\
        --reasoning-key $GOOGLE_API_KEY

    # ── Phase 2: evaluate with GPT-4o ──
    NANONETS_API_KEY=... OPENAI_API_KEY=... python benchmarks/financebench_full.py \\
        --reasoning-model gpt-4o \\
        --reasoning-url https://api.openai.com/v1 \\
        --reasoning-key $OPENAI_API_KEY

    # ── Post-hoc: re-judge existing results ──
    python benchmarks/financebench_full.py --judge-only --judge-model gpt-4o

    # ── Options ──
    --limit N              Only evaluate first N questions
    --clear-results        Wipe previous answer cache before running
    --concurrency N        Max parallel answer-generation tasks (default 3)
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

from nanoindex import NanoIndex
from nanoindex.core.tree_builder import build_document_tree
from nanoindex.core.refiner import refine_tree
from nanoindex.core.enricher import enrich_tree
from nanoindex.models import ExtractionResult
from nanoindex.utils.tree_ops import iter_nodes, save_tree, load_tree

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("financebench")

FINANCEBENCH_DIR = Path(__file__).resolve().parent.parent.parent / "financebench"
QUESTIONS_PATH = FINANCEBENCH_DIR / "data" / "financebench_open_source.jsonl"
PDFS_DIR = FINANCEBENCH_DIR / "pdfs"

BENCH_DIR = Path(__file__).resolve().parent
EXTRACTION_CACHE = BENCH_DIR / "extraction_cache"
TREE_CACHE = BENCH_DIR / "cache_v3"
RESULTS_DIR = BENCH_DIR / "results_full"


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_questions(limit: int | None = None) -> list[dict]:
    with open(QUESTIONS_PATH) as f:
        questions = [json.loads(line) for line in f]
    if limit:
        questions = questions[:limit]
    return questions


# ------------------------------------------------------------------
# Heuristic scoring (fast, no LLM needed)
# ------------------------------------------------------------------

def heuristic_score(predicted: str, gold: str) -> dict:
    pred_norm = predicted.lower().strip().replace(",", "")
    gold_norm = gold.lower().strip().replace(",", "")
    exact_match = gold_norm in pred_norm

    pred_nums = set(re.findall(r"[\d]+\.?\d*", pred_norm))
    gold_nums = set(re.findall(r"[\d]+\.?\d*", gold_norm))
    number_overlap = bool(pred_nums & gold_nums) if gold_nums else False

    def _norm(n: str) -> str:
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
# LLM-as-judge scoring (PageIndex / Mafin2.5 methodology)
# ------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are an expert evaluator for AI-generated responses to queries. Your task \
is to determine whether the AI-generated answer correctly answers the query \
based on the golden answer provided by a human expert.

Numerical Accuracy:
- Rounding differences should be ignored if they do not meaningfully change \
the conclusion.
- You can allow some flexibility in accuracy. For example, 1.2 is considered \
similar to 1.23. Two numbers are considered similar if one can be rounded to \
the other.
- Fractions, percentage, and numerics could be considered similar, for example: \
"11 of 14" is considered equivalent to "79%" and "0.79".

Evaluation Criteria:
- If the golden answer or any of its equivalence can be inferred or generated \
from the AI-generated answer, then the AI-generated answer is considered correct.
- If any number, percentage, fraction, or figure in the golden answer is not \
present in the AI-generated answer, but can be inferred or generated from the \
AI-generated answer or implicitly exist in the AI-generated answer, then the \
AI-generated answer is considered correct.
- The AI-generated answer is considered correct if it conveys the same or \
similar meaning, conclusion, or rationale as the golden answer.
- If the AI-generated answer is a superset of the golden answer, it is also \
considered correct.
- If the AI-generated answer provides a valid answer or reasonable \
interpretation compared to the golden answer, it is considered correct.
- If the AI-generated answer contains subjective judgments or opinions, it is \
considered correct as long as they are reasonable and justifiable compared to \
the golden answer.
- Otherwise, the AI-generated answer is incorrect.

Inputs:
- Query: {query}
- AI-Generated Answer: {predicted}
- Golden Answer: {gold}

Your output should be ONLY a boolean value: `True` or `False`, nothing else."""


async def llm_judge(query: str, predicted: str, gold: str, llm) -> bool | None:
    prompt = _JUDGE_PROMPT.format(query=query, predicted=predicted, gold=gold)
    try:
        resp = await llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=64,
        )
        return "true" in resp.strip().lower()
    except Exception as exc:
        logger.warning("Judge call failed: %s", exc)
        return None


# ------------------------------------------------------------------
# Document indexing with caching
# ------------------------------------------------------------------

async def get_extraction(doc_name: str, ni: NanoIndex) -> ExtractionResult:
    EXTRACTION_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = EXTRACTION_CACHE / f"{doc_name}.json"

    if cache_path.exists():
        with open(cache_path) as f:
            return ExtractionResult.model_validate(json.load(f))

    pdf_path = PDFS_DIR / f"{doc_name}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    from nanoindex.core.extractor import extract_document
    client = ni._get_client()

    t0 = time.time()
    extraction = await extract_document(pdf_path, client)
    logger.info("  Extracted %s in %.1fs (%d pages)", doc_name, time.time() - t0, extraction.page_count)

    with open(cache_path, "w") as f:
        json.dump(extraction.model_dump(), f, ensure_ascii=False)
    return extraction


async def get_tree(doc_name: str, ni: NanoIndex):
    TREE_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = TREE_CACHE / f"{doc_name}.json"

    if cache_path.exists():
        return load_tree(cache_path)

    extraction = await get_extraction(doc_name, ni)
    t0 = time.time()
    tree = build_document_tree(extraction, doc_name, ni.config)
    llm = ni._get_llm()
    tree = await refine_tree(tree, llm, ni.config)
    tree = await enrich_tree(tree, llm, ni.config)
    elapsed = time.time() - t0

    all_nodes = list(iter_nodes(tree.structure))
    logger.info("  Built tree for %s: %d nodes, %d top-level (%.1fs)",
                doc_name, len(all_nodes), len(tree.structure), elapsed)

    save_tree(tree, cache_path)
    return tree


# ------------------------------------------------------------------
# Phase 1: Build all tree caches
# ------------------------------------------------------------------

async def build_all_caches(ni: NanoIndex, questions: list[dict]):
    doc_names = sorted(set(q["doc_name"] for q in questions))
    cached = [d for d in doc_names if (TREE_CACHE / f"{d}.json").exists()]
    to_build = [d for d in doc_names if d not in set(cached)]

    logger.info(
        "Tree cache: %d/%d docs cached, %d to build",
        len(cached), len(doc_names), len(to_build),
    )

    errors: list[tuple[str, str]] = []
    for idx, doc_name in enumerate(to_build):
        pdf_path = PDFS_DIR / f"{doc_name}.pdf"
        if not pdf_path.exists():
            logger.warning("[%d/%d] SKIP %s — PDF not found", idx + 1, len(to_build), doc_name)
            errors.append((doc_name, "PDF not found"))
            continue

        logger.info("[%d/%d] Building tree for %s...", idx + 1, len(to_build), doc_name)
        try:
            await get_tree(doc_name, ni)
        except Exception as exc:
            logger.error("  FAILED %s: %s", doc_name, exc)
            errors.append((doc_name, str(exc)))

    final_cached = [d for d in doc_names if (TREE_CACHE / f"{d}.json").exists()]
    logger.info(
        "\nCache build complete: %d/%d docs cached",
        len(final_cached), len(doc_names),
    )
    if errors:
        logger.warning("Errors during cache build:")
        for name, err in errors:
            logger.warning("  %s: %s", name, err[:80])

    return final_cached


# ------------------------------------------------------------------
# Phase 2: Evaluation loop
# ------------------------------------------------------------------

async def run_eval(
    questions: list[dict],
    ni: NanoIndex,
    *,
    judge_llm=None,
    concurrency: int = 3,
    use_vision: bool = False,
    answer_mode: str = "text",
) -> list[dict]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_cache = RESULTS_DIR / "answers.jsonl"

    existing: dict[str, dict] = {}
    if results_cache.exists():
        with open(results_cache) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    existing[r["id"]] = r

    doc_questions: dict[str, list[dict]] = {}
    for q in questions:
        doc_questions.setdefault(q["doc_name"], []).append(q)

    n_cached = sum(1 for q in questions if q["financebench_id"] in existing)
    logger.info(
        "Evaluating %d questions across %d documents (%d cached, %d to run)",
        len(questions), len(doc_questions), n_cached, len(questions) - n_cached,
    )

    reasoning_model = ni.config.reasoning_llm_model or ni.config.llm_model
    logger.info("Reasoning LLM: %s", reasoning_model)

    results: list[dict] = []
    sem = asyncio.Semaphore(concurrency)

    for doc_idx, (doc_name, doc_qs) in enumerate(sorted(doc_questions.items())):
        pdf_path = PDFS_DIR / f"{doc_name}.pdf"
        if not pdf_path.exists():
            logger.warning("PDF not found: %s", pdf_path)
            for q in doc_qs:
                results.append({
                    "id": q["financebench_id"], "doc_name": doc_name,
                    "question": q["question"], "gold_answer": q["answer"],
                    "predicted": "SKIPPED — PDF not found",
                    "heuristic_score": 0.0, "llm_judge": None,
                })
            continue

        uncached_qs = [q for q in doc_qs if q["financebench_id"] not in existing]
        if not uncached_qs:
            for q in doc_qs:
                results.append(existing[q["financebench_id"]])
            logger.info("[%d/%d] %s — all %d cached ✓",
                        doc_idx + 1, len(doc_questions), doc_name, len(doc_qs))
            continue

        logger.info("[%d/%d] %s (%d questions, %d new)",
                    doc_idx + 1, len(doc_questions), doc_name,
                    len(doc_qs), len(uncached_qs))

        try:
            tree = await get_tree(doc_name, ni)
        except Exception as exc:
            logger.error("  Tree load/build failed for %s: %s", doc_name, exc)
            for q in doc_qs:
                if q["financebench_id"] in existing:
                    results.append(existing[q["financebench_id"]])
                else:
                    r = {
                        "id": q["financebench_id"], "doc_name": doc_name,
                        "question": q["question"], "gold_answer": q["answer"],
                        "predicted": f"INDEX_FAIL: {exc}",
                        "heuristic_score": 0.0, "llm_judge": None,
                    }
                    results.append(r)
                    _append_result(results_cache, r)
            continue

        async def _answer_question(q, tree=tree, doc_name=doc_name, pdf_path=pdf_path):
            qid = q["financebench_id"]
            if qid in existing:
                return existing[qid]

            async with sem:
                logger.info("  Q: %s", q["question"][:80])
                try:
                    ask_kwargs: dict = {
                        "include_metadata": True,
                        "mode": answer_mode,
                        "pdf_path": pdf_path,
                    }
                    ans = await ni.async_ask(q["question"], tree, **ask_kwargs)
                    predicted = ans.content
                    reasoning = getattr(ans, "reasoning", None) or ""
                    citations = [
                        {"title": c.title, "doc_name": c.doc_name, "pages": c.pages}
                        for c in ans.citations
                    ]
                except Exception as exc:
                    logger.error("    Answer failed: %s", exc)
                    predicted = f"ERROR: {exc}"
                    reasoning = ""
                    citations = []

                h = heuristic_score(predicted, q["answer"])

                judge_result = None
                if judge_llm and not predicted.startswith("ERROR"):
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
                    "company": q.get("company"),
                    "question": q["question"],
                    "question_type": q.get("question_type"),
                    "question_reasoning": q.get("question_reasoning"),
                    "gold_answer": q["answer"],
                    "predicted": predicted,
                    "reasoning": reasoning,
                    "heuristic_score": h["heuristic_score"],
                    "exact_match": h["exact_match"],
                    "key_number_found": h["key_number_found"],
                    "llm_judge": judge_result,
                    "evidence_page_hit": page_hit,
                    "citations": citations,
                    "reasoning_model": reasoning_model,
                    "mode": answer_mode,
                }
                _append_result(results_cache, result)

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


def _append_result(path: Path, result: dict):
    with open(path, "a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------
# Post-hoc LLM judging of existing results
# ------------------------------------------------------------------

async def judge_existing_results(results: list[dict], judge_llm) -> list[dict]:
    to_judge = [r for r in results if r.get("llm_judge") is None
                and not r["predicted"].startswith(("SKIPPED", "INDEX_FAIL", "ERROR"))]

    logger.info("Judging %d answers with LLM...", len(to_judge))

    sem = asyncio.Semaphore(5)

    async def _judge_one(r):
        async with sem:
            verdict = await llm_judge(r["question"], r["predicted"], r["gold_answer"], judge_llm)
            r["llm_judge"] = verdict
            return r

    tasks = [_judge_one(r) for r in to_judge]
    await asyncio.gather(*tasks)

    return results


# ------------------------------------------------------------------
# Summary printing
# ------------------------------------------------------------------

def print_summary(results: list[dict]):
    total = len(results)
    if total == 0:
        print("No results.")
        return

    answerable = [r for r in results if not r["predicted"].startswith(("SKIPPED", "INDEX_FAIL"))]

    h_score = sum(r["heuristic_score"] for r in results) / total if total else 0
    exact = sum(1 for r in results if r.get("exact_match"))
    key_num = sum(1 for r in results if r.get("key_number_found"))

    judged = [r for r in results if r.get("llm_judge") is not None]
    judge_correct = sum(1 for r in judged if r["llm_judge"])

    page_hits = [r for r in results if r.get("evidence_page_hit") is not None]
    page_correct = sum(1 for r in page_hits if r["evidence_page_hit"])

    reasoning_models = set(r.get("reasoning_model", "unknown") for r in results)

    print(f"\n{'='*70}")
    print(f"  FinanceBench Evaluation — {total} questions")
    print(f"  Reasoning LLM: {', '.join(reasoning_models)}")
    print(f"{'='*70}")
    print(f"  Heuristic avg score: {h_score:.1%}")
    print(f"  Exact match:         {exact}/{total} ({exact/total:.1%})")
    print(f"  Key number found:    {key_num}/{total} ({key_num/total:.1%})")
    if judged:
        print(f"  LLM Judge accuracy:  {judge_correct}/{len(judged)} ({judge_correct/len(judged):.1%})")
    if page_hits:
        print(f"  Evidence page hit:   {page_correct}/{len(page_hits)} ({page_correct/len(page_hits):.1%})")

    from collections import defaultdict
    by_type = defaultdict(list)
    for r in results:
        by_type[r.get("question_type", "unknown")].append(r)

    print(f"\n  By question type:")
    for qtype, rs in sorted(by_type.items()):
        h_avg = sum(r["heuristic_score"] for r in rs) / len(rs)
        j_rs = [r for r in rs if r.get("llm_judge") is not None]
        j_str = ""
        if j_rs:
            j_acc = sum(1 for r in j_rs if r["llm_judge"]) / len(j_rs)
            j_str = f" | judge={j_acc:.0%}"
        print(f"    {qtype or 'unknown':35s}: h={h_avg:.1%}{j_str} ({len(rs)} Qs)")

    by_reason = defaultdict(list)
    for r in results:
        by_reason[r.get("question_reasoning") or "unknown"].append(r)

    print(f"\n  By reasoning type:")
    for rtype, rs in sorted(by_reason.items()):
        h_avg = sum(r["heuristic_score"] for r in rs) / len(rs)
        j_rs = [r for r in rs if r.get("llm_judge") is not None]
        j_str = ""
        if j_rs:
            j_acc = sum(1 for r in j_rs if r["llm_judge"]) / len(j_rs)
            j_str = f" | judge={j_acc:.0%}"
        print(f"    {rtype[:40]:40s}: h={h_avg:.1%}{j_str} ({len(rs)} Qs)")

    errors = [r for r in results if r["predicted"].startswith(("SKIPPED", "INDEX_FAIL", "ERROR"))]
    if errors:
        print(f"\n  Errors/skipped: {len(errors)}")
        for r in errors[:10]:
            print(f"    [{r['doc_name']}] {r['predicted'][:80]}")

    print(f"{'='*70}\n")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _resolve_reasoning_key(args) -> str | None:
    """Find the right API key for the reasoning LLM."""
    if args.reasoning_key:
        return args.reasoning_key
    model = args.reasoning_model or ""
    if model.startswith("claude"):
        return os.environ.get("ANTHROPIC_API_KEY")
    if model.startswith("gemini"):
        return os.environ.get("GOOGLE_API_KEY")
    if model.startswith("gpt"):
        return os.environ.get("OPENAI_API_KEY")
    return None


def _resolve_reasoning_url(args) -> str | None:
    """Infer the base URL from model name if not explicitly provided."""
    if args.reasoning_url:
        return args.reasoning_url
    model = args.reasoning_model or ""
    if model.startswith("claude"):
        return None  # Anthropic SDK doesn't use a base_url
    if model.startswith("gemini"):
        return "https://generativelanguage.googleapis.com/v1beta/openai/"
    if model.startswith("gpt"):
        return "https://api.openai.com/v1"
    return None


def main():
    parser = argparse.ArgumentParser(
        description="FinanceBench evaluation (PageIndex-compatible methodology)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Phase 1: build tree caches
  NANONETS_API_KEY=... python benchmarks/financebench_full.py --build-cache-only

  # Phase 2: evaluate with Claude
  NANONETS_API_KEY=... ANTHROPIC_API_KEY=... python benchmarks/financebench_full.py \\
      --reasoning-model claude-sonnet-4-20250514

  # Phase 2: evaluate with Gemini
  NANONETS_API_KEY=... GOOGLE_API_KEY=... python benchmarks/financebench_full.py \\
      --reasoning-model gemini-2.5-flash

  # Phase 2: evaluate with GPT-4o
  NANONETS_API_KEY=... OPENAI_API_KEY=... python benchmarks/financebench_full.py \\
      --reasoning-model gpt-4o
""",
    )

    phase = parser.add_argument_group("Phase control")
    phase.add_argument("--build-cache-only", action="store_true",
                       help="Only build tree caches — skip evaluation (Phase 1)")
    phase.add_argument("--judge-only", action="store_true",
                       help="Only re-judge existing results — skip evaluation")
    phase.add_argument("--clear-results", action="store_true",
                       help="Clear previous answer cache before running")

    reasoning = parser.add_argument_group("Reasoning LLM (for retrieval + answer generation)")
    reasoning.add_argument("--reasoning-model", default=None,
                           help="Model name (e.g. claude-sonnet-4-20250514, gemini-2.5-flash, gpt-4o)")
    reasoning.add_argument("--reasoning-url", default=None,
                           help="Base URL (auto-detected for Claude/Gemini/GPT)")
    reasoning.add_argument("--reasoning-key", default=None,
                           help="API key (auto-detected from env: ANTHROPIC_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY)")

    judge = parser.add_argument_group("LLM-as-judge")
    judge.add_argument("--use-llm-judge", action="store_true",
                       help="Use LLM-as-judge scoring in addition to heuristic")
    judge.add_argument("--judge-model", default="gpt-4o")
    judge.add_argument("--judge-url", default="https://api.openai.com/v1")
    judge.add_argument("--judge-key", default=None,
                       help="API key for judge (defaults to JUDGE_API_KEY env or reasoning key)")

    parser.add_argument("--limit", type=int, default=None, help="Limit to first N questions")
    parser.add_argument("--concurrency", type=int, default=3, help="Max parallel answer tasks")
    parser.add_argument("--vision", action="store_true",
                        help="Use vision mode: pass PDF page images to the LLM after retrieval")
    parser.add_argument("--agentic", action="store_true",
                        help="Use agentic retrieval: LLM iteratively reads sections before answering")

    args = parser.parse_args()

    if not QUESTIONS_PATH.exists():
        print(f"ERROR: FinanceBench data not found at {QUESTIONS_PATH}")
        print("Clone: git clone https://github.com/patronus-ai/financebench ../financebench")
        sys.exit(1)

    questions = load_questions(args.limit)
    logger.info("Loaded %d questions across %d documents",
                len(questions), len(set(q["doc_name"] for q in questions)))

    # ---- Build NanoIndex config ----
    config_kwargs: dict = {"nanonets_api_key": os.environ.get("NANONETS_API_KEY")}

    if args.reasoning_model:
        config_kwargs["reasoning_llm_model"] = args.reasoning_model
        rkey = _resolve_reasoning_key(args)
        rurl = _resolve_reasoning_url(args)
        if rkey:
            config_kwargs["reasoning_llm_api_key"] = rkey
        if rurl:
            config_kwargs["reasoning_llm_base_url"] = rurl

    ni = NanoIndex(**config_kwargs)
    ni.config.confidence_threshold = 0.0

    # ---- Phase 1: cache only ----
    if args.build_cache_only:
        logger.info("Phase 1: Building tree caches...")
        asyncio.run(build_all_caches(ni, questions))
        return

    # ---- Clear results ----
    if args.clear_results:
        results_file = RESULTS_DIR / "answers.jsonl"
        if results_file.exists():
            results_file.unlink()
            logger.info("Cleared previous results")

    # ---- Validation for Phase 2 ----
    if args.reasoning_model and not args.judge_only:
        rkey = _resolve_reasoning_key(args)
        if not rkey and not args.reasoning_model.startswith("claude"):
            logger.warning(
                "No API key found for reasoning model %s. "
                "Set the appropriate env var or use --reasoning-key.",
                args.reasoning_model,
            )

    # ---- Judge LLM setup ----
    judge_llm = None
    if args.use_llm_judge or args.judge_only:
        from nanoindex.core.llm import LLMClient
        jkey = (
            args.judge_key
            or os.environ.get("JUDGE_API_KEY")
            or _resolve_reasoning_key(args)
            or os.environ.get("OPENAI_API_KEY")
        )
        if not jkey:
            logger.error("No judge API key found. Set JUDGE_API_KEY or OPENAI_API_KEY.")
            sys.exit(1)
        judge_llm = LLMClient(
            api_key=jkey,
            base_url=args.judge_url,
            model=args.judge_model,
        )

    # ---- Judge-only mode ----
    if args.judge_only:
        results_cache = RESULTS_DIR / "answers.jsonl"
        if not results_cache.exists():
            print("ERROR: No existing results to judge")
            sys.exit(1)
        with open(results_cache) as f:
            results = [json.loads(line) for line in f if line.strip()]
        logger.info("Loaded %d existing results for re-judging", len(results))
        results = asyncio.run(judge_existing_results(results, judge_llm))
        with open(results_cache, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print_summary(results)
        return

    # ---- Phase 2: evaluate ----
    if args.agentic:
        mode = "agentic_vision" if args.vision else "agentic"
        logger.info("Agentic mode enabled (%s) — LLM iteratively reads sections", mode)
    elif args.vision:
        mode = "vision"
        logger.info("Vision mode enabled — PDF page images will be passed to the LLM")
    else:
        mode = "text"
    results = asyncio.run(run_eval(
        questions, ni, judge_llm=judge_llm,
        concurrency=args.concurrency, use_vision=args.vision,
        answer_mode=mode,
    ))

    output_path = RESULTS_DIR / "summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print_summary(results)


if __name__ == "__main__":
    main()
