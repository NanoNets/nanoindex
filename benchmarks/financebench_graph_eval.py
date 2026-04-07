"""FinanceBench evaluation with GLiNER2 graphs: fast + agentic_graph modes.

Features:
  - Incremental saving: results saved after each answer (crash-safe)
  - Resume: skips already-answered questions on restart
  - LLM-as-judge: uses Claude to score answers beyond heuristic matching

Usage:
    python benchmarks/financebench_graph_eval.py --skip-graphs
    python benchmarks/financebench_graph_eval.py --skip-graphs --limit 20
    python benchmarks/financebench_graph_eval.py --graphs-only
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

from nanoindex import NanoIndex
from nanoindex.models import DocumentTree, DocumentGraph
from nanoindex.utils.tree_ops import load_tree, iter_nodes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("financebench_graph_eval")

FINANCEBENCH_DIR = Path(__file__).resolve().parent.parent.parent / "financebench"
QUESTIONS_PATH = FINANCEBENCH_DIR / "data" / "financebench_open_source.jsonl"
PDFS_DIR = FINANCEBENCH_DIR / "pdfs"

TREE_CACHE = Path("benchmarks/cache_v3")
GRAPH_CACHE = Path("benchmarks/graphs_v4")
RESULTS_DIR = Path("benchmarks/results_graph_v4")


# ------------------------------------------------------------------
# Graph building
# ------------------------------------------------------------------

def load_graph(path: Path) -> DocumentGraph:
    with open(path) as f:
        return DocumentGraph(**json.load(f))


def save_graph(graph: DocumentGraph, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(graph.model_dump(), f)


async def build_all_graphs(ni: NanoIndex) -> dict[str, DocumentGraph]:
    GRAPH_CACHE.mkdir(parents=True, exist_ok=True)
    graphs: dict[str, DocumentGraph] = {}
    tree_files = sorted(TREE_CACHE.glob("*.json"))
    logger.info("Found %d cached trees in %s", len(tree_files), TREE_CACHE)

    for idx, tree_file in enumerate(tree_files):
        doc_name = tree_file.stem
        graph_file = GRAPH_CACHE / f"{doc_name}.json"

        if graph_file.exists():
            try:
                graph = load_graph(graph_file)
                graphs[doc_name] = graph
                logger.info("[%d/%d] Cached: %s (%de/%dr)",
                    idx + 1, len(tree_files), doc_name,
                    len(graph.entities), len(graph.relationships))
                continue
            except Exception:
                pass

        logger.info("[%d/%d] Building graph: %s", idx + 1, len(tree_files), doc_name)
        t0 = time.time()
        try:
            tree = load_tree(tree_file)
            node_count = len(list(iter_nodes(tree.structure)))
            from nanoindex.core.tree_validator import validate_tree
            validation = validate_tree(tree)
            from nanoindex.core.gliner_extractor import extract_entities_gliner_v1
            from nanoindex.core.entity_resolver import resolve_entities
            graph = extract_entities_gliner_v1(tree, skip_relationships=True)
            graph = resolve_entities(graph)
            elapsed = time.time() - t0
            logger.info("  Graph built in %.1fs: %de/%dr (%d nodes)",
                elapsed, len(graph.entities), len(graph.relationships), node_count)
            save_graph(graph, graph_file)
            graphs[doc_name] = graph
        except Exception as exc:
            logger.error("  FAILED: %s", exc)
    return graphs


# ------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------

def load_questions(limit: int | None = None) -> list[dict]:
    with open(QUESTIONS_PATH) as f:
        questions = [json.loads(line) for line in f]
    if limit:
        questions = questions[:limit]
    return questions


def heuristic_score(predicted: str, gold: str) -> dict:
    pred_norm = predicted.lower().strip().replace(",", "")
    gold_norm = gold.lower().strip().replace(",", "")
    exact_match = gold_norm in pred_norm
    pred_nums = set(re.findall(r"[\d]+\.?\d*", pred_norm))
    gold_nums = set(re.findall(r"[\d]+\.?\d*", gold_norm))
    number_overlap = bool(pred_nums & gold_nums) if gold_nums else False

    def _norm(n):
        return n.rstrip("0").rstrip(".") if "." in n else n

    gold_key = re.findall(r"\$?([\d]+\.?\d*)", gold_norm)
    gold_clean = [_norm(n) for n in gold_key]
    pred_clean = {_norm(n) for n in pred_nums}
    key_found = any(n in pred_norm or n in pred_clean for n in gold_clean) if gold_clean else False

    return {
        "exact_match": exact_match,
        "number_overlap": number_overlap,
        "key_number_found": key_found,
        "heuristic_score": 1.0 if exact_match else (0.75 if key_found else (0.5 if number_overlap else 0.0)),
    }


_LLM_JUDGE_PROMPT = """\
You are a financial accuracy judge. Compare the predicted answer to the gold answer.

Question: {question}
Gold answer: {gold}
Predicted answer: {predicted}

Rules:
1. If the predicted answer contains the same key facts/numbers as the gold answer, it is CORRECT even if worded differently or more verbose.
2. If the predicted answer has the right number but wrong units or sign, it is INCORRECT.
3. If the predicted answer says "cannot determine" or refuses but the gold answer has a clear answer, it is INCORRECT.
4. For yes/no questions, the conclusion must match.
5. For numerical answers, the key number must match (minor rounding differences like $1577 vs $1577.00 are OK).

Respond with ONLY a JSON object:
{{"correct": true/false, "reasoning": "<one sentence>"}}"""


async def llm_judge(question: str, gold: str, predicted: str, llm) -> dict:
    """Use LLM to judge if the predicted answer is correct."""
    prompt = _LLM_JUDGE_PROMPT.format(
        question=question, gold=gold, predicted=predicted,
    )
    try:
        resp = await llm.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=200)
        resp = resp.strip()
        if resp.startswith("```"):
            resp = re.sub(r"^```(?:json)?\s*", "", resp)
            resp = re.sub(r"\s*```$", "", resp)
        data = json.loads(resp)
        return {"llm_judge": data.get("correct", False), "llm_reasoning": data.get("reasoning", "")}
    except Exception as exc:
        logger.warning("LLM judge failed: %s", exc)
        return {"llm_judge": None, "llm_reasoning": str(exc)}


# ------------------------------------------------------------------
# Incremental save / resume
# ------------------------------------------------------------------

def _results_file(mode: str) -> Path:
    return RESULTS_DIR / f"{mode}_answers.jsonl"


def load_completed(mode: str) -> dict[str, dict]:
    """Load already-completed results. Returns {question_id: result}."""
    path = _results_file(mode)
    if not path.exists():
        return {}
    completed = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                completed[r["id"]] = r
    return completed


def append_result(mode: str, result: dict):
    """Append one result to the JSONL file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_results_file(mode), "a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------
# Eval
# ------------------------------------------------------------------

async def run_eval(
    questions: list[dict],
    graphs: dict[str, DocumentGraph],
    ni: NanoIndex,
    modes: list[str],
) -> dict[str, list[dict]]:
    all_results: dict[str, list[dict]] = {m: [] for m in modes}

    # Load already-completed results for resume
    completed: dict[str, dict[str, dict]] = {}
    for mode in modes:
        completed[mode] = load_completed(mode)
        if completed[mode]:
            logger.info("Resuming %s: %d already done", mode, len(completed[mode]))
            all_results[mode] = list(completed[mode].values())

    # Get LLM for judge
    llm = ni._get_reasoning_llm()

    doc_questions: dict[str, list[dict]] = {}
    for q in questions:
        doc_questions.setdefault(q["doc_name"], []).append(q)

    total_q = len(questions) * len(modes)
    done_count = sum(len(c) for c in completed.values())
    logger.info("Evaluating %d questions x %d modes = %d total (%d already done)",
                len(questions), len(modes), total_q, done_count)

    for doc_idx, (doc_name, doc_qs) in enumerate(doc_questions.items()):
        tree_file = TREE_CACHE / f"{doc_name}.json"
        pdf_path = PDFS_DIR / f"{doc_name}.pdf"

        if not tree_file.exists():
            logger.warning("Tree not cached: %s — skipping", doc_name)
            for mode in modes:
                for q in doc_qs:
                    if q["financebench_id"] not in completed[mode]:
                        result = {
                            "id": q["financebench_id"], "question": q["question"],
                            "gold_answer": q["answer"], "predicted": "SKIPPED",
                            "heuristic_score": 0.0, "score": 0.0, "doc_name": doc_name,
                            "mode": mode,
                        }
                        all_results[mode].append(result)
                        append_result(mode, result)
            continue

        tree = load_tree(tree_file)
        graph = graphs.get(doc_name)
        if graph:
            ni._graphs[doc_name] = graph

        logger.info("[%d/%d] %s: %d questions, graph=%s",
            doc_idx + 1, len(doc_questions), doc_name, len(doc_qs),
            f"{len(graph.entities)}e/{len(graph.relationships)}r" if graph else "NONE")

        for q in doc_qs:
            for mode in modes:
                # Skip if already done
                if q["financebench_id"] in completed[mode]:
                    continue

                logger.info("  [%s] Q: %s", mode, q["question"][:70])
                t0 = time.time()

                try:
                    ans = await ni.async_ask(
                        q["question"], tree, mode=mode,
                        pdf_path=str(pdf_path) if pdf_path.exists() else None,
                        include_metadata=True,
                    )
                    predicted = ans.content
                    elapsed = time.time() - t0
                    citations = [{"title": c.title, "pages": c.pages} for c in ans.citations]
                except Exception as exc:
                    elapsed = time.time() - t0
                    logger.error("    [%s] FAILED (%.1fs): %s", mode, elapsed, exc)
                    predicted = f"ERROR: {exc}"
                    citations = []

                # Heuristic score
                match = heuristic_score(predicted, q["answer"])

                # LLM judge
                judge = await llm_judge(q["question"], q["answer"], predicted, llm)

                # Evidence page check
                evidence_pages = set()
                for ev in q.get("evidence", []):
                    pg = ev.get("evidence_page_num")
                    if pg is not None:
                        evidence_pages.add(pg + 1)
                cited_pages = set()
                for c in citations:
                    cited_pages.update(c.get("pages", []))
                page_hit = bool(evidence_pages & cited_pages) if evidence_pages else None

                # Final score: LLM judge if available, else heuristic
                llm_correct = judge.get("llm_judge")
                if llm_correct is True:
                    score = 1.0
                elif llm_correct is False:
                    score = 0.0
                else:
                    score = match["heuristic_score"]

                result = {
                    "id": q["financebench_id"],
                    "question": q["question"],
                    "gold_answer": q["answer"],
                    "predicted": predicted,
                    "heuristic_score": match["heuristic_score"],
                    "exact_match": match["exact_match"],
                    "key_number_found": match["key_number_found"],
                    "llm_judge": judge.get("llm_judge"),
                    "llm_reasoning": judge.get("llm_reasoning", ""),
                    "score": score,
                    "evidence_page_hit": page_hit,
                    "question_type": q.get("question_type"),
                    "question_reasoning": q.get("question_reasoning"),
                    "doc_name": doc_name,
                    "citations": citations,
                    "mode": mode,
                    "time_s": round(elapsed, 1),
                }
                all_results[mode].append(result)
                append_result(mode, result)

                status = "+" if score >= 0.75 else "~" if score > 0 else "-"
                judge_str = "LLM:Y" if llm_correct else ("LLM:N" if llm_correct is False else "LLM:?")
                logger.info(
                    "    [%s] %s %.1fs score=%.2f %s | Gold: %s | Pred: %s",
                    mode, status, elapsed, score, judge_str,
                    q["answer"][:50], predicted[:50],
                )

    return all_results


def print_summary(all_results: dict[str, list[dict]]):
    print(f"\n{'='*80}")
    print(f"  FinanceBench Graph Evaluation Summary")
    print(f"{'='*80}")

    for mode, results in all_results.items():
        total = len(results)
        if total == 0:
            continue

        avg_score = sum(r["score"] for r in results) / total
        heuristic_avg = sum(r.get("heuristic_score", r["score"]) for r in results) / total
        llm_correct = sum(1 for r in results if r.get("llm_judge") is True)
        llm_incorrect = sum(1 for r in results if r.get("llm_judge") is False)
        llm_unknown = sum(1 for r in results if r.get("llm_judge") is None)
        exact = sum(1 for r in results if r.get("exact_match"))
        key_num = sum(1 for r in results if r.get("key_number_found"))
        page_hits = [r for r in results if r.get("evidence_page_hit") is not None]
        page_rate = sum(1 for r in page_hits if r["evidence_page_hit"]) / len(page_hits) if page_hits else 0
        avg_time = sum(r.get("time_s", 0) for r in results) / total
        errors = sum(1 for r in results if r.get("predicted", "").startswith("ERROR"))

        print(f"\n  Mode: {mode}")
        print(f"  {'─'*50}")
        print(f"  Questions:           {total}")
        print(f"  LLM Judge Accuracy:  {llm_correct}/{total} ({llm_correct/total:.1%})")
        print(f"  LLM Judge Wrong:     {llm_incorrect}/{total}")
        print(f"  Heuristic Avg:       {heuristic_avg:.1%}")
        print(f"  Exact match:         {exact}/{total} ({exact/total:.1%})")
        print(f"  Key number found:    {key_num}/{total} ({key_num/total:.1%})")
        print(f"  Evidence page hit:   {sum(1 for r in page_hits if r['evidence_page_hit'])}/{len(page_hits)} ({page_rate:.1%})")
        print(f"  Avg time/question:   {avg_time:.1f}s")
        print(f"  Errors:              {errors}")

        by_type = defaultdict(list)
        for r in results:
            by_type[r.get("question_type", "unknown")].append(r)
        print(f"\n  By question type:")
        for qtype, rs in sorted(by_type.items()):
            correct = sum(1 for r in rs if r.get("llm_judge") is True)
            print(f"    {qtype:25s}: {correct}/{len(rs)} ({correct/len(rs):.0%})")

    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--graphs-only", action="store_true")
    parser.add_argument("--skip-graphs", action="store_true")
    args = parser.parse_args()

    ni = NanoIndex(llm="anthropic:claude-sonnet-4-6")

    if args.skip_graphs:
        logger.info("Loading cached graphs from %s", GRAPH_CACHE)
        graphs = {}
        for gf in sorted(GRAPH_CACHE.glob("*.json")):
            if gf.stem != "_build_stats":
                try:
                    graphs[gf.stem] = load_graph(gf)
                except Exception:
                    pass
        logger.info("Loaded %d cached graphs", len(graphs))
    else:
        graphs = asyncio.run(build_all_graphs(ni))

    if args.graphs_only:
        return

    if not QUESTIONS_PATH.exists():
        print(f"ERROR: FinanceBench not found at {QUESTIONS_PATH}")
        sys.exit(1)

    questions = load_questions(args.limit)
    logger.info("Loaded %d questions", len(questions))

    modes = ["fast_vision", "agentic_graph_vision"]
    all_results = asyncio.run(run_eval(questions, graphs, ni, modes))

    # Save summary
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for mode, results in all_results.items():
        total = len(results)
        if total == 0:
            continue
        llm_correct = sum(1 for r in results if r.get("llm_judge") is True)
        summary[mode] = {
            "total": total,
            "llm_judge_accuracy": round(llm_correct / total, 4) if total else 0,
            "heuristic_avg": round(sum(r.get("heuristic_score", 0) for r in results) / total, 4),
            "exact_match": sum(1 for r in results if r.get("exact_match")),
            "key_number_found": sum(1 for r in results if r.get("key_number_found")),
            "avg_time_s": round(sum(r.get("time_s", 0) for r in results) / total, 1),
        }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print_summary(all_results)


if __name__ == "__main__":
    main()
