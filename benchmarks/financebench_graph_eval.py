"""FinanceBench evaluation with GLiNER2 graphs: fast + agentic_graph modes.

Phase 1: Build graphs for all cached trees (GLiNER2 + entity resolution + validation)
Phase 2: Run eval on fast and agentic_graph modes

Usage:
    # Build graphs only (no eval)
    python benchmarks/financebench_graph_eval.py --graphs-only

    # Run full eval (build graphs if missing, then evaluate)
    python benchmarks/financebench_graph_eval.py

    # Limit questions
    python benchmarks/financebench_graph_eval.py --limit 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
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
# Phase 1: Build graphs
# ------------------------------------------------------------------

def load_graph(path: Path) -> DocumentGraph:
    with open(path) as f:
        return DocumentGraph(**json.load(f))


def save_graph(graph: DocumentGraph, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(graph.model_dump(), f)


async def build_all_graphs(ni: NanoIndex) -> dict[str, DocumentGraph]:
    """Build GLiNER2 graphs for all cached trees."""
    GRAPH_CACHE.mkdir(parents=True, exist_ok=True)
    graphs: dict[str, DocumentGraph] = {}

    tree_files = sorted(TREE_CACHE.glob("*.json"))
    logger.info("Found %d cached trees in %s", len(tree_files), TREE_CACHE)

    for idx, tree_file in enumerate(tree_files):
        doc_name = tree_file.stem
        graph_file = GRAPH_CACHE / f"{doc_name}.json"

        # Load from cache if exists
        if graph_file.exists():
            try:
                graph = load_graph(graph_file)
                graphs[doc_name] = graph
                logger.info(
                    "[%d/%d] Loaded cached graph: %s (%d entities, %d rels)",
                    idx + 1, len(tree_files), doc_name,
                    len(graph.entities), len(graph.relationships),
                )
                continue
            except Exception:
                logger.warning("Failed to load cached graph %s, rebuilding", graph_file)

        # Build graph
        logger.info("[%d/%d] Building graph: %s", idx + 1, len(tree_files), doc_name)
        t0 = time.time()

        try:
            tree = load_tree(tree_file)
            node_count = len(list(iter_nodes(tree.structure)))

            # Validate tree first
            from nanoindex.core.tree_validator import validate_tree
            validation = validate_tree(tree)
            if not validation.passed:
                logger.warning("  Tree validation FAILED: %s", validation.errors)

            # Build graph (GLiNER v1 fast + entity resolution, skip spaCy rels for speed)
            from nanoindex.core.gliner_extractor import extract_entities_gliner_v1
            from nanoindex.core.entity_resolver import resolve_entities

            graph = extract_entities_gliner_v1(tree, skip_relationships=True)
            graph = resolve_entities(graph)

            elapsed = time.time() - t0
            logger.info(
                "  Graph built in %.1fs: %d entities, %d relationships (%d nodes, %s coverage)",
                elapsed, len(graph.entities), len(graph.relationships),
                node_count, f"{validation.stats.get('page_coverage', 0):.0%}",
            )

            # Save graph
            save_graph(graph, graph_file)
            graphs[doc_name] = graph

        except Exception as exc:
            elapsed = time.time() - t0
            logger.error("  Graph build FAILED (%.1fs): %s", elapsed, exc)
            continue

    logger.info(
        "\nGraph building complete: %d/%d graphs built",
        len(graphs), len(tree_files),
    )

    # Summary stats
    total_entities = sum(len(g.entities) for g in graphs.values())
    total_rels = sum(len(g.relationships) for g in graphs.values())
    logger.info("Total: %d entities, %d relationships across %d documents",
                total_entities, total_rels, len(graphs))

    return graphs


# ------------------------------------------------------------------
# Phase 2: Evaluation
# ------------------------------------------------------------------

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
        "number_overlap": number_overlap,
        "key_number_found": key_found,
        "score": 1.0 if exact_match else (0.75 if key_found else (0.5 if number_overlap else 0.0)),
    }


async def run_eval(
    questions: list[dict],
    graphs: dict[str, DocumentGraph],
    ni: NanoIndex,
    modes: list[str],
) -> dict[str, list[dict]]:
    """Run evaluation across multiple modes."""
    all_results: dict[str, list[dict]] = {m: [] for m in modes}

    # Group questions by document
    doc_questions: dict[str, list[dict]] = {}
    for q in questions:
        doc_questions.setdefault(q["doc_name"], []).append(q)

    logger.info("Evaluating %d questions across %d documents, modes: %s",
                len(questions), len(doc_questions), modes)

    for doc_idx, (doc_name, doc_qs) in enumerate(doc_questions.items()):
        tree_file = TREE_CACHE / f"{doc_name}.json"
        pdf_path = PDFS_DIR / f"{doc_name}.pdf"

        if not tree_file.exists():
            logger.warning("Tree not cached: %s — skipping", doc_name)
            for mode in modes:
                for q in doc_qs:
                    all_results[mode].append({
                        "id": q["financebench_id"],
                        "question": q["question"],
                        "gold_answer": q["answer"],
                        "predicted": "SKIPPED — tree not cached",
                        "score": 0.0,
                        "doc_name": doc_name,
                    })
            continue

        tree = load_tree(tree_file)
        graph = graphs.get(doc_name)

        # Store graph for this document
        if graph:
            ni._graphs[doc_name] = graph

        logger.info(
            "[%d/%d] %s: %d questions, graph=%s",
            doc_idx + 1, len(doc_questions), doc_name, len(doc_qs),
            f"{len(graph.entities)}e/{len(graph.relationships)}r" if graph else "NONE",
        )

        for q in doc_qs:
            for mode in modes:
                logger.info("  [%s] Q: %s", mode, q["question"][:70])
                t0 = time.time()

                try:
                    ans = await ni.async_ask(
                        q["question"], tree,
                        mode=mode,
                        pdf_path=str(pdf_path) if pdf_path.exists() else None,
                        include_metadata=True,
                    )
                    predicted = ans.content
                    elapsed = time.time() - t0
                    citations = [
                        {"title": c.title, "pages": c.pages}
                        for c in ans.citations
                    ]
                except Exception as exc:
                    elapsed = time.time() - t0
                    logger.error("    [%s] FAILED (%.1fs): %s", mode, elapsed, exc)
                    predicted = f"ERROR: {exc}"
                    citations = []

                match = answer_matches(predicted, q["answer"])

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
                    "mode": mode,
                    "time_s": round(elapsed, 1),
                }
                all_results[mode].append(result)

                status = "+" if match["score"] >= 0.75 else "~" if match["score"] > 0 else "-"
                logger.info(
                    "    [%s] %s %.1fs score=%.2f | Gold: %s | Pred: %s",
                    mode, status, elapsed, match["score"],
                    q["answer"][:60], predicted[:60],
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
        exact = sum(1 for r in results if r.get("exact_match"))
        key_num = sum(1 for r in results if r.get("key_number_found"))
        page_hits = [r for r in results if r.get("evidence_page_hit") is not None]
        page_rate = sum(1 for r in page_hits if r["evidence_page_hit"]) / len(page_hits) if page_hits else 0
        avg_time = sum(r.get("time_s", 0) for r in results) / total
        errors = sum(1 for r in results if r["predicted"].startswith("ERROR"))

        print(f"\n  Mode: {mode}")
        print(f"  {'─'*40}")
        print(f"  Questions:         {total}")
        print(f"  Average score:     {avg_score:.1%}")
        print(f"  Exact match:       {exact}/{total} ({exact/total:.1%})")
        print(f"  Key number found:  {key_num}/{total} ({key_num/total:.1%})")
        print(f"  Evidence page hit: {sum(1 for r in page_hits if r['evidence_page_hit'])}/{len(page_hits)} ({page_rate:.1%})")
        print(f"  Avg time/question: {avg_time:.1f}s")
        print(f"  Errors:            {errors}")

        # By question type
        by_type = defaultdict(list)
        for r in results:
            by_type[r.get("question_type", "unknown")].append(r)
        print(f"\n  By question type:")
        for qtype, rs in sorted(by_type.items()):
            avg = sum(r["score"] for r in rs) / len(rs)
            print(f"    {qtype:25s}: {avg:.1%} ({len(rs)}q)")

    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--graphs-only", action="store_true", help="Only build graphs, skip eval")
    parser.add_argument("--skip-graphs", action="store_true", help="Skip graph building, use cached")
    args = parser.parse_args()

    ni = NanoIndex(llm="anthropic:claude-sonnet-4-6")

    # Phase 1: Build graphs
    if args.skip_graphs:
        logger.info("Loading cached graphs from %s", GRAPH_CACHE)
        graphs = {}
        for gf in sorted(GRAPH_CACHE.glob("*.json")):
            try:
                graphs[gf.stem] = load_graph(gf)
            except Exception:
                pass
        logger.info("Loaded %d cached graphs", len(graphs))
    else:
        graphs = asyncio.run(build_all_graphs(ni))

    if args.graphs_only:
        logger.info("Graphs-only mode — done.")
        return

    # Phase 2: Eval
    if not QUESTIONS_PATH.exists():
        print(f"ERROR: FinanceBench data not found at {QUESTIONS_PATH}")
        print("Clone it: git clone https://github.com/patronus-ai/financebench.git")
        sys.exit(1)

    questions = load_questions(args.limit)
    logger.info("Loaded %d questions", len(questions))

    modes = ["fast", "agentic_graph"]
    all_results = asyncio.run(run_eval(questions, graphs, ni, modes))

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for mode, results in all_results.items():
        out = RESULTS_DIR / f"{mode}_answers.jsonl"
        with open(out, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("Saved %s results to %s", mode, out)

    # Save summary
    summary = {}
    for mode, results in all_results.items():
        total = len(results)
        if total == 0:
            continue
        summary[mode] = {
            "total": total,
            "avg_score": round(sum(r["score"] for r in results) / total, 4),
            "exact_match": sum(1 for r in results if r.get("exact_match")),
            "key_number_found": sum(1 for r in results if r.get("key_number_found")),
            "avg_time_s": round(sum(r.get("time_s", 0) for r in results) / total, 1),
        }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print_summary(all_results)


if __name__ == "__main__":
    main()
