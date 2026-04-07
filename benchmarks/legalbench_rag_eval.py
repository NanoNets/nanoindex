"""LegalBench-RAG evaluation for NanoIndex.

Implements the RetrievalMethod interface from LegalBench-RAG and evaluates
NanoIndex's entity-graph retrieval against the character-level benchmark.

Usage:
    python benchmarks/legalbench_rag_eval.py
    python benchmarks/legalbench_rag_eval.py --limit 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("legalbench_rag")

BENCHMARK_DIR = Path("benchmarks/LegalBench-RAG")
CORPUS_DIR = BENCHMARK_DIR / "corpus"
RESULTS_DIR = Path("benchmarks/results_legalbench_rag")


# ------------------------------------------------------------------
# NanoIndex Retrieval Method
# ------------------------------------------------------------------

class NanoIndexRetrieval:
    """NanoIndex-based retrieval for LegalBench-RAG.

    For each document:
    1. Build a text tree (section-aware splitting with char offsets)
    2. Extract entities with GLiNER (legal_contract labels)
    3. Build entity graph

    For each query:
    1. Entity keyword match against the graph
    2. Graph expansion to find related sections
    3. Return character spans of matching tree nodes
    """

    def __init__(self):
        self.documents: dict[str, str] = {}  # file_path -> content
        self.trees: dict[str, object] = {}   # file_path -> DocumentTree
        self.graphs: dict[str, object] = {}  # file_path -> DocumentGraph
        self.node_spans: dict[str, list[tuple[str, int, int]]] = {}  # file_path -> [(node_id, start, end)]

    async def ingest_document(self, file_path: str, content: str) -> None:
        """Build tree + entity graph for a document."""
        from nanoindex.core.text_tree_builder import build_text_tree
        from nanoindex.core.gliner_extractor import extract_entities_gliner
        from nanoindex.core.entity_resolver import resolve_entities
        from nanoindex.utils.tree_ops import iter_nodes

        self.documents[file_path] = content

        # Build tree with character span tracking
        tree = build_text_tree(content, doc_name=file_path)
        self.trees[file_path] = tree

        # Track character spans for each node
        spans = []
        pos = 0
        for node in iter_nodes(tree.structure):
            node_text = node.text or ""
            if not node_text.strip():
                continue
            # Find the node's text in the original content
            idx = content.find(node_text[:100], pos)
            if idx >= 0:
                spans.append((node.node_id, idx, idx + len(node_text)))
                pos = idx
            elif hasattr(node, '_char_span'):
                s, e = node._char_span
                spans.append((node.node_id, s, e))
        self.node_spans[file_path] = spans

        # Extract entities and build graph
        try:
            graph = extract_entities_gliner(tree)
            graph = resolve_entities(graph)
            self.graphs[file_path] = graph
        except Exception as exc:
            logger.warning("Entity extraction failed for %s: %s", file_path, exc)

    def query(self, query: str, top_k: int = 5) -> list[dict]:
        """Find relevant text spans across all documents.

        Returns list of {"file_path": str, "span": (start, end), "score": float}
        """
        from nanoindex.core.document_index import DocumentIndex
        from nanoindex.utils.tree_ops import iter_nodes

        all_results: list[dict] = []

        for file_path, tree in self.trees.items():
            graph = self.graphs.get(file_path)
            if not graph:
                continue

            # Build document index for this doc
            idx = DocumentIndex(tree, graph)

            # Query using entity matching + graph expansion
            candidates = idx.query_nodes(query, max_results=top_k, hops=2)

            # Map node IDs back to character spans
            node_span_map = {nid: (s, e) for nid, s, e in self.node_spans.get(file_path, [])}

            for node_id, score in candidates:
                if node_id in node_span_map:
                    start, end = node_span_map[node_id]
                    all_results.append({
                        "file_path": file_path,
                        "span": (start, end),
                        "score": score,
                    })

        # Sort by score, limit to top_k
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def compute_precision_recall(
    retrieved: list[dict],
    ground_truth: list[dict],
) -> tuple[float, float]:
    """Character-level precision and recall."""
    total_retrieved_len = 0
    relevant_retrieved_len = 0

    for r in retrieved:
        r_start, r_end = r["span"]
        total_retrieved_len += r_end - r_start

        for gt in ground_truth:
            if r["file_path"] == gt["file_path"]:
                gt_start, gt_end = gt["span"]
                common_start = max(r_start, gt_start)
                common_end = min(r_end, gt_end)
                if common_end > common_start:
                    relevant_retrieved_len += common_end - common_start

    total_relevant_len = 0
    relevant_found_len = 0

    for gt in ground_truth:
        gt_start, gt_end = gt["span"]
        total_relevant_len += gt_end - gt_start

        for r in retrieved:
            if r["file_path"] == gt["file_path"]:
                r_start, r_end = r["span"]
                common_start = max(r_start, gt_start)
                common_end = min(r_end, gt_end)
                if common_end > common_start:
                    relevant_found_len += common_end - common_start

    precision = relevant_retrieved_len / total_retrieved_len if total_retrieved_len > 0 else 0
    recall = relevant_found_len / total_relevant_len if total_relevant_len > 0 else 0

    return precision, recall


async def run_eval(limit: int | None = None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    retrieval = NanoIndexRetrieval()

    # Load all benchmarks
    all_tests = []
    for name in ["cuad", "maud", "contractnli", "privacy_qa"]:
        bench_file = BENCHMARK_DIR / "benchmarks" / f"{name}.json"
        if not bench_file.exists():
            logger.warning("Benchmark file not found: %s", bench_file)
            continue
        data = json.load(open(bench_file))
        tests = data["tests"]
        for t in tests:
            t["_dataset"] = name
        all_tests.extend(tests)

    if limit:
        all_tests = all_tests[:limit]

    logger.info("Total queries: %d", len(all_tests))

    # Find all unique documents needed
    doc_paths = set()
    for t in all_tests:
        for s in t["snippets"]:
            doc_paths.add(s["file_path"])

    logger.info("Ingesting %d documents...", len(doc_paths))
    t0 = time.time()

    for i, doc_path in enumerate(sorted(doc_paths)):
        full_path = CORPUS_DIR / doc_path
        if not full_path.exists():
            logger.warning("Document not found: %s", full_path)
            continue
        content = full_path.read_text()
        await retrieval.ingest_document(doc_path, content)
        if (i + 1) % 10 == 0:
            logger.info("  Ingested %d/%d documents", i + 1, len(doc_paths))

    ingest_time = time.time() - t0
    logger.info("Ingestion complete in %.1fs", ingest_time)

    # Run queries
    results = []
    precisions = []
    recalls = []

    for i, test in enumerate(all_tests):
        query = test["query"]
        ground_truth = [{"file_path": s["file_path"], "span": tuple(s["span"])} for s in test["snippets"]]

        retrieved = retrieval.query(query, top_k=5)

        precision, recall = compute_precision_recall(retrieved, ground_truth)
        precisions.append(precision)
        recalls.append(recall)

        result = {
            "query": query,
            "dataset": test.get("_dataset", "unknown"),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "retrieved_count": len(retrieved),
            "ground_truth_count": len(ground_truth),
        }
        results.append(result)

        # Incremental save
        with open(RESULTS_DIR / "answers.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")

        if (i + 1) % 50 == 0:
            avg_p = sum(precisions) / len(precisions)
            avg_r = sum(recalls) / len(recalls)
            logger.info(
                "  [%d/%d] Avg Precision: %.1f%%, Avg Recall: %.1f%%",
                i + 1, len(all_tests), avg_p * 100, avg_r * 100,
            )

    # Final summary
    from collections import defaultdict
    by_dataset = defaultdict(lambda: {"p": [], "r": []})
    for r in results:
        by_dataset[r["dataset"]]["p"].append(r["precision"])
        by_dataset[r["dataset"]]["r"].append(r["recall"])

    print(f"\n{'='*60}")
    print(f"  LegalBench-RAG Results — NanoIndex")
    print(f"{'='*60}")
    avg_p = sum(precisions) / len(precisions) if precisions else 0
    avg_r = sum(recalls) / len(recalls) if recalls else 0
    print(f"  Overall:  Precision={avg_p:.1%}  Recall={avg_r:.1%}  F1={2*avg_p*avg_r/(avg_p+avg_r):.1%}" if avg_p + avg_r > 0 else "  No results")
    print()
    for ds, vals in sorted(by_dataset.items()):
        p = sum(vals["p"]) / len(vals["p"])
        r = sum(vals["r"]) / len(vals["r"])
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0
        print(f"  {ds:15s}: P={p:.1%}  R={r:.1%}  F1={f1:.1%}  ({len(vals['p'])} queries)")
    print(f"{'='*60}")

    # Save summary
    summary = {
        "total_queries": len(results),
        "avg_precision": round(avg_p, 4),
        "avg_recall": round(avg_r, 4),
        "ingest_time_s": round(ingest_time, 1),
        "by_dataset": {
            ds: {
                "precision": round(sum(v["p"]) / len(v["p"]), 4),
                "recall": round(sum(v["r"]) / len(v["r"]), 4),
                "count": len(v["p"]),
            }
            for ds, v in by_dataset.items()
        },
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Results saved to %s", RESULTS_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(run_eval(args.limit))


if __name__ == "__main__":
    main()
