"""Rebuild all FinanceBench trees with bounding boxes + spaCy graphs.

Uses V1 API (best tree structure + bboxes) and spaCy (free graph).

Usage:
    NANONETS_API_KEY=... python benchmarks/rebuild_financebench.py
    NANONETS_API_KEY=... python benchmarks/rebuild_financebench.py --limit 10
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
from nanoindex.utils.tree_ops import save_tree, iter_nodes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rebuild")

FINANCEBENCH_DIR = Path(__file__).resolve().parent.parent.parent / "financebench"
PDF_DIR = FINANCEBENCH_DIR / "pdfs"
TREE_CACHE = Path(__file__).resolve().parent / "cache_v3"
GRAPH_CACHE = Path(__file__).resolve().parent / "graph_cache"


async def rebuild_one(pdf_path: Path, ni: NanoIndex) -> dict:
    """Rebuild tree + graph for one document."""
    doc_name = pdf_path.stem
    tree_path = TREE_CACHE / f"{doc_name}.json"
    graph_path = GRAPH_CACHE / f"{doc_name}.json"

    t0 = time.time()
    tree = await ni.async_index(pdf_path)
    index_time = time.time() - t0

    nodes = list(iter_nodes(tree.structure))
    bboxes = len(tree.all_bounding_boxes)
    graph = ni._graphs.get(doc_name)
    entities = len(graph.entities) if graph else 0
    rels = len(graph.relationships) if graph else 0

    # Save tree
    save_tree(tree, tree_path)

    # Save graph
    if graph:
        GRAPH_CACHE.mkdir(parents=True, exist_ok=True)
        with open(graph_path, "w") as f:
            json.dump(graph.model_dump(), f, ensure_ascii=False)

    return {
        "doc": doc_name,
        "nodes": len(nodes),
        "bboxes": bboxes,
        "entities": entities,
        "relationships": rels,
        "time": round(index_time, 1),
    }


async def main():
    api_key = os.environ.get("NANONETS_API_KEY")
    if not api_key:
        print("ERROR: Set NANONETS_API_KEY")
        sys.exit(1)

    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    ni = NanoIndex(nanonets_api_key=api_key)

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if limit:
        pdfs = pdfs[:limit]

    # Skip already rebuilt (check for bboxes in existing tree)
    to_rebuild = []
    skipped = 0
    for pdf in pdfs:
        tree_path = TREE_CACHE / f"{pdf.stem}.json"
        if tree_path.exists():
            try:
                with open(tree_path) as f:
                    data = json.load(f)
                has_bboxes = bool(data.get("all_bounding_boxes"))
                graph_path = GRAPH_CACHE / f"{pdf.stem}.json"
                has_graph = graph_path.exists()
                if has_bboxes and has_graph:
                    skipped += 1
                    continue
            except Exception:
                pass
        to_rebuild.append(pdf)

    logger.info("FinanceBench rebuild: %d total, %d already done, %d to rebuild",
                len(pdfs), skipped, len(to_rebuild))

    results = []
    errors = []

    for idx, pdf in enumerate(to_rebuild):
        logger.info("[%d/%d] %s", idx + 1, len(to_rebuild), pdf.name[:50])
        try:
            r = await rebuild_one(pdf, ni)
            results.append(r)
            logger.info("  Done: %d nodes, %d bboxes, %d entities, %d rels (%.0fs)",
                       r["nodes"], r["bboxes"], r["entities"], r["relationships"], r["time"])
        except Exception as exc:
            logger.error("  FAILED: %s", str(exc)[:100])
            errors.append({"doc": pdf.stem, "error": str(exc)[:200]})

    # Summary
    total_nodes = sum(r["nodes"] for r in results)
    total_bboxes = sum(r["bboxes"] for r in results)
    total_entities = sum(r["entities"] for r in results)
    total_rels = sum(r["relationships"] for r in results)
    total_time = sum(r["time"] for r in results)

    logger.info("\n" + "=" * 60)
    logger.info("REBUILD COMPLETE")
    logger.info("=" * 60)
    logger.info("  Documents: %d rebuilt, %d skipped, %d errors", len(results), skipped, len(errors))
    logger.info("  Nodes: %d total", total_nodes)
    logger.info("  Bounding boxes: %d total", total_bboxes)
    logger.info("  Entities: %d total", total_entities)
    logger.info("  Relationships: %d total", total_rels)
    logger.info("  Time: %.0fs (%.0fs avg)", total_time, total_time / max(len(results), 1))

    if errors:
        logger.warning("Errors (%d):", len(errors))
        for e in errors:
            logger.warning("  %s: %s", e["doc"][:40], e["error"][:80])


if __name__ == "__main__":
    asyncio.run(main())
