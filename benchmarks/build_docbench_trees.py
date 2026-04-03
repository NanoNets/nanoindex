"""Build NanoIndex trees for DocBench PDFs.

Usage:
    NANONETS_API_KEY=your_key_here \
    python benchmarks/build_docbench_trees.py
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
from nanoindex.core.extractor import extract_document
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
logger = logging.getLogger("build_trees")

DOCBENCH_DIR = Path(__file__).resolve().parent.parent.parent / "DocBench"
DATA_DIR = DOCBENCH_DIR / "data"

BENCH_DIR = Path(__file__).resolve().parent
TREE_CACHE = BENCH_DIR / "docbench_tree_cache"


async def build_tree(pdf_path: Path, ni: NanoIndex) -> None:
    doc_name = pdf_path.stem
    tree_path = TREE_CACHE / f"{doc_name}.json"

    if tree_path.exists():
        return

    TREE_CACHE.mkdir(parents=True, exist_ok=True)

    client = ni._get_client()
    t0 = time.time()
    extraction = await extract_document(pdf_path, client)
    logger.info("  Extracted %s: %d pages (%.1fs)",
                doc_name[:50], extraction.page_count, time.time() - t0)

    tree = build_document_tree(extraction, doc_name, ni.config)
    llm = ni._get_llm()
    tree = await refine_tree(tree, llm, ni.config)
    tree = await enrich_tree(tree, llm, ni.config)

    n_nodes = sum(1 for _ in iter_nodes(tree.structure))
    logger.info("  Built tree: %d nodes (%.1fs)", n_nodes, time.time() - t0)
    save_tree(tree, tree_path)


async def main():
    api_key = os.environ.get("NANONETS_API_KEY")
    if not api_key:
        print("ERROR: Set NANONETS_API_KEY")
        sys.exit(1)

    ni = NanoIndex(nanonets_api_key=api_key)

    # Find all PDFs in DocBench data folders
    pdfs = sorted(DATA_DIR.glob("*/*.pdf"))
    cached = [p for p in pdfs if (TREE_CACHE / f"{p.stem}.json").exists()]
    to_build = [p for p in pdfs if not (TREE_CACHE / f"{p.stem}.json").exists()]

    logger.info("DocBench tree builder: %d PDFs total, %d cached, %d to build",
                len(pdfs), len(cached), len(to_build))

    errors = []
    for idx, pdf_path in enumerate(to_build):
        logger.info("[%d/%d] %s", idx + 1, len(to_build), pdf_path.name[:60])
        try:
            await build_tree(pdf_path, ni)
        except Exception as exc:
            logger.error("  FAILED %s: %s", pdf_path.name[:50], str(exc)[:100])
            errors.append((pdf_path.name, str(exc)[:100]))

    final_cached = sum(1 for p in pdfs if (TREE_CACHE / f"{p.stem}.json").exists())
    logger.info("\nComplete: %d/%d trees built", final_cached, len(pdfs))
    if errors:
        logger.warning("Errors (%d):", len(errors))
        for name, err in errors:
            logger.warning("  %s: %s", name[:50], err)


if __name__ == "__main__":
    asyncio.run(main())
