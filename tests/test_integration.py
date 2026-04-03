"""Integration test — exercises the full NanoIndex pipeline against the live Nanonets API.

Usage:
    export NANONETS_API_KEY=your_key_here
    python -m pytest tests/test_integration.py -v -s

Or run directly:
    python tests/test_integration.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("integration_test")

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_PDF = FIXTURES / "sample_report.pdf"
OUTPUT_DIR = Path(__file__).parent / "output"


def require_api_key() -> str:
    key = os.environ.get("NANONETS_API_KEY", "")
    if not key or key == "your_nanonets_api_key_here":
        print(
            "\n  NANONETS_API_KEY not set. Run:\n"
            "    export NANONETS_API_KEY=<your key>\n"
            "    python tests/test_integration.py\n"
        )
        sys.exit(1)
    return key


# ──────────────────────────────────────────────────────────────────
# Step 1: Raw API call — verify extraction endpoints work
# ──────────────────────────────────────────────────────────────────

async def test_raw_extraction(api_key: str) -> dict:
    """Test the Nanonets client directly — markdown + hierarchy extraction."""
    from nanoindex.core.client import NanonetsClient

    client = NanonetsClient(api_key=api_key)

    logger.info("─── Step 1a: Markdown extraction with bounding boxes ───")
    md_resp = await client.extract_sync(
        SAMPLE_PDF,
        output_format="markdown",
        include_metadata="bounding_boxes",
    )
    logger.info("  Status: %s", md_resp.get("status"))
    logger.info("  Processing time: %ss", md_resp.get("processing_time"))

    result = md_resp.get("result", {})
    md_obj = result.get("markdown", {})
    if isinstance(md_obj, dict):
        content = md_obj.get("content", "")
        metadata = md_obj.get("metadata", {})
        bb_meta = metadata.get("bounding_boxes", {})
        elements = bb_meta.get("elements", [])
        logger.info("  Markdown length: %d chars", len(content))
        logger.info("  Bounding box elements: %d", len(elements))
        logger.info("  First 200 chars: %s", content[:200].replace("\n", "\\n"))
    else:
        logger.info("  Markdown (raw): %s", str(md_obj)[:200])

    logger.info("\n─── Step 1b: Hierarchy JSON extraction ───")
    hier_resp = await client.extract_sync(
        SAMPLE_PDF,
        output_format="json",
        json_options="hierarchy_output",
    )
    logger.info("  Status: %s", hier_resp.get("status"))

    result_h = hier_resp.get("result", {})
    json_obj = result_h.get("json", {})
    if isinstance(json_obj, dict):
        content_h = json_obj.get("content", {})
        doc = content_h.get("document", content_h)
        sections = doc.get("sections", [])
        tables = doc.get("tables", [])
        kv_pairs = doc.get("key_value_pairs", [])
        logger.info("  Sections: %d", len(sections))
        logger.info("  Tables: %d", len(tables))
        logger.info("  KV pairs: %d", len(kv_pairs))
        for s in sections[:5]:
            logger.info("    Section: '%s' (level %s)", s.get("title") or s.get("heading"), s.get("level"))
    else:
        logger.info("  JSON (raw): %s", str(json_obj)[:300])

    await client.close()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "raw_markdown_response.json", "w") as f:
        json.dump(md_resp, f, indent=2, ensure_ascii=False)
    with open(OUTPUT_DIR / "raw_hierarchy_response.json", "w") as f:
        json.dump(hier_resp, f, indent=2, ensure_ascii=False)
    logger.info("  Raw responses saved to tests/output/")

    return {"markdown": md_resp, "hierarchy": hier_resp}


# ──────────────────────────────────────────────────────────────────
# Step 2: Extractor — parse responses into ExtractionResult
# ──────────────────────────────────────────────────────────────────

async def test_extractor(api_key: str):
    """Test the extractor module — parallel extraction + parsing."""
    from nanoindex.core.client import NanonetsClient
    from nanoindex.core.extractor import extract_document

    logger.info("\n─── Step 2: Document extraction (parallel calls) ───")

    client = NanonetsClient(api_key=api_key)
    extraction = await extract_document(SAMPLE_PDF, client)

    logger.info("  Markdown length: %d chars", len(extraction.markdown))
    logger.info("  Hierarchy sections: %d", len(extraction.hierarchy_sections))
    logger.info("  Hierarchy tables: %d", len(extraction.hierarchy_tables))
    logger.info("  Hierarchy KV pairs: %d", len(extraction.hierarchy_kv_pairs))
    logger.info("  Bounding boxes: %d", len(extraction.bounding_boxes))
    logger.info("  Page count: %d", extraction.page_count)
    logger.info("  Processing time: %.2fs", extraction.processing_time)

    for sec in extraction.hierarchy_sections[:5]:
        logger.info("    Section: '%s' (level %d) — %d chars content", sec.title, sec.level, len(sec.content))
        for sub in sec.subsections[:3]:
            logger.info("      Sub: '%s' (level %d)", sub.title, sub.level)

    if extraction.bounding_boxes:
        for bb in extraction.bounding_boxes[:5]:
            logger.info("    BBox: page=%d, type=%s, conf=%.2f, text='%s'",
                        bb.page, bb.region_type, bb.confidence, (bb.text or "")[:50])

    await client.close()
    return extraction


# ──────────────────────────────────────────────────────────────────
# Step 3: Tree builder — build tree from extraction result
# ──────────────────────────────────────────────────────────────────

def test_tree_builder(extraction):
    """Test tree building from the extraction result."""
    from nanoindex.config import load_config
    from nanoindex.core.tree_builder import build_document_tree

    logger.info("\n─── Step 3: Tree building ───")

    config = load_config(nanonets_api_key="dummy")
    config.confidence_threshold = 0.0  # Don't filter for testing
    tree = build_document_tree(extraction, "sample_report", config)

    logger.info("  Doc name: %s", tree.doc_name)
    logger.info("  Top-level nodes: %d", len(tree.structure))
    logger.info("  Extraction metadata: %s", tree.extraction_metadata)

    from nanoindex.utils.tree_ops import iter_nodes, tree_to_outline
    all_nodes = list(iter_nodes(tree.structure))
    logger.info("  Total nodes: %d", len(all_nodes))

    outline = tree_to_outline(tree.structure)
    logger.info("  Outline:\n%s", outline)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    from nanoindex.utils.tree_ops import save_tree
    save_tree(tree, OUTPUT_DIR / "tree_no_summaries.json")
    logger.info("  Tree saved to tests/output/tree_no_summaries.json")

    return tree


# ──────────────────────────────────────────────────────────────────
# Step 4: Enricher — add summaries via LLM
# ──────────────────────────────────────────────────────────────────

async def test_enricher(tree, api_key: str):
    """Test summary generation using the Nanonets chat API."""
    from nanoindex.config import load_config
    from nanoindex.core.enricher import enrich_tree
    from nanoindex.core.llm import LLMClient

    logger.info("\n─── Step 4: Summary enrichment ───")

    config = load_config(nanonets_api_key=api_key)
    config.add_summaries = True
    config.add_doc_description = True

    llm = LLMClient(
        api_key=api_key,
        base_url=config.llm_base_url,
        model=config.llm_model,
    )

    tree = await enrich_tree(tree, llm, config)

    from nanoindex.utils.tree_ops import iter_nodes
    for node in iter_nodes(tree.structure):
        logger.info("  [%s] %s → summary: %s",
                     node.node_id, node.title, (node.summary or "")[:80])

    if tree.doc_description:
        logger.info("  Doc description: %s", tree.doc_description[:200])

    from nanoindex.utils.tree_ops import save_tree
    save_tree(tree, OUTPUT_DIR / "tree_with_summaries.json")
    logger.info("  Enriched tree saved to tests/output/tree_with_summaries.json")

    await llm.close()
    return tree


# ──────────────────────────────────────────────────────────────────
# Step 5: Retriever — tree search
# ──────────────────────────────────────────────────────────────────

async def test_retriever(tree, api_key: str):
    """Test tree search with a sample question."""
    from nanoindex.config import load_config
    from nanoindex.core.llm import LLMClient
    from nanoindex.core.retriever import search

    logger.info("\n─── Step 5: Tree search ───")

    config = load_config(nanonets_api_key=api_key)
    llm = LLMClient(
        api_key=api_key,
        base_url=config.llm_base_url,
        model=config.llm_model,
    )

    queries = [
        "What was the revenue for Q4 2025?",
        "What are the risk factors mentioned?",
        "What is the FY 2026 guidance?",
    ]

    all_results = []
    for q in queries:
        logger.info("  Query: '%s'", q)
        results = await search(q, tree, llm, config)
        logger.info("    Found %d node(s):", len(results))
        for rn in results:
            logger.info("      • [%s] %s (pp. %d-%d) — %d chars text",
                         rn.node.node_id, rn.node.title,
                         rn.node.start_index, rn.node.end_index,
                         len(rn.text))
        all_results.append((q, results))

    await llm.close()
    return all_results


# ──────────────────────────────────────────────────────────────────
# Step 6: Generator — answer generation
# ──────────────────────────────────────────────────────────────────

async def test_generator(tree, api_key: str):
    """Test end-to-end answer generation."""
    from nanoindex.config import load_config
    from nanoindex.core.generator import generate_answer
    from nanoindex.core.llm import LLMClient
    from nanoindex.core.retriever import search

    logger.info("\n─── Step 6: Answer generation ───")

    config = load_config(nanonets_api_key=api_key)
    llm = LLMClient(
        api_key=api_key,
        base_url=config.llm_base_url,
        model=config.llm_model,
    )

    question = "What was the total revenue and how did cloud services perform?"
    logger.info("  Question: '%s'", question)

    nodes = await search(question, tree, llm, config)
    answer = await generate_answer(question, nodes, llm, mode="text")

    logger.info("  Answer: %s", answer.content[:500])
    logger.info("  Citations: %s", [(c.title, c.pages) for c in answer.citations])

    await llm.close()
    return answer


# ──────────────────────────────────────────────────────────────────
# Step 7: High-level API — NanoIndex class
# ──────────────────────────────────────────────────────────────────

def test_high_level_api(api_key: str):
    """Test the public NanoIndex API (sync wrappers)."""
    from nanoindex import NanoIndex
    from nanoindex.utils.tree_ops import save_tree

    logger.info("\n─── Step 7: High-level NanoIndex API ───")

    ni = NanoIndex(nanonets_api_key=api_key)

    logger.info("  Indexing document...")
    tree = ni.index(SAMPLE_PDF, add_summaries=True, add_doc_description=True)

    logger.info("  Tree built: %d top-level nodes", len(tree.structure))
    save_tree(tree, OUTPUT_DIR / "tree_highlevel.json")

    question = "What is the company's FY 2026 revenue guidance?"
    logger.info("  Searching: '%s'", question)
    results = ni.search(question, tree)
    logger.info("  Found %d node(s)", len(results))

    logger.info("  Asking: '%s'", question)
    answer = ni.ask(question, tree)
    logger.info("  Answer: %s", answer.content[:300])

    ni.close()
    return tree, answer


# ──────────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────────

async def run_all():
    api_key = require_api_key()

    print("\n" + "=" * 70)
    print("  NanoIndex Integration Test")
    print("  PDF: %s" % SAMPLE_PDF)
    print("=" * 70)

    if not SAMPLE_PDF.exists():
        print(f"  ERROR: {SAMPLE_PDF} not found. Run the PDF creation script first.")
        sys.exit(1)

    # Step 1: Raw API
    raw = await test_raw_extraction(api_key)

    # Step 2: Extractor
    extraction = await test_extractor(api_key)

    # Step 3: Tree builder
    tree = test_tree_builder(extraction)

    # Step 4: Enrichment (requires LLM)
    tree = await test_enricher(tree, api_key)

    # Step 5: Retrieval
    search_results = await test_retriever(tree, api_key)

    # Step 6: Answer generation
    answer = await test_generator(tree, api_key)

    print("\n" + "=" * 70)
    print("  All integration tests passed!")
    print("  Outputs saved to: tests/output/")
    print("=" * 70)

    # Step 7: High-level (runs sync, so outside the async block)
    return api_key


def main():
    api_key = asyncio.run(run_all())

    print("\n  Running high-level API test (sync)...")
    test_high_level_api(api_key)

    print("\n" + "=" * 70)
    print("  ALL TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
