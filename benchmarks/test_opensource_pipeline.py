"""Test the full open-source pipeline: PyMuPDF parser -> tree -> graph -> stats.

Runs without any API key for parsing. The entity-graph step requires an LLM key
(NANONETS_API_KEY or LLM_API_KEY) and is skipped gracefully when unavailable.
"""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nanoindex import NanoIndex
from nanoindex.core.parsers import get_parser
from nanoindex.core.context_extractor import enrich_modal_contexts
from nanoindex.core.tree_builder import build_document_tree
from nanoindex.utils.tree_ops import iter_nodes


async def main():
    test_pdf = Path("tests/fixtures/q1-fy25-earnings.pdf")
    if not test_pdf.exists():
        print("No test PDF found at tests/fixtures/q1-fy25-earnings.pdf — skipping.")
        return

    # ------------------------------------------------------------------
    # Step 1: Parse PDF with PyMuPDF (no API key needed)
    # ------------------------------------------------------------------
    print("1. Parsing with PyMuPDF (no API key needed)...")
    parser = get_parser("pymupdf")
    parsed = await parser.parse(test_pdf)
    print(f"   Pages: {parsed.page_count}")
    print(f"   Modal items: {len(parsed.modal_contents)}")

    # ------------------------------------------------------------------
    # Step 2: Enrich modal content with surrounding context
    # ------------------------------------------------------------------
    print("\n2. Enriching modal content with surrounding context...")
    enrich_modal_contexts(parsed)
    enriched = sum(1 for m in parsed.modal_contents if m.surrounding_text)
    print(f"   Enriched {enriched}/{len(parsed.modal_contents)} modal items with context")

    # ------------------------------------------------------------------
    # Step 3: Build tree from parsed document
    # ------------------------------------------------------------------
    print("\n3. Building tree from parsed document...")
    ni = NanoIndex(parser="pymupdf")
    extraction = parsed.to_extraction_result()
    tree = build_document_tree(extraction, test_pdf.stem, ni.config)
    node_count = sum(1 for _ in iter_nodes(tree.structure))
    print(f"   Tree nodes: {node_count}")

    # ------------------------------------------------------------------
    # Step 4: Build entity graph (needs LLM key)
    # ------------------------------------------------------------------
    has_llm_key = bool(
        os.environ.get("NANONETS_API_KEY")
        or os.environ.get("LLM_API_KEY")
    )

    graph = None
    modal_ents = []
    if has_llm_key:
        print("\n4. Building entity graph (including multimodal entities)...")
        graph = await ni.async_build_graph(tree, parsed)
        modal_ents = [e for e in graph.entities if e.entity_type in ("Image", "Table")]
        print(f"   Entities: {len(graph.entities)}")
        print(f"   Relationships: {len(graph.relationships)}")
        print(f"   Modal entities (Image + Table): {len(modal_ents)}")
    else:
        print("\n4. Skipping entity graph (no NANONETS_API_KEY or LLM_API_KEY set).")

    # ------------------------------------------------------------------
    # Step 5: Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Pages:              {parsed.page_count}")
    print(f"  Tree nodes:         {node_count}")
    if graph is not None:
        print(f"  Total entities:     {len(graph.entities)}")
        print(f"  Modal entities:     {len(modal_ents)}")
        print(f"  Relationships:      {len(graph.relationships)}")
    else:
        print("  Total entities:     (skipped — no LLM key)")
        print("  Modal entities:     (skipped — no LLM key)")
        print("  Relationships:      (skipped — no LLM key)")
    print("=" * 60)
    print("\nDone! Open-source parsing works without any API key.")
    if not has_llm_key:
        print("Set NANONETS_API_KEY or LLM_API_KEY to also run the graph step.")


if __name__ == "__main__":
    asyncio.run(main())
