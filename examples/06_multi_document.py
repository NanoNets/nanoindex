"""Multi-document search: index several PDFs and query across them.

Demonstrates all three document selection strategies:
  1. Direct — search all docs (or specific ones)
  2. Metadata — filter by structured fields (company, year, type)
  3. Description — LLM picks relevant docs from their descriptions

Usage:
    export NANONETS_API_KEY=your_key
    python examples/06_multi_document.py <pdf_dir> "What was 3M's revenue in 2022?"

    # With a reasoning LLM (e.g. GPT-4o):
    export REASONING_LLM_API_KEY=sk-...
    python examples/06_multi_document.py <pdf_dir> "Compare 3M and Adobe revenue" \
        --reasoning-model gpt-4o --reasoning-url https://api.openai.com/v1
"""

import argparse
import sys
from pathlib import Path

from nanoindex import NanoIndex, NanoIndexConfig, DocumentStore
from nanoindex.utils.tree_ops import save_tree, load_tree


def build_store(pdf_dir: Path, ni: NanoIndex, cache_dir: Path) -> DocumentStore:
    """Index all PDFs in a directory and build a DocumentStore."""
    store = DocumentStore()
    cache_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {pdf_dir}")
        sys.exit(1)

    for pdf in pdfs:
        tree_cache = cache_dir / f"{pdf.stem}.json"
        if tree_cache.exists():
            print(f"  Loading cached: {pdf.name}")
            tree = load_tree(tree_cache)
        else:
            print(f"  Indexing: {pdf.name} …")
            tree = ni.index(pdf, add_summaries=True, add_doc_description=True)
            save_tree(tree, tree_cache)

        metadata = _extract_metadata(pdf.stem)
        store.add(tree, metadata=metadata)

    return store


def _extract_metadata(filename: str) -> dict:
    """Heuristic: parse company, year, and type from FinanceBench filenames.

    Examples: "3M_2022_10K" → {"company": "3M", "year": 2022, "doc_type": "10-K"}
    """
    parts = filename.split("_")
    meta = {"filename": filename}
    if len(parts) >= 3:
        meta["company"] = parts[0]
        year_str = parts[1].replace("Q1", "").replace("Q2", "").replace("Q3", "").replace("Q4", "")
        try:
            meta["year"] = int(year_str)
        except ValueError:
            pass
        raw_type = parts[2].upper()
        if "10K" in raw_type:
            meta["doc_type"] = "10-K"
        elif "10Q" in raw_type:
            meta["doc_type"] = "10-Q"
        else:
            meta["doc_type"] = raw_type
    return meta


def main():
    parser = argparse.ArgumentParser(description="Multi-document search with NanoIndex")
    parser.add_argument("pdf_dir", help="Directory containing PDF files")
    parser.add_argument("question", help="Question to ask across all documents")
    parser.add_argument("--strategy", default="direct", choices=["direct", "metadata", "description"])
    parser.add_argument("--company", help="Filter by company (metadata strategy)")
    parser.add_argument("--year", type=int, help="Filter by year (metadata strategy)")
    parser.add_argument("--cache-dir", default="examples/multi_doc_cache")
    parser.add_argument("--reasoning-model", help="Reasoning LLM model (e.g. gpt-4o)")
    parser.add_argument("--reasoning-url", help="Reasoning LLM base URL")
    parser.add_argument("--reasoning-key", help="Reasoning LLM API key")
    args = parser.parse_args()

    config_kwargs = {}
    if args.reasoning_model:
        config_kwargs["reasoning_llm_model"] = args.reasoning_model
    if args.reasoning_url:
        config_kwargs["reasoning_llm_base_url"] = args.reasoning_url
    if args.reasoning_key:
        config_kwargs["reasoning_llm_api_key"] = args.reasoning_key

    ni = NanoIndex(**config_kwargs)

    print(f"\n=== Building document store from {args.pdf_dir} ===\n")
    store = build_store(Path(args.pdf_dir), ni, Path(args.cache_dir))
    print(f"\nStore contains {store.count} documents\n")

    for entry in store.list_documents():
        desc = (entry.description or "")[:60]
        print(f"  [{entry.doc_id}] {entry.metadata} — {desc}")

    strategy = args.strategy
    filters = None

    if strategy == "metadata":
        filters = {}
        if args.company:
            filters["company"] = args.company
        if args.year:
            filters["year"] = args.year
        if not filters:
            print("\nError: --company or --year required for metadata strategy")
            sys.exit(1)
        print(f"\n=== Searching with metadata filter: {filters} ===\n")

    elif strategy == "description":
        print(f"\n=== Searching with LLM description matching ===\n")

    else:
        print(f"\n=== Searching all {store.count} documents (direct) ===\n")

    answer = ni.multi_ask(
        args.question,
        store,
        strategy=strategy,
        filters=filters,
    )

    print(f"Question: {args.question}\n")
    print(f"Answer:\n{answer.content}\n")

    if answer.citations:
        print("Citations:")
        for c in answer.citations:
            doc_tag = f"[{c.doc_name}] " if c.doc_name else ""
            print(f"  {doc_tag}{c.title} (pp. {c.pages})")


if __name__ == "__main__":
    main()
