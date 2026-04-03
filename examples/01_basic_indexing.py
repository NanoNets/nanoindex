"""Basic indexing: extract a PDF and build a tree index.

Usage:
    export NANONETS_API_KEY=your_key_here
    python examples/01_basic_indexing.py path/to/document.pdf
"""

import sys
from pathlib import Path

from nanoindex import NanoIndex
from nanoindex.utils.tree_ops import save_tree

def main():
    if len(sys.argv) < 2:
        print("Usage: python 01_basic_indexing.py <pdf_path> [output.json]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else f"{Path(pdf_path).stem}_tree.json"

    ni = NanoIndex()

    print(f"Indexing {pdf_path} …")
    tree = ni.index(pdf_path, add_summaries=False)

    save_tree(tree, output_path)
    print(f"Tree saved to {output_path}")
    print(f"  Nodes: {sum(1 for _ in _iter(tree.structure))}")
    print(f"  Pages: {tree.extraction_metadata.get('pages_processed', '?')}")


def _iter(nodes):
    for n in nodes:
        yield n
        yield from _iter(n.nodes)


if __name__ == "__main__":
    main()
