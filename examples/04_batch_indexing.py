"""Batch indexing: index all PDFs in a directory.

Usage:
    export NANONETS_API_KEY=your_key_here
    python examples/04_batch_indexing.py ./documents/ ./indexes/
"""

import sys
from pathlib import Path

from nanoindex import NanoIndex
from nanoindex.utils.tree_ops import save_tree


def main():
    if len(sys.argv) < 3:
        print("Usage: python 04_batch_indexing.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDF files found in {input_dir}")
        sys.exit(1)

    ni = NanoIndex()

    for pdf in pdfs:
        print(f"\nIndexing {pdf.name} …")
        try:
            tree = ni.index(pdf, add_summaries=False)
            out_path = output_dir / f"{pdf.stem}_tree.json"
            save_tree(tree, out_path)
            print(f"  ✓ Saved to {out_path}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    print(f"\nDone. Processed {len(pdfs)} files.")


if __name__ == "__main__":
    main()
