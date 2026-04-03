"""Vision-based query: use page images for multimodal answer generation.

Usage:
    export NANONETS_API_KEY=your_key_here
    python examples/03_vision_query.py path/to/document.pdf "Describe Figure 3"
"""

import sys

from nanoindex import NanoIndex


def main():
    if len(sys.argv) < 3:
        print("Usage: python 03_vision_query.py <pdf_path> <question>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    question = sys.argv[2]

    ni = NanoIndex()

    print(f"Indexing {pdf_path} …")
    tree = ni.index(pdf_path)

    print(f"\nGenerating vision-based answer for: {question}")
    answer = ni.ask(question, tree, mode="vision", pdf_path=pdf_path)

    print(f"\nAnswer:\n{answer.content}")
    if answer.citations:
        print("\nCitations:")
        for c in answer.citations:
            print(f"  • {c.title} (pp. {', '.join(str(p) for p in c.pages)})")


if __name__ == "__main__":
    main()
