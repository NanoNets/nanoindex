"""End-to-end: index a document, search, and answer a question.

Usage:
    export NANONETS_API_KEY=your_key_here
    python examples/02_search_and_ask.py path/to/document.pdf "What was the revenue?"
"""

import sys

from nanoindex import NanoIndex


def main():
    if len(sys.argv) < 3:
        print("Usage: python 02_search_and_ask.py <pdf_path> <question>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    question = sys.argv[2]

    ni = NanoIndex()

    print(f"Indexing {pdf_path} …")
    tree = ni.index(pdf_path, add_summaries=True)

    print(f"\nSearching for: {question}")
    results = ni.search(question, tree)

    print(f"\nFound {len(results)} relevant section(s):")
    for rn in results:
        print(f"  • {rn.node.title} [{rn.node.node_id}] (pp. {rn.node.start_index}-{rn.node.end_index})")

    print("\nGenerating answer …")
    answer = ni.ask(question, tree)

    print(f"\nAnswer:\n{answer.content}")
    if answer.citations:
        print("\nCitations:")
        for c in answer.citations:
            print(f"  • {c.title} (pp. {', '.join(str(p) for p in c.pages)})")


if __name__ == "__main__":
    main()
