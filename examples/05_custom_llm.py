"""Custom LLM: use OpenAI / Anthropic / Ollama for reasoning while
keeping Nanonets for extraction and enrichment.

Usage:
    export NANONETS_API_KEY=your_nanonets_key
    export REASONING_LLM_API_KEY=sk-your-openai-key
    python examples/05_custom_llm.py path/to/document.pdf "What was Q3 revenue?"
"""

import sys

from nanoindex import NanoIndex, NanoIndexConfig


def main():
    if len(sys.argv) < 3:
        print("Usage: python 05_custom_llm.py <pdf_path> <question>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    question = sys.argv[2]

    config = NanoIndexConfig(
        nanonets_api_key="your_nanonets_key",  # Or set NANONETS_API_KEY env var

        # Default LLM — fast/cheap model for enrichment + refining
        llm_base_url="https://extraction-api.nanonets.com/v1",
        llm_model="nanonets-ocr-3",

        # Reasoning LLM — more capable model for search + answer generation
        reasoning_llm_base_url="https://api.openai.com/v1",
        reasoning_llm_api_key="sk-your-openai-key",  # Or set REASONING_LLM_API_KEY env var
        reasoning_llm_model="gpt-4o",

        add_summaries=True,
    )

    ni = NanoIndex(config)

    print(f"Indexing {pdf_path} with Nanonets extraction + default LLM enrichment …")
    tree = ni.index(pdf_path)

    print(f"\nAsking (via reasoning LLM): {question}")
    answer = ni.ask(question, tree)

    print(f"\nAnswer:\n{answer.content}")


if __name__ == "__main__":
    main()
