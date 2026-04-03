"""Quick test: verify agentic retrieval mode on a known failure case."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nanoindex import NanoIndex
from nanoindex.utils.tree_ops import load_tree

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("test_agentic")

NANONETS_KEY = os.environ.get("NANONETS_API_KEY", "dummy-not-needed-for-cached-trees")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

DOC_NAME = "ACTIVISIONBLIZZARD_2019_10K"
QUESTION = "What is the FY2017 - FY2019 3 year average of capex as a % of revenue for Activision Blizzard? Answer in units of percents and round to one decimal place."
GOLD = "1.9%"


async def main():
    tree_path = Path(__file__).parent / "cache_v3" / f"{DOC_NAME}.json"
    pdf_path = Path(__file__).resolve().parent.parent.parent / "financebench" / "pdfs" / f"{DOC_NAME}.pdf"

    tree = load_tree(tree_path)
    logger.info("Tree loaded: %d top-level nodes", len(tree.structure))

    ni = NanoIndex(
        nanonets_api_key=NANONETS_KEY,
        reasoning_llm_model="claude-sonnet-4-20250514",
        reasoning_llm_api_key=ANTHROPIC_KEY,
    )

    logger.info("--- Testing AGENTIC + VISION mode ---")
    answer = await ni.async_ask(
        QUESTION,
        tree,
        mode="agentic_vision",
        pdf_path=str(pdf_path),
        include_metadata=True,
    )

    print("\n" + "=" * 60)
    print(f"Question: {QUESTION}")
    print(f"Gold answer: {GOLD}")
    print(f"Predicted: {answer.content}")
    print(f"Mode: {answer.mode}")
    print(f"Citations: {len(answer.citations)}")
    for c in answer.citations:
        print(f"  - [{c.node_id}] {c.title} (pp. {c.pages})")
    print("=" * 60)

    gold_in_pred = GOLD.replace("%", "").strip() in answer.content
    print(f"\nGold value '{GOLD}' found in answer: {gold_in_pred}")


if __name__ == "__main__":
    asyncio.run(main())
