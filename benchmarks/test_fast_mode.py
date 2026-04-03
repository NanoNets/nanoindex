"""Test fast retrieval mode on a few FinanceBench questions.

Builds graph + embeddings for test documents, then compares:
  - agentic_vision (current, expensive)
  - fast (new, cheap)

Usage:
    ANTHROPIC_API_KEY=... OPENAI_API_KEY=... python benchmarks/test_fast_mode.py
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
from nanoindex.core.entity_extractor import save_graph
from nanoindex.core.embedder import save_embeddings
from nanoindex.core.llm import LLMClient
from nanoindex.utils.tree_ops import load_tree

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_fast")

FINANCEBENCH_DIR = Path(__file__).resolve().parent.parent.parent / "financebench"
QUESTIONS_PATH = FINANCEBENCH_DIR / "data" / "financebench_open_source.jsonl"
PDFS_DIR = FINANCEBENCH_DIR / "pdfs"
TREE_CACHE = Path(__file__).resolve().parent / "cache_v3"
GRAPH_CACHE = Path(__file__).resolve().parent / "graph_cache"
EMBED_CACHE = Path(__file__).resolve().parent / "embed_cache"

# Test on first 10 questions
NUM_TEST = 10

_JUDGE_PROMPT = """\
Determine if the AI answer correctly answers the query based on the golden answer.
Rounding differences are OK. Same meaning = correct.
Query: {query}
AI Answer: {predicted}
Golden Answer: {gold}
Output ONLY: True or False"""


async def judge(query, predicted, gold, llm):
    try:
        resp = await llm.chat(
            [{"role": "user", "content": _JUDGE_PROMPT.format(query=query, predicted=predicted[:1500], gold=gold)}],
            temperature=0.0, max_tokens=64,
        )
        return "true" in resp.strip().lower()
    except:
        return None


async def main():
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if not anthropic_key:
        print("ERROR: Set ANTHROPIC_API_KEY")
        sys.exit(1)

    # Load questions
    with open(QUESTIONS_PATH) as f:
        all_questions = [json.loads(line) for line in f]
    questions = all_questions[:NUM_TEST]

    # Set up NanoIndex with fast mode config
    ni = NanoIndex(
        nanonets_api_key=os.environ.get("NANONETS_API_KEY"),
        reasoning_llm_model="claude-sonnet-4-6",
        reasoning_llm_api_key=anthropic_key,
        embedding_model="local:all-MiniLM-L6-v2",
        build_graph=True,
        build_embeddings=True,
    )

    judge_llm = LLMClient(api_key=anthropic_key, model="claude-sonnet-4-6")

    # Build graph + embeddings for test documents
    GRAPH_CACHE.mkdir(parents=True, exist_ok=True)
    EMBED_CACHE.mkdir(parents=True, exist_ok=True)

    test_docs = set(q["doc_name"] for q in questions)
    logger.info("Building graph + embeddings for %d documents", len(test_docs))

    for doc_name in sorted(test_docs):
        tree_path = TREE_CACHE / f"{doc_name}.json"
        graph_path = GRAPH_CACHE / f"{doc_name}.json"
        embed_path = EMBED_CACHE / f"{doc_name}.npz"

        if not tree_path.exists():
            logger.warning("No tree for %s, skipping", doc_name)
            continue

        tree = load_tree(tree_path)

        # Build or load graph
        if graph_path.exists():
            logger.info("Loading cached graph for %s", doc_name)
            ni.load_graph(doc_name, graph_path)
        else:
            logger.info("Building graph for %s...", doc_name)
            graph = await ni.async_build_graph(tree)
            save_graph(graph, graph_path)

        # Build or load embeddings
        if embed_path.exists():
            logger.info("Loading cached embeddings for %s", doc_name)
            ni.load_embeddings(doc_name, embed_path)
        else:
            logger.info("Building embeddings for %s...", doc_name)
            embeddings = await ni.async_build_embeddings(tree)
            save_embeddings(embeddings, embed_path)

    # Run test: fast mode vs agentic_vision
    results = []
    for idx, q in enumerate(questions):
        doc_name = q["doc_name"]
        tree_path = TREE_CACHE / f"{doc_name}.json"
        pdf_path = PDFS_DIR / f"{doc_name}.pdf"

        if not tree_path.exists() or not pdf_path.exists():
            continue

        tree = load_tree(tree_path)
        logger.info("\n[%d/%d] %s", idx + 1, len(questions), q["question"][:80])

        # --- Fast mode ---
        t0 = time.time()
        try:
            fast_ans = await ni.async_ask(q["question"], tree, mode="fast", pdf_path=pdf_path)
            fast_pred = fast_ans.content
        except Exception as exc:
            fast_pred = f"ERROR: {exc}"
        fast_time = time.time() - t0

        fast_judge = await judge(q["question"], fast_pred, q["answer"], judge_llm) if not fast_pred.startswith("ERROR") else None

        # --- Agentic vision mode ---
        t0 = time.time()
        try:
            agent_ans = await ni.async_ask(q["question"], tree, mode="agentic_vision", pdf_path=pdf_path)
            agent_pred = agent_ans.content
        except Exception as exc:
            agent_pred = f"ERROR: {exc}"
        agent_time = time.time() - t0

        agent_judge = await judge(q["question"], agent_pred, q["answer"], judge_llm) if not agent_pred.startswith("ERROR") else None

        fast_status = "✅" if fast_judge else "❌"
        agent_status = "✅" if agent_judge else "❌"

        logger.info("  FAST:    %s (%.1fs) | Pred: %s", fast_status, fast_time, fast_pred[:80])
        logger.info("  AGENTIC: %s (%.1fs) | Pred: %s", agent_status, agent_time, agent_pred[:80])
        logger.info("  GOLD: %s", q["answer"][:80])

        results.append({
            "question": q["question"],
            "gold": q["answer"],
            "fast_judge": fast_judge,
            "fast_time": round(fast_time, 1),
            "agent_judge": agent_judge,
            "agent_time": round(agent_time, 1),
        })

    # Summary
    fast_correct = sum(1 for r in results if r["fast_judge"])
    agent_correct = sum(1 for r in results if r["agent_judge"])
    fast_avg_time = sum(r["fast_time"] for r in results) / len(results) if results else 0
    agent_avg_time = sum(r["agent_time"] for r in results) / len(results) if results else 0

    print(f"\n{'='*60}")
    print(f"  Fast Mode Test — {len(results)} questions")
    print(f"{'='*60}")
    print(f"  Fast mode:    {fast_correct}/{len(results)} correct | avg {fast_avg_time:.1f}s")
    print(f"  Agentic mode: {agent_correct}/{len(results)} correct | avg {agent_avg_time:.1f}s")
    print(f"  Speedup:      {agent_avg_time/fast_avg_time:.1f}x faster" if fast_avg_time > 0 else "")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
