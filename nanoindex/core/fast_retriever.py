"""Fast retrieval: graph-based navigation + LLM selection.

No vectors. No embeddings. Pure graph traversal.

  1. Extract entity keywords from the question
  2. Look up matching entities in the graph → find source tree nodes
  3. Expand via graph relationships → add related nodes
  4. LLM picks final nodes from the candidates (1 small LLM call)

Cost: 1 LLM call for selection + 1 for answer = 2 calls total.
"""

from __future__ import annotations

import json
import logging
import re

from nanoindex.config import NanoIndexConfig
from nanoindex.core.graph_builder import (
    build_entity_to_nodes,
    build_nx_graph,
    entity_keyword_match,
    graph_expand,
)
from nanoindex.core.llm import LLMClient
from nanoindex.models import (
    DocumentGraph,
    DocumentTree,
    RetrievedNode,
)
from nanoindex.utils.tree_ops import collect_text, find_node, iter_nodes

logger = logging.getLogger(__name__)


_SELECT_PROMPT = """\
You are a document retrieval expert. Given a question and a list of \
candidate document sections, pick the {top_k} most relevant sections \
that are likely to contain the answer.

Return ONLY a JSON array of section IDs, e.g. ["0003.0001", "0004.0002"].

Question: {query}

Candidate sections:
{candidates}

JSON array of the {top_k} most relevant IDs:"""


def _build_candidate_outline(tree: DocumentTree, candidate_ids: set[str]) -> str:
    """Build a compact outline of only the candidate nodes."""
    lines: list[str] = []
    for node in iter_nodes(tree.structure):
        if node.node_id in candidate_ids:
            line = f"[{node.node_id}] {node.title}"
            if node.summary:
                line += f": {node.summary[:200]}"
            if node.start_index and node.end_index:
                line += f" (pp. {node.start_index}-{node.end_index})"
            lines.append(line)
    return "\n".join(lines)


def _parse_node_ids(text: str) -> list[str]:
    """Extract node IDs from LLM response."""
    text = text.strip()
    if "```" in text:
        text = re.sub(r"```(?:json|JSON)?\s*", "", text)
    try:
        arr = json.loads(text)
        if isinstance(arr, list):
            return [str(x) for x in arr]
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            arr = json.loads(match.group())
            if isinstance(arr, list):
                return [str(x) for x in arr]
        except json.JSONDecodeError:
            pass
    return re.findall(r"\b(\d{4}(?:\.\d{4})*)\b", text)


async def fast_search(
    query: str,
    tree: DocumentTree,
    llm: LLMClient,
    config: NanoIndexConfig,
    *,
    node_embeddings: dict | None = None,  # Kept for backward compat, ignored
    graph: DocumentGraph | None = None,
) -> list[RetrievedNode]:
    """Fast retrieval using graph navigation. No embeddings.

    If a graph is available: entity keyword match → graph expand → LLM select.
    If no graph: falls back to full tree outline (same as agentic round 1).
    """
    candidate_ids: set[str] = set()
    all_node_ids = {n.node_id for n in iter_nodes(tree.structure)}

    # --- Graph-based retrieval ---
    if graph and graph.entities:
        entity_to_nodes = build_entity_to_nodes(graph)

        # Step 1: Direct keyword match — find entities mentioned in the question
        keyword_ids = entity_keyword_match(query, entity_to_nodes)
        candidate_ids |= keyword_ids
        if keyword_ids:
            logger.info("Entity keyword match: %d nodes", len(keyword_ids))

        # Step 2: Graph expansion — follow relationships to related nodes
        if candidate_ids and config.graph_hops > 0:
            nx_graph = build_nx_graph(graph)
            expanded = graph_expand(nx_graph, candidate_ids, entity_to_nodes, hops=config.graph_hops)
            new_from_graph = expanded - candidate_ids
            candidate_ids |= new_from_graph
            if new_from_graph:
                logger.info("Graph expansion: +%d nodes", len(new_from_graph))

    # --- Fallback: no graph → use all nodes ---
    if not candidate_ids:
        logger.info("No graph available — using full tree for selection")
        candidate_ids = all_node_ids

    # Filter to valid nodes
    candidate_ids &= all_node_ids
    logger.info("Candidates: %d / %d nodes", len(candidate_ids), len(all_node_ids))

    # --- LLM selection from candidates ---
    top_k = min(config.fast_top_k_final, len(candidate_ids))

    if len(candidate_ids) <= top_k:
        selected_ids = list(candidate_ids)
    else:
        outline = _build_candidate_outline(tree, candidate_ids)
        prompt = _SELECT_PROMPT.format(query=query, candidates=outline, top_k=top_k)
        try:
            resp = await llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
            )
            selected_ids = _parse_node_ids(resp)
            selected_ids = [nid for nid in selected_ids if nid in candidate_ids]
            if not selected_ids:
                selected_ids = list(candidate_ids)[:top_k]
        except Exception:
            logger.warning("LLM selection failed, using top candidates", exc_info=True)
            selected_ids = list(candidate_ids)[:top_k]

    logger.info("Fast retrieval: selected %d nodes", len(selected_ids))

    # --- Build results ---
    results: list[RetrievedNode] = []
    for nid in selected_ids:
        node = find_node(tree.structure, nid)
        if node:
            text = collect_text(node)
            results.append(RetrievedNode(
                node=node,
                text=text,
                doc_name=tree.doc_name,
            ))

    return results
