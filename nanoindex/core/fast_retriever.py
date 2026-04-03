"""Fast retrieval: embedding search + graph expansion + LLM selection.

Three-tier retrieval that dramatically reduces LLM token usage:
  Tier 1: Embedding cosine similarity → top-K candidate nodes (1 embed call)
  Tier 1b: Entity keyword match + graph expansion → additional candidates (zero cost)
  Tier 2: LLM picks final nodes from ~25 candidates (1 small LLM call, not 300+)

Cost: 1 embed + 1 small LLM + 1 answer = ~3 calls vs 6+ in agentic mode.
"""

from __future__ import annotations

import json
import logging
import re

from nanoindex.config import NanoIndexConfig
from nanoindex.core.embedder import cosine_search, embed_query
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
    node_embeddings: dict[str, list[float]] | None = None,
    graph: DocumentGraph | None = None,
) -> list[RetrievedNode]:
    """Fast retrieval: embed → graph expand → LLM select.

    If node_embeddings is None, falls back to graph-only or direct LLM.
    If graph is None, falls back to embedding-only.
    Either or both can be provided.
    """
    candidate_ids: set[str] = set()
    all_node_ids = {n.node_id for n in iter_nodes(tree.structure)}

    # --- Tier 1a: Embedding search ---
    if node_embeddings:
        embed_key = config.embedding_api_key or config.require_llm_key()
        query_vec = await embed_query(
            query,
            api_key=embed_key,
            model=config.embedding_model,
            base_url=config.embedding_base_url,
        )
        embed_results = cosine_search(query_vec, node_embeddings, top_k=config.fast_top_k_embed)
        embed_ids = {nid for nid, _ in embed_results}
        candidate_ids |= embed_ids
        logger.info("Embedding search: %d candidates", len(embed_ids))

    # --- Tier 1b: Entity keyword match + graph expansion ---
    if graph:
        entity_to_nodes = build_entity_to_nodes(graph)

        # Direct keyword match
        keyword_ids = entity_keyword_match(query, entity_to_nodes)
        candidate_ids |= keyword_ids
        if keyword_ids:
            logger.info("Entity keyword match: %d nodes", len(keyword_ids))

        # Graph expansion from all candidates so far
        if candidate_ids and config.graph_hops > 0:
            nx_graph = build_nx_graph(graph)
            expanded = graph_expand(nx_graph, candidate_ids, entity_to_nodes, hops=config.graph_hops)
            new_from_graph = expanded - candidate_ids
            candidate_ids |= new_from_graph
            if new_from_graph:
                logger.info("Graph expansion: +%d nodes", len(new_from_graph))

    # --- Fallback: if no embeddings or graph, use all nodes (degrade to current behavior) ---
    if not candidate_ids:
        logger.warning("No embeddings or graph available — falling back to full tree")
        candidate_ids = all_node_ids

    # Filter to valid nodes
    candidate_ids &= all_node_ids
    logger.info("Total candidates before LLM selection: %d / %d nodes", len(candidate_ids), len(all_node_ids))

    # --- Tier 2: LLM selection from candidates ---
    top_k = min(config.fast_top_k_final, len(candidate_ids))

    if len(candidate_ids) <= top_k:
        # Already small enough — skip LLM call
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
            # Filter to valid candidates
            selected_ids = [nid for nid in selected_ids if nid in candidate_ids]
            if not selected_ids:
                # LLM didn't return valid IDs — fall back to top embedding scores
                selected_ids = list(candidate_ids)[:top_k]
        except Exception:
            logger.warning("LLM selection failed, using top embedding candidates", exc_info=True)
            selected_ids = list(candidate_ids)[:top_k]

    logger.info("Fast retrieval: selected %d nodes", len(selected_ids))

    # --- Build RetrievedNode results ---
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
