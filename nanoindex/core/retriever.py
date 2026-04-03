"""Tree search retriever: single-pass and layer-by-layer.

Given a user query and a ``DocumentTree``, the retriever asks an LLM to
navigate the tree structure and select the most relevant leaf nodes.

Two strategies:
  - **Single-pass** — show the entire tree outline in one prompt (fast, works
    when the outline fits within the context window).
  - **Layer-by-layer** — iteratively expand selected branches across multiple
    rounds, keeping each prompt under the context limit.  Required for very
    large documents.
"""

from __future__ import annotations

import json
import logging
import re

from nanoindex.config import NanoIndexConfig
from nanoindex.core.llm import LLMClient
from nanoindex.models import DocumentTree, RetrievedNode, TreeNode
from nanoindex.utils.tokens import count_tokens
from nanoindex.utils.tree_ops import collect_text, find_node, tree_to_outline

logger = logging.getLogger(__name__)

_CONTEXT_BUDGET = 120_000  # tokens reserved for the tree outline in a prompt

_SEARCH_USER = """\
Below is a document outline. Pick the 1-5 most specific section IDs that \
answer the question. Prefer child/leaf sections over broad parent sections. \
For example, pick "0003.0001" instead of "0003" if the child is more relevant.

Return ONLY a JSON array of IDs, e.g. ["0003.0001", "0004.0002"].

Question: {query}

{outline}

JSON array:"""

_LAYER_USER = """\
From this document outline, select which top-level sections are most relevant \
to the question. Return ONLY a JSON array of IDs.

Question: {query}

{outline}

Return a JSON array of IDs:"""

_EXPAND_USER = """\
Narrow down to the most relevant sub-sections for the question. \
Return ONLY a JSON array of IDs.

Question: {query}

{outline}

Return a JSON array of IDs:"""


# ------------------------------------------------------------------
# Node-id parsing helper
# ------------------------------------------------------------------

def _parse_node_ids(text: str) -> list[str]:
    """Robustly extract a JSON array of node-id strings from LLM output."""
    text = text.strip()
    if "```" in text:
        text = re.sub(r"```(?:json|JSON)?\s*", "", text)
        text = text.strip()
    # Try direct JSON parse first
    try:
        arr = json.loads(text)
        if isinstance(arr, list):
            return [str(x) for x in arr]
    except json.JSONDecodeError:
        pass
    # Fallback: extract first JSON array from the text
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            arr = json.loads(match.group())
            if isinstance(arr, list):
                return [str(x) for x in arr]
        except json.JSONDecodeError:
            pass
    # Last resort: extract anything that looks like a node_id
    return re.findall(r"\b(\d{4}(?:\.\d{4})*)\b", text)


# ------------------------------------------------------------------
# Fuzzy node resolution — walk up the ID hierarchy on miss
# ------------------------------------------------------------------

def _resolve_node(structure: list[TreeNode], nid: str) -> TreeNode | None:
    """Try *nid*, then progressively strip trailing segments to find an ancestor.

    E.g. ``0059.0001.0001`` → ``0059.0001`` → ``0059``.
    """
    candidate = nid
    while candidate:
        node = find_node(structure, candidate)
        if node is not None:
            if candidate != nid:
                logger.info("Resolved hallucinated '%s' → existing '%s'", nid, candidate)
            return node
        if "." not in candidate:
            break
        candidate = candidate.rsplit(".", 1)[0]
    return None


# ------------------------------------------------------------------
# Single-pass search
# ------------------------------------------------------------------

async def _single_pass_search(
    query: str,
    tree: DocumentTree,
    llm: LLMClient,
) -> list[str]:
    """Show the full tree outline and ask the LLM for relevant node IDs."""
    outline = tree_to_outline(tree.structure)
    prompt = _SEARCH_USER.format(query=query, outline=outline)
    messages = [{"role": "user", "content": prompt}]
    resp = await llm.chat(messages, temperature=0.0, max_tokens=1024)
    ids = _parse_node_ids(resp)
    logger.debug("Single-pass LLM response: %s", resp[:200])
    if not ids:
        logger.warning("Single-pass returned no IDs. Raw response: %s", resp[:300])
    else:
        logger.info("Single-pass selected %d IDs: %s", len(ids), ids[:10])
    return ids


# ------------------------------------------------------------------
# Layer-by-layer search
# ------------------------------------------------------------------

def _outline_for_nodes(nodes: list[TreeNode]) -> str:
    """Build an outline string showing only the given nodes (one level)."""
    lines: list[str] = []
    for node in nodes:
        line = f"- [{node.node_id}] {node.title}"
        if node.summary:
            line += f": {node.summary}"
        if node.start_index and node.end_index:
            line += f" (pp. {node.start_index}-{node.end_index})"
        child_count = len(node.nodes)
        if child_count:
            line += f" [{child_count} sub-sections]"
        lines.append(line)
    return "\n".join(lines)


async def _layer_search(
    query: str,
    tree: DocumentTree,
    llm: LLMClient,
    max_rounds: int = 3,
) -> list[str]:
    """Iteratively drill into the tree, one layer at a time."""
    current_nodes = tree.structure
    selected_ids: list[str] = []

    for round_num in range(max_rounds):
        outline = _outline_for_nodes(current_nodes)

        if round_num == 0:
            prompt = _LAYER_USER.format(query=query, outline=outline)
        else:
            prompt = _EXPAND_USER.format(query=query, outline=outline)

        messages = [{"role": "user", "content": prompt}]
        resp = await llm.chat(messages, temperature=0.0, max_tokens=1024)
        ids = _parse_node_ids(resp)

        if not ids:
            break

        # Resolve selected nodes
        selected: list[TreeNode] = []
        for nid in ids:
            n = find_node(tree.structure, nid)
            if n:
                selected.append(n)

        if not selected:
            selected_ids.extend(ids)
            break

        # If selected nodes have no children, we're done
        children_to_expand: list[TreeNode] = []
        for n in selected:
            if n.nodes:
                children_to_expand.extend(n.nodes)
            else:
                selected_ids.append(n.node_id)

        if not children_to_expand:
            selected_ids.extend(n.node_id for n in selected if n.node_id not in selected_ids)
            break

        current_nodes = children_to_expand

    return selected_ids


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

async def search(
    query: str,
    tree: DocumentTree,
    llm: LLMClient,
    config: NanoIndexConfig,
) -> list[RetrievedNode]:
    """Search the document tree for nodes relevant to *query*.

    Automatically picks single-pass or layer-by-layer depending on
    whether the full outline fits in the context budget.
    """
    outline = tree_to_outline(tree.structure)
    outline_tokens = count_tokens(outline)

    if outline_tokens <= _CONTEXT_BUDGET:
        node_ids = await _single_pass_search(query, tree, llm)
    else:
        logger.info(
            "Outline is %d tokens (budget %d) — using layer-by-layer search",
            outline_tokens, _CONTEXT_BUDGET,
        )
        node_ids = await _layer_search(query, tree, llm)

    if not node_ids:
        return []

    _MAX_RESULTS = 8
    seen_input: set[str] = set()
    seen_resolved: set[str] = set()
    results: list[RetrievedNode] = []
    for nid in node_ids:
        if nid in seen_input:
            continue
        seen_input.add(nid)
        node = _resolve_node(tree.structure, nid)
        if node is None:
            logger.warning("LLM selected non-existent node_id '%s' (no ancestor found)", nid)
            continue
        if node.node_id in seen_resolved:
            continue
        seen_resolved.add(node.node_id)
        text = node.text or collect_text(node)
        results.append(RetrievedNode(node=node, text=text))
        if len(results) >= _MAX_RESULTS:
            break

    return results
