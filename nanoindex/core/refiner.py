"""Recursive splitting of oversized tree nodes.

Two-phase strategy per node:

  **Phase A** — heuristic: split on sub-headings already present in the
  node's text (markdown ``#``/``##``/``###`` patterns).  Zero LLM cost.

  **Phase B** — LLM-assisted: ask the model to identify 3-8 logical
  subsection titles, then split the text at those boundaries.

  **Fallback** — paragraph-boundary chunking (same as the old
  ``_split_large_nodes`` but invoked only as a last resort).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from nanoindex.config import NanoIndexConfig
from nanoindex.core.llm import LLMClient
from nanoindex.models import DocumentTree, TreeNode
from nanoindex.utils.tokens import count_tokens
from nanoindex.utils.tree_ops import assign_node_ids, iter_nodes

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

_SPLIT_PROMPT = """\
You are a document structure expert. The following text is from a section \
titled "{title}". Break it into 3-8 logical subsections.

Return ONLY a JSON array of subsection titles in document order:
["<title 1>", "<title 2>", ...]

Text (first {char_limit} characters):
{text}

JSON:"""

_MAX_CONCURRENT = 3
_MAX_REFINE_PASSES = 4
_LLM_CHAR_LIMIT = 40_000


async def refine_tree(
    tree: DocumentTree,
    llm: LLMClient,
    config: NanoIndexConfig,
) -> DocumentTree:
    """Split oversized leaf nodes recursively.  Modifies *tree* in-place."""
    changed = True
    iteration = 0

    while changed and iteration < _MAX_REFINE_PASSES:
        changed = False
        iteration += 1

        oversized = _find_oversized(tree.structure, config.max_node_tokens)
        if not oversized:
            break

        logger.info(
            "Refine pass %d: %d oversized node(s) to split",
            iteration,
            len(oversized),
        )

        sem = asyncio.Semaphore(_MAX_CONCURRENT)

        async def _split_one(node: TreeNode) -> bool:
            async with sem:
                return await _split_node(node, llm, config)

        results = await asyncio.gather(*[_split_one(n) for n in oversized])
        changed = any(results)

    assign_node_ids(tree.structure)
    return tree


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _find_oversized(
    nodes: list[TreeNode],
    max_tokens: int,
) -> list[TreeNode]:
    """Find leaf nodes whose text exceeds *max_tokens*."""
    result: list[TreeNode] = []
    for node in iter_nodes(nodes):
        if node.nodes:
            continue
        token_count = count_tokens(node.text) if node.text else 0
        if token_count > max_tokens:
            result.append(node)
    return result


async def _split_node(
    node: TreeNode,
    llm: LLMClient,
    config: NanoIndexConfig,
) -> bool:
    """Try to split an oversized node.  Returns ``True`` on success."""
    page_span = max(node.end_index - node.start_index + 1, 1)

    # Phase A: sub-heading split (always attempted first)
    if node.text and _try_heading_split(node):
        logger.info("Phase A split '%s' into %d children", node.title, len(node.nodes))
        return True

    # Phase B: LLM split — only for multi-page nodes worth the LLM cost
    if (
        config.split_strategy in ("llm", "hybrid")
        and page_span > config.max_node_pages
        and node.text
    ):
        if await _try_llm_split(node, llm):
            logger.info("Phase B (LLM) split '%s' into %d children", node.title, len(node.nodes))
            return True

    # Fallback: paragraph chunking
    if node.text and count_tokens(node.text) > config.max_node_tokens:
        _paragraph_split(node, config.max_node_tokens)
        logger.info("Fallback split '%s' into %d chunks", node.title, len(node.nodes))
        return True

    return False


# ------------------------------------------------------------------
# Phase A — sub-heading split
# ------------------------------------------------------------------


def _try_heading_split(node: TreeNode) -> bool:
    """Split a node's text on internal markdown headings."""
    if not node.text:
        return False

    matches = list(_HEADING_RE.finditer(node.text))
    if len(matches) < 2:
        return False

    prefix_text = node.text[: matches[0].start()].strip()

    for i, m in enumerate(matches):
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(node.text)
        section_text = node.text[start:end].strip()
        if not section_text:
            continue
        node.nodes.append(
            TreeNode(
                title=title,
                level=node.level + 1,
                text=section_text,
                start_index=node.start_index,
                end_index=node.end_index,
            )
        )

    if node.nodes:
        node.text = prefix_text or None
        _estimate_child_pages(node)
        return True
    return False


# ------------------------------------------------------------------
# Phase B — LLM-assisted split
# ------------------------------------------------------------------


async def _try_llm_split(node: TreeNode, llm: LLMClient) -> bool:
    """Ask the LLM to identify subsection boundaries in the node's text."""
    if not node.text:
        return False

    prompt = _SPLIT_PROMPT.format(
        title=node.title,
        char_limit=_LLM_CHAR_LIMIT,
        text=node.text[:_LLM_CHAR_LIMIT],
    )

    try:
        response = await llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
        )
        titles = _parse_json_titles(response)
    except Exception:
        logger.warning("LLM split failed for '%s'", node.title, exc_info=True)
        return False

    if len(titles) < 2:
        return False

    return _split_text_by_titles(node, titles)


def _parse_json_titles(response: str) -> list[str]:
    """Extract a list of title strings from the LLM response."""
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(t).strip() for t in data if t]
    except json.JSONDecodeError:
        pass

    array_match = re.search(r"\[.*\]", text, re.DOTALL)
    if array_match:
        try:
            data = json.loads(array_match.group())
            if isinstance(data, list):
                return [str(t).strip() for t in data if t]
        except json.JSONDecodeError:
            pass

    return []


def _split_text_by_titles(node: TreeNode, titles: list[str]) -> bool:
    """Split node text at positions matching the given titles."""
    text = node.text or ""
    text_lower = text.lower()

    positions: list[tuple[int, str]] = []
    for title in titles:
        idx = text_lower.find(title.lower())
        if idx >= 0:
            positions.append((idx, title))

    if len(positions) < 2:
        return _split_text_evenly(node, titles)

    positions.sort(key=lambda p: p[0])

    prefix_text = text[: positions[0][0]].strip()

    for i, (pos, title) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        section_text = text[pos:end].strip()
        if not section_text:
            continue
        node.nodes.append(
            TreeNode(
                title=title,
                level=node.level + 1,
                text=section_text,
                start_index=node.start_index,
                end_index=node.end_index,
            )
        )

    if node.nodes:
        node.text = prefix_text or None
        _estimate_child_pages(node)
        return True
    return False


def _split_text_evenly(node: TreeNode, titles: list[str]) -> bool:
    """Split node text roughly evenly using the given titles."""
    text = node.text or ""
    if not text or len(titles) < 2:
        return False

    paragraphs = text.split("\n\n")
    if len(paragraphs) < len(titles):
        return False

    chunk_size = len(paragraphs) // len(titles)

    for i, title in enumerate(titles):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < len(titles) - 1 else len(paragraphs)
        chunk_text = "\n\n".join(paragraphs[start:end]).strip()
        if not chunk_text:
            continue
        node.nodes.append(
            TreeNode(
                title=title,
                level=node.level + 1,
                text=chunk_text,
                start_index=node.start_index,
                end_index=node.end_index,
            )
        )

    if node.nodes:
        node.text = None
        _estimate_child_pages(node)
        return True
    return False


# ------------------------------------------------------------------
# Fallback — paragraph chunking
# ------------------------------------------------------------------


def _paragraph_split(node: TreeNode, max_tokens: int) -> None:
    """Last-resort paragraph-boundary chunking."""
    paragraphs = (node.text or "").split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        pt = count_tokens(para)
        if current_tokens + pt > max_tokens and current:
            chunks.append("\n\n".join(current))
            current = []
            current_tokens = 0
        current.append(para)
        current_tokens += pt

    if current:
        chunks.append("\n\n".join(current))

    for i, chunk in enumerate(chunks):
        node.nodes.append(
            TreeNode(
                title=f"{node.title} (part {i + 1})",
                level=node.level + 1,
                text=chunk,
                start_index=node.start_index,
                end_index=node.end_index,
            )
        )

    if node.nodes:
        node.text = None
        _estimate_child_pages(node)


# ------------------------------------------------------------------
# Page-range estimation
# ------------------------------------------------------------------


def _estimate_child_pages(parent: TreeNode) -> None:
    """Distribute parent's page range across children proportionally."""
    if not parent.nodes:
        return
    total_chars = sum(len(c.text or "") for c in parent.nodes)
    if total_chars == 0:
        return

    page_span = parent.end_index - parent.start_index + 1
    current_page = parent.start_index

    for child in parent.nodes:
        child_chars = len(child.text or "")
        child_pages = max(1, round(page_span * child_chars / total_chars))
        child.start_index = current_page
        child.end_index = min(current_page + child_pages - 1, parent.end_index)
        current_page = child.end_index + 1

    parent.nodes[-1].end_index = parent.end_index
