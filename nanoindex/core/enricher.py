"""LLM-based summary generation for tree nodes.

This is the **only** step in the indexing pipeline that calls an LLM.
It is fully optional — the tree is usable without summaries.

Summaries are generated concurrently with bounded parallelism so we
don't overwhelm rate limits.
"""

from __future__ import annotations

import asyncio
import logging

from nanoindex.config import NanoIndexConfig
from nanoindex.core.llm import LLMClient
from nanoindex.models import DocumentTree, TreeNode
from nanoindex.utils.tokens import count_tokens
from nanoindex.utils.tree_ops import iter_nodes

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = """\
Summarize this section in TWO parts (max 60 words total):
1. One sentence describing the section.
2. Key data items: list the specific metrics, line items, or data points \
present (e.g. revenue, net income, total assets, capex, operating cash flow, \
gross margin, EPS, shares outstanding). If the section contains a financial \
table, list every row label visible.

Title: {title}
Content: {content}

Summary:"""

_DOC_DESCRIPTION_PROMPT = """\
You are a document indexing assistant. Given the document structure below, \
write a concise 2-3 sentence description of what this document is about.

Document: {doc_name}

Structure:
{outline}

Description:"""

_MAX_CONCURRENT = 15


async def enrich_tree(
    tree: DocumentTree,
    llm: LLMClient,
    config: NanoIndexConfig,
) -> DocumentTree:
    """Generate summaries for tree nodes and optionally a document description.

    Modifies *tree* in-place and returns it.
    """
    model = config.summary_model or llm.model

    if config.add_summaries:
        await _generate_summaries(tree.structure, llm, model, config.min_node_tokens)

    if config.add_doc_description:
        tree.doc_description = await _generate_doc_description(tree, llm, model)

    return tree


async def _generate_summaries(
    nodes: list[TreeNode],
    llm: LLMClient,
    model: str,
    min_tokens: int,
) -> None:
    """Generate summaries for all nodes concurrently."""
    all_nodes = [n for n in iter_nodes(nodes) if n.summary is None]

    sem = asyncio.Semaphore(_MAX_CONCURRENT)

    async def _summarise(node: TreeNode) -> None:
        content = node.text or ""
        if count_tokens(content) < min_tokens:
            node.summary = content[:200] if content else node.title
            return

        prompt = _SUMMARY_PROMPT.format(title=node.title, content=content[:30000])
        messages = [{"role": "user", "content": prompt}]

        async with sem:
            try:
                node.summary = await llm.chat(messages, model=model, max_tokens=1024)
            except Exception:
                logger.warning("Summary generation failed for node '%s'", node.title, exc_info=True)
                node.summary = node.title

    await asyncio.gather(*[_summarise(n) for n in all_nodes])


async def _generate_doc_description(
    tree: DocumentTree,
    llm: LLMClient,
    model: str,
) -> str:
    """Generate a top-level document description from the tree outline."""
    from nanoindex.utils.tree_ops import tree_to_outline

    outline = tree_to_outline(tree.structure)
    prompt = _DOC_DESCRIPTION_PROMPT.format(doc_name=tree.doc_name, outline=outline[:20000])
    messages = [{"role": "user", "content": prompt}]

    try:
        return await llm.chat(messages, model=model, max_tokens=512)
    except Exception:
        logger.warning("Doc description generation failed", exc_info=True)
        return ""
