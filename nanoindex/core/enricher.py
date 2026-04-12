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

_SUMMARY_PROMPT_FINANCE = """\
Summarize this section in TWO parts (max 60 words total):
1. One sentence describing the section.
2. Key data items: list the specific metrics, line items, or data points \
present (e.g. revenue, net income, total assets, capex, operating cash flow, \
gross margin, EPS, shares outstanding). If the section contains a financial \
table, list every row label visible.

Title: {title}
Content: {content}

Summary:"""

_SUMMARY_PROMPT_GENERAL = """\
Summarize this section in TWO parts (max 60 words total):
1. One sentence describing what this section covers.
2. Key items: list the specific topics, entities, data points, or findings \
present. If the section contains a table, list every column header visible.

Title: {title}
Content: {content}

Summary:"""

_FINANCE_DOMAINS = {"sec_10k", "sec_10q", "financial", "earnings", "insurance"}

_DOC_DESCRIPTION_PROMPT = """\
You are a document indexing assistant. Given the document structure below, \
write a concise 2-3 sentence description of what this document is about.

Document: {doc_name}

Structure:
{outline}

Description:"""

_MAX_CONCURRENT = 30
_MAX_RETRIES = 3
_RETRY_DELAY = 2  # seconds, doubles on each retry


async def enrich_tree(
    tree: DocumentTree,
    llm: LLMClient,
    config: NanoIndexConfig,
) -> DocumentTree:
    """Generate summaries for tree nodes and optionally a document description.

    Modifies *tree* in-place and returns it.
    """
    model = config.summary_model or llm.model
    domain = getattr(tree, "domain", "") or ""
    summary_prompt = _SUMMARY_PROMPT_FINANCE if domain in _FINANCE_DOMAINS else _SUMMARY_PROMPT_GENERAL

    if config.add_summaries:
        await _generate_summaries(tree.structure, llm, model, config.min_node_tokens, summary_prompt)

    if config.add_doc_description:
        tree.doc_description = await _generate_doc_description(tree, llm, model)

    return tree


async def _generate_summaries(
    nodes: list[TreeNode],
    llm: LLMClient,
    model: str,
    min_tokens: int,
    summary_prompt: str = _SUMMARY_PROMPT_GENERAL,
) -> None:
    """Generate summaries for all nodes concurrently."""
    # Process bottom-up: leaves first so parent summaries can use children's summaries
    all_nodes_ordered = list(iter_nodes(nodes))
    # Separate leaves (no children or all children already summarised) from parents
    leaves = [n for n in all_nodes_ordered if n.summary is None and not n.nodes]
    parents = [n for n in all_nodes_ordered if n.summary is None and n.nodes]
    # Reverse parents so deepest parents go first
    parents.reverse()
    all_nodes = leaves  # will process parents after leaves complete

    sem = asyncio.Semaphore(_MAX_CONCURRENT)

    async def _summarise(node: TreeNode) -> None:
        content = node.text or ""
        if count_tokens(content) < min_tokens:
            # For parent nodes, synthesise content from children's summaries
            if node.nodes:
                child_summaries = [
                    f"- {c.title}: {c.summary}"
                    for c in node.nodes if c.summary and c.summary != c.title
                ]
                if child_summaries:
                    content = f"Section: {node.title}\nContains:\n" + "\n".join(child_summaries[:15])
                else:
                    node.summary = content[:200] if content else node.title
                    return
            else:
                node.summary = content[:200] if content else node.title
                return

        prompt = summary_prompt.format(title=node.title, content=content[:30000])
        messages = [{"role": "user", "content": prompt}]

        async with sem:
            for attempt in range(_MAX_RETRIES):
                try:
                    node.summary = await llm.chat(messages, model=model, max_tokens=1024)
                    return
                except Exception as exc:
                    exc_str = str(exc).lower()
                    if "rate" in exc_str or "429" in exc_str or "quota" in exc_str or "overloaded" in exc_str:
                        delay = _RETRY_DELAY * (2 ** attempt)
                        logger.info("Rate limited on '%s', retrying in %ds (attempt %d/%d)",
                                    node.title[:30], delay, attempt + 1, _MAX_RETRIES)
                        await asyncio.sleep(delay)
                    else:
                        logger.warning("Summary failed for '%s': %s", node.title[:30], exc)
                        node.summary = node.title
                        return
            logger.warning("Summary failed after %d retries for '%s'", _MAX_RETRIES, node.title[:30])
            node.summary = node.title

    # Phase 1: summarise leaves concurrently
    if all_nodes:
        await asyncio.gather(*[_summarise(n) for n in all_nodes])

    # Phase 2: summarise parents (bottom-up, children already done)
    if parents:
        await asyncio.gather(*[_summarise(n) for n in parents])


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
