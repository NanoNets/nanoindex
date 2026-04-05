"""Answer generation: text-based and vision-based.

Takes retrieved nodes (with their text and optionally page images) and
produces a structured ``Answer`` with citations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from nanoindex.core.llm import LLMClient
from nanoindex.exceptions import GenerationError
from nanoindex.models import Answer, BoundingBox, Citation, DocumentTree, PageDimensions, RetrievedNode
from nanoindex.utils.pdf import render_pages
from nanoindex.utils.tokens import count_tokens

logger = logging.getLogger(__name__)

_MAX_CONTEXT_TOKENS = 150_000

_TEXT_USER = """\
Answer the question based on the context. \
Show your calculations and reasoning step by step when the question \
involves numbers, ratios, or comparisons.

Question: {query}

Context:
{context}

Provide a clear answer based only on the context provided."""

_VISION_USER = """\
Answer the question based on the images of the document pages and the text \
context below. Show your calculations and reasoning step by step when the \
question involves numbers, ratios, or comparisons.

Question: {query}

Text context:
{context}

Provide a clear answer based only on the context provided."""


def _build_text_context(nodes: list[RetrievedNode], max_tokens: int = _MAX_CONTEXT_TOKENS) -> str:
    """Concatenate retrieved node texts, truncating to fit *max_tokens*."""
    parts: list[str] = []
    total_tokens = 0

    for rn in nodes:
        doc_tag = f"[{rn.doc_name}] " if rn.doc_name else ""
        header = f"--- {doc_tag}Section: {rn.node.title} [{rn.node.node_id}]"
        if rn.node.start_index:
            header += f" (pp. {rn.node.start_index}-{rn.node.end_index})"
        header += " ---"
        text = rn.text or "(no text available)"

        block = f"{header}\n\n{text}"
        block_tokens = count_tokens(block)

        if total_tokens + block_tokens > max_tokens:
            remaining = max_tokens - total_tokens
            if remaining > 200:
                chars_budget = remaining * 4
                parts.append(f"{header}\n\n{text[:chars_budget]}…")
            break
        parts.append(block)
        total_tokens += block_tokens

    return "\n\n".join(parts)


def _build_citations(
    nodes: list[RetrievedNode],
    tree: DocumentTree | None = None,
    include_metadata: bool = False,
) -> list[Citation]:
    citations: list[Citation] = []
    for rn in nodes:
        pages = (
            list(range(rn.node.start_index, rn.node.end_index + 1))
            if rn.node.start_index else []
        )

        # Always propagate bounding boxes from the node itself
        bboxes: list[BoundingBox] = list(rn.node.bounding_boxes)
        dims: list[PageDimensions] = []

        if include_metadata and pages:
            page_set = set(pages)
            if tree:
                # Enrich with all bboxes for cited pages from the tree
                tree_bboxes = [bb for bb in tree.all_bounding_boxes if bb.page in page_set]
                if tree_bboxes:
                    bboxes = tree_bboxes
                dims = [pd for pd in tree.page_dimensions if pd.page in page_set]

        citations.append(Citation(
            node_id=rn.node.node_id,
            title=rn.node.title,
            doc_name=rn.doc_name,
            pages=pages,
            bounding_boxes=bboxes,
            page_dimensions=dims,
        ))
    return citations


# ------------------------------------------------------------------
# Text-based generation
# ------------------------------------------------------------------

async def generate_text_answer(
    query: str,
    nodes: list[RetrievedNode],
    llm: LLMClient,
    *,
    tree: DocumentTree | None = None,
    include_metadata: bool = False,
) -> Answer:
    """Generate an answer using node text as context."""
    if not nodes:
        return Answer(content="No relevant sections were found in the document.", mode="text")

    context = _build_text_context(nodes)
    prompt = _TEXT_USER.format(query=query, context=context)
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": prompt},
    ]

    try:
        content = await llm.chat(messages, max_tokens=2048)
    except Exception as exc:
        raise GenerationError(f"Text answer generation failed: {exc}") from exc

    if not content:
        content = "Unable to generate an answer from the provided context."

    return Answer(
        content=content,
        citations=_build_citations(nodes, tree, include_metadata),
        mode="text",
    )


# ------------------------------------------------------------------
# Vision-based generation
# ------------------------------------------------------------------

async def generate_vision_answer(
    query: str,
    nodes: list[RetrievedNode],
    llm: LLMClient,
    pdf_path: str | Path,
    *,
    tree: DocumentTree | None = None,
    include_metadata: bool = False,
) -> Answer:
    """Generate an answer using page images sent to a VLM."""
    if not nodes:
        return Answer(content="No relevant sections were found in the document.", mode="vision")

    page_numbers: list[int] = []
    for rn in nodes:
        if rn.node.start_index:
            for p in range(rn.node.start_index, rn.node.end_index + 1):
                if p not in page_numbers:
                    page_numbers.append(p)

    page_numbers = sorted(page_numbers)[:10]

    image_uris = render_pages(pdf_path, page_numbers)

    user_content: list[dict[str, Any]] = []

    for i, uri in enumerate(image_uris):
        user_content.append({
            "type": "image_url",
            "image_url": {"url": uri},
        })

    context_text = _build_text_context(nodes)
    prompt = _VISION_USER.format(query=query, context=context_text[:40000])
    user_content.append({"type": "text", "text": prompt})

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": user_content},
    ]

    try:
        content = await llm.chat(messages, max_tokens=2048)
    except Exception as exc:
        raise GenerationError(f"Vision answer generation failed: {exc}") from exc

    return Answer(
        content=content,
        citations=_build_citations(nodes, tree, include_metadata),
        mode="vision",
    )


# ------------------------------------------------------------------
# Unified entry point
# ------------------------------------------------------------------

async def generate_answer(
    query: str,
    nodes: list[RetrievedNode],
    llm: LLMClient,
    *,
    mode: str = "text",
    pdf_path: str | Path | None = None,
    tree: DocumentTree | None = None,
    include_metadata: bool = False,
) -> Answer:
    """Generate an answer in the requested mode.

    When *include_metadata* is ``True``, each citation in the returned
    ``Answer`` carries the bounding boxes and page dimensions for its
    source pages — enabling the caller to highlight exact regions.
    """
    if mode == "vision":
        if pdf_path is None:
            raise GenerationError("pdf_path is required for vision mode")
        return await generate_vision_answer(
            query, nodes, llm, pdf_path,
            tree=tree, include_metadata=include_metadata,
        )
    return await generate_text_answer(
        query, nodes, llm,
        tree=tree, include_metadata=include_metadata,
    )
