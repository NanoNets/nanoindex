"""Context extractor for multimodal items.

Extracts surrounding text for each non-text element so the LLM/VLM has
context when analyzing it.
"""

from __future__ import annotations

import logging

from nanoindex.models import ModalContent, ParsedDocument
from nanoindex.utils.tokens import count_tokens

logger = logging.getLogger(__name__)


def extract_context(
    item: ModalContent,
    parsed: ParsedDocument,
    max_tokens: int = 500,
) -> str:
    """Extract surrounding text context for a multimodal item.

    Uses the item's page number to find nearby text from the parsed document.
    Returns text from the same page plus adjacent pages, truncated to max_tokens.
    """
    if not parsed.pages or item.page < 1:
        # Fallback: search markdown for nearby content
        return ""

    page_idx = item.page - 1  # 0-based

    context_parts: list[str] = []

    # Previous page
    if page_idx > 0:
        context_parts.append(parsed.pages[page_idx - 1][-500:])

    # Same page
    if page_idx < len(parsed.pages):
        context_parts.append(parsed.pages[page_idx])

    # Next page
    if page_idx + 1 < len(parsed.pages):
        context_parts.append(parsed.pages[page_idx + 1][:500])

    context = "\n\n".join(context_parts)

    # Truncate to token limit
    while count_tokens(context) > max_tokens and len(context) > 100:
        context = context[: int(len(context) * 0.8)]

    return context.strip()


def enrich_modal_contexts(parsed: ParsedDocument, max_tokens: int = 500) -> None:
    """Fill in surrounding_text for all modal contents in-place."""
    for item in parsed.modal_contents:
        if not item.surrounding_text:
            item.surrounding_text = extract_context(item, parsed, max_tokens)
