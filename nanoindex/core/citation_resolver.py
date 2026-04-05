"""Resolve citations to exact bounding boxes by matching answer text to bbox content.

After the LLM generates an answer, this module narrows the bounding boxes
in each citation to only those that actually contain the cited information.
"""

from __future__ import annotations

import logging
import re
from nanoindex.models import Answer, BoundingBox, DocumentTree
from nanoindex.utils.tree_ops import find_node

logger = logging.getLogger(__name__)


def resolve_citations(answer: Answer, tree: DocumentTree) -> Answer:
    """Narrow citation bounding boxes to only those matching the answer content.

    For each citation, finds which bboxes contain text that appears in the answer.
    Modifies the answer in place and returns it.
    """
    if not answer.citations:
        return answer

    # Extract key phrases from the answer for matching
    key_phrases = _extract_key_phrases(answer.content)
    if not key_phrases:
        return answer

    for citation in answer.citations:
        if not citation.bounding_boxes:
            # Try to populate from the tree node
            node = find_node(tree.structure, citation.node_id)
            if node and node.bounding_boxes:
                citation.bounding_boxes = list(node.bounding_boxes)

            # Also try tree-level bboxes for the cited pages
            if not citation.bounding_boxes and tree.all_bounding_boxes:
                page_set = set(citation.pages)
                citation.bounding_boxes = [
                    bb for bb in tree.all_bounding_boxes if bb.page in page_set
                ]

        if not citation.bounding_boxes:
            continue

        # Score each bbox against the key phrases
        matched = _match_bboxes(citation.bounding_boxes, key_phrases)

        if matched:
            citation.bounding_boxes = matched
            logger.debug(
                "Citation %s: narrowed %d -> %d bboxes",
                citation.node_id,
                len(citation.bounding_boxes),
                len(matched),
            )

    return answer


def _extract_key_phrases(text: str) -> list[str]:
    """Extract key phrases from answer text for bbox matching.

    Pulls out: numbers, dollar amounts, percentages, proper nouns,
    and short quoted phrases.
    """
    phrases = []

    # Dollar amounts: $127.4, $1,000,000, $52.3 million
    for m in re.finditer(r"\$[\d,]+\.?\d*\s*(?:million|billion|M|B|K)?", text):
        # Normalize: remove spaces, lowercase
        raw = m.group().replace(",", "")
        phrases.append(raw)
        # Also add just the number part
        num = re.search(r"[\d.]+", raw)
        if num:
            phrases.append(num.group())

    # Percentages: 74.2%, 23%
    for m in re.finditer(r"[\d.]+%", text):
        phrases.append(m.group())
        # Also without %
        phrases.append(m.group().rstrip("%"))

    # Large numbers without $: 127.4, 14.1, 512
    for m in re.finditer(r"\b\d+\.?\d+\b", text):
        val = m.group()
        if len(val) >= 3 and val not in phrases:
            phrases.append(val)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for p in phrases:
        lower = p.lower().strip()
        if lower and lower not in seen and len(lower) >= 2:
            seen.add(lower)
            unique.append(lower)

    return unique


def _match_bboxes(
    bboxes: list[BoundingBox],
    key_phrases: list[str],
) -> list[BoundingBox]:
    """Score bboxes against key phrases and return those with matches."""
    scored: list[tuple[int, BoundingBox]] = []

    for bbox in bboxes:
        if not bbox.text:
            continue
        text_lower = bbox.text.lower()
        score = sum(1 for phrase in key_phrases if phrase in text_lower)
        if score > 0:
            scored.append((score, bbox))

    if not scored:
        return []

    # Sort by score descending, return top matches
    scored.sort(key=lambda x: -x[0])
    return [bb for _, bb in scored]
