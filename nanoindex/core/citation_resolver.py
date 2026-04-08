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

    For each citation, scores every bbox against key phrases extracted from
    the answer (numbers, dollar amounts, percentages, entity names, key terms).
    Keeps only bboxes that contain matching text.
    """
    if not answer.citations:
        return answer

    key_phrases = _extract_key_phrases(answer.content)
    if not key_phrases:
        return answer

    for citation in answer.citations:
        if not citation.bounding_boxes:
            # Populate from the tree node
            node = find_node(tree.structure, citation.node_id)
            if node and node.bounding_boxes:
                citation.bounding_boxes = list(node.bounding_boxes)

            # Fallback: tree-level bboxes for cited pages
            if not citation.bounding_boxes and tree.all_bounding_boxes:
                page_set = set(citation.pages)
                citation.bounding_boxes = [
                    bb for bb in tree.all_bounding_boxes if bb.page in page_set
                ]

        if not citation.bounding_boxes:
            continue

        # Score and narrow
        original_count = len(citation.bounding_boxes)
        matched = _match_bboxes(citation.bounding_boxes, key_phrases)

        if matched:
            citation.bounding_boxes = matched
            if len(matched) < original_count:
                logger.debug(
                    "Citation %s: narrowed %d -> %d bboxes",
                    citation.node_id, original_count, len(matched),
                )

    return answer


def _extract_key_phrases(text: str) -> list[str]:
    """Extract key phrases from answer text for bbox matching.

    Pulls out:
    1. Dollar amounts ($127.4 million, $1,577)
    2. Percentages (74.2%, 23%)
    3. Numbers with 3+ digits (1577, 32765)
    4. Capitalized entity names (3M Company, Tim Cook, Safety and Industrial)
    5. Key financial/legal terms from the answer
    """
    phrases = []

    # 1. Dollar amounts: $127.4, $1,000,000, $52.3 million
    for m in re.finditer(r"\$[\d,]+\.?\d*\s*(?:million|billion|M|B|K)?", text):
        raw = m.group().replace(",", "")
        phrases.append(raw)
        num = re.search(r"[\d.]+", raw)
        if num:
            phrases.append(num.group())

    # 2. Percentages: 74.2%, 23%
    for m in re.finditer(r"[\d.]+%", text):
        phrases.append(m.group())
        phrases.append(m.group().rstrip("%"))

    # 3. Numbers with 3+ digits (not already captured)
    for m in re.finditer(r"\b\d[\d,]*\.?\d*\b", text):
        val = m.group().replace(",", "")
        if len(val) >= 3 and val not in phrases:
            phrases.append(val)

    # 4. Capitalized multi-word names (entity names, section titles)
    # "3M Company", "Safety and Industrial", "Tim Cook", "Net Income"
    for m in re.finditer(
        r"\b([A-Z][a-zA-Z]*(?:\s+(?:and|of|the|for|in|&)\s+[A-Z][a-zA-Z]*|"
        r"\s+[A-Z][a-zA-Z]*){1,4})\b",
        text,
    ):
        name = m.group().strip()
        if len(name) >= 4:
            phrases.append(name.lower())

    # 5. Single capitalized words that are likely entities (not sentence starters)
    # Only grab if preceded by non-sentence-start context
    for m in re.finditer(r"(?<=[,;:]\s)([A-Z][a-zA-Z]{3,})\b", text):
        phrases.append(m.group().lower())

    # 6. Key financial terms that appear in the answer
    _TERMS = [
        "net sales", "net income", "revenue", "operating income", "total assets",
        "total liabilities", "cash flow", "capital expenditure", "depreciation",
        "amortization", "dividends", "earnings per share", "gross profit",
        "operating expenses", "cost of sales", "working capital", "accounts payable",
        "accounts receivable", "inventory", "shareholders equity",
        "provision for income taxes", "income before taxes",
    ]
    text_lower = text.lower()
    for term in _TERMS:
        if term in text_lower:
            phrases.append(term)

    # Deduplicate preserving order
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
    """Score bboxes against key phrases and return those with matches.

    Each bbox is scored by how many key phrases its text contains.
    Only bboxes with at least one match are returned, sorted by score.
    """
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

    scored.sort(key=lambda x: -x[0])
    return [bb for _, bb in scored]
