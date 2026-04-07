"""Title disambiguation for tree nodes with repetitive headings.

Problem: Documents like earnings releases have dozens of sections titled
"Reconciliation of Non-GAAP Financial Measures" or "Condensed Consolidated
Statement of Earnings." The agent can't tell them apart from the outline.

Solution: After tree building + enrichment, detect sibling nodes with
duplicate titles and rewrite them using distinguishing content from the
node's text. No LLM needed — pure heuristic extraction.

This runs after enrichment and before querying.
"""

from __future__ import annotations

import logging
import re
from collections import Counter

from nanoindex.models import DocumentTree, TreeNode
from nanoindex.utils.tree_ops import iter_nodes

logger = logging.getLogger(__name__)


def disambiguate_titles(tree: DocumentTree) -> DocumentTree:
    """Find nodes with duplicate titles among siblings and make them unique.

    Extracts distinguishing info from:
    1. Table headers (quarter labels, year labels, segment names)
    2. First meaningful line of text (subtitles)
    3. Page numbers (as last resort)

    Modifies tree in-place and returns it.
    """
    # Find ALL duplicate titles across the entire tree, not just same-parent siblings.
    # A document can have "Reconciliation of Non-GAAP" under different parents
    # but the agent sees them all in the same flat outline.
    all_nodes = list(iter_nodes(tree.structure))
    title_counts = Counter(n.title.strip().lower() for n in all_nodes if n.title)
    dup_titles = {t for t, c in title_counts.items() if c > 1}

    fixed = 0
    for node in all_nodes:
        if not node.title or node.title.strip().lower() not in dup_titles:
            continue

        # Try to find a distinguishing subtitle
        subtitle = _extract_subtitle(node)
        if subtitle and subtitle.lower() != node.title.strip().lower():
            old_title = node.title
            node.title = f"{node.title} — {subtitle}"
            fixed += 1

            # Also fix the summary if it just echoes the old title
            if node.summary and node.summary.strip().lower() == old_title.strip().lower():
                node.summary = f"{node.title}. Page {node.start_index}."

    if fixed:
        logger.info("Disambiguated %d node titles", fixed)

    # Also compute and store the outline entropy
    all_nodes = list(iter_nodes(tree.structure))
    titles = [n.title.strip().lower() for n in all_nodes if n.title]
    if titles:
        unique_ratio = len(set(titles)) / len(titles)
        tree.extraction_metadata["outline_entropy"] = round(unique_ratio, 2)
        logger.info("Outline entropy: %.0f%% unique titles", unique_ratio * 100)

    return tree


def _extract_subtitle(node: TreeNode) -> str | None:
    """Extract a distinguishing subtitle from node content.

    Tries multiple strategies in order of reliability.
    """
    title_lower = (node.title or "").strip().lower()

    # First try: child titles (most reliable — they often have quarter/segment labels)
    if node.nodes:
        child_titles = [
            c.title for c in node.nodes
            if c.title and c.title.strip().lower() != title_lower
        ]
        if child_titles:
            return "; ".join(child_titles[:3])

    # Gather text: own text, then child text, then summary
    text = node.text or ""
    if not text.strip():
        for child in (node.nodes or []):
            if (child.text or "").strip():
                text = child.text
                break
    if not text.strip():
        if node.summary and node.summary.strip().lower() != title_lower:
            text = node.summary
    if not text.strip():
        if node.start_index:
            return f"p.{node.start_index}"
        return None

    # Strategy 1: Find quarter/year labels in first 500 chars
    # Patterns like "Q1 2023", "FY2022", "Second Quarter", "Six Months", "2022 ACTUAL vs. 2021"
    first_chunk = text[:500]
    quarter_match = re.search(
        r'(Q[1-4]\s*\d{4}|(?:First|Second|Third|Fourth)\s+Quarter|'
        r'(?:Six|Three|Nine|Twelve)\s+Months|'
        r'(?:FY|Full[- ]?Year)\s*\d{4}|'
        r'\d{4}\s*(?:QTD|YTD)|'
        r'\d{4}\s+ACTUAL\s+vs\.?\s+\d{4})',
        first_chunk, re.IGNORECASE,
    )
    if quarter_match:
        return quarter_match.group(0).strip()

    # Strategy 2: Find a subtitle after the main title
    # Often the text starts with the title then has a more specific subtitle on the next line
    lines = [l.strip() for l in text[:500].split("\n") if l.strip()]

    # Skip lines that are just the title
    title_lower = (node.title or "").strip().lower()
    meaningful_lines = []
    for line in lines[:5]:
        clean = re.sub(r'<[^>]+>', '', line).strip()  # strip HTML tags
        if not clean:
            continue
        if clean.lower() == title_lower:
            continue
        if len(clean) < 5:
            continue
        meaningful_lines.append(clean)

    if meaningful_lines:
        # Take the first meaningful line that's different from title
        subtitle = meaningful_lines[0]
        # Clean it up: remove company name prefix if it matches doc
        subtitle = re.sub(r'^(?:Johnson\s*&\s*Johnson|J&J)\s+(?:and\s+)?(?:Subsidiaries\s+)?', '', subtitle, flags=re.IGNORECASE)
        # Truncate to reasonable length
        if len(subtitle) > 80:
            subtitle = subtitle[:77] + "..."
        if subtitle and subtitle.lower() != title_lower:
            return subtitle

    # Strategy 3: Look for segment/topic keywords
    segment_match = re.search(
        r'((?:Income|Revenue|Sales|Earnings|Assets|Liabilities|Equity|Cash Flow|'
        r'Operating|Investing|Financing|Segment|Geographic|Region|'
        r'Consumer|Pharmaceutical|MedTech|Industrial|Healthcare|'
        r'Innovative Medicine|Essential Health)\s*(?:by\s+\w+)?)',
        first_chunk, re.IGNORECASE,
    )
    if segment_match:
        return segment_match.group(0).strip()

    # Strategy 4: Page number
    if node.start_index:
        return f"p.{node.start_index}"

    return None
