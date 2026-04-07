"""Build a document tree from plain text (no PDF/OCR needed).

For legal contracts, NDAs, M&A agreements, and other text files that have
structure embedded in formatting: section numbers, ALL CAPS headings,
ARTICLE/SECTION markers, lettered clauses.

Returns a DocumentTree with character-level span tracking on each node,
enabling character-level precision/recall evaluation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from nanoindex.models import DocumentTree, TreeNode

logger = logging.getLogger(__name__)


@dataclass
class TextSpan:
    """A span in the source text with start/end character offsets."""
    start: int
    end: int
    text: str
    heading: str = ""
    depth: int = 0


# Patterns for detecting section structure in legal documents
_SECTION_PATTERNS = [
    # ARTICLE I, ARTICLE II, etc.
    (re.compile(r'^(ARTICLE\s+[IVXLCDM]+\.?\s*.*)$', re.MULTILINE), 0),
    # SECTION 1.1, Section 2.3, etc.
    (re.compile(r'^((?:SECTION|Section)\s+\d+(?:\.\d+)*\.?\s*.*)$', re.MULTILINE), 1),
    # Numbered sections: 1. , 2. , 10. (at start of line)
    (re.compile(r'^(\d{1,2}\.\s+[A-Z].{0,80})$', re.MULTILINE), 1),
    # Lettered subsections: (a), (b), (i), (ii)
    (re.compile(r'^(\([a-z]\)\s+.{0,80})$', re.MULTILINE), 2),
    (re.compile(r'^(\([ivx]+\)\s+.{0,80})$', re.MULTILINE), 3),
    # ALL CAPS headings (at least 3 words, all caps)
    (re.compile(r'^([A-Z][A-Z\s,]{10,80})$', re.MULTILINE), 0),
    # Numbered subsections: 1.1, 1.2, 2.1 etc. with text
    (re.compile(r'^(\d+\.\d+\.?\s+.{0,80})$', re.MULTILINE), 2),
]

# Fallback: split by double newlines (paragraphs)
_PARAGRAPH_SPLIT = re.compile(r'\n\s*\n')


def build_text_tree(
    text: str,
    doc_name: str = "document",
    *,
    max_node_chars: int = 4000,
) -> DocumentTree:
    """Build a tree from plain text, preserving character offsets.

    Each node stores its character span in extraction_metadata["char_span"].

    Strategy:
    1. Try to find section headings (ARTICLE, SECTION, numbered)
    2. If enough structure found, build hierarchical tree
    3. If not, fall back to paragraph-level splitting
    """
    sections = _find_sections(text)

    if len(sections) < 3:
        # Not enough structure — fall back to paragraph splitting
        sections = _split_paragraphs(text, max_chars=max_node_chars)
        logger.info("Text tree (paragraph split): %d nodes from %d chars", len(sections), len(text))
    else:
        logger.info("Text tree (section split): %d sections from %d chars", len(sections), len(text))

    # Build tree nodes
    nodes = []
    for i, span in enumerate(sections):
        node_id = f"{i:04d}"
        node = TreeNode(
            node_id=node_id,
            title=span.heading[:100] if span.heading else f"Section {i + 1}",
            text=span.text,
            summary=span.heading[:200] if span.heading else span.text[:200],
        )
        # Store character span for evaluation
        node._char_span = (span.start, span.end)  # type: ignore[attr-defined]
        nodes.append(node)

    tree = DocumentTree(
        doc_name=doc_name,
        structure=nodes,
        extraction_metadata={
            "source": "text_tree_builder",
            "total_chars": len(text),
            "node_count": len(nodes),
        },
    )
    return tree


def _find_sections(text: str) -> list[TextSpan]:
    """Find section boundaries using heading patterns."""
    # Collect all heading matches with positions
    headings: list[tuple[int, int, str, int]] = []  # (start, end, heading_text, depth)

    for pattern, depth in _SECTION_PATTERNS:
        for m in pattern.finditer(text):
            heading_text = m.group(1).strip()
            # Skip very short matches or likely false positives
            if len(heading_text) < 3:
                continue
            headings.append((m.start(), m.end(), heading_text, depth))

    if not headings:
        return []

    # Sort by position and deduplicate overlapping headings
    headings.sort(key=lambda x: x[0])
    deduped: list[tuple[int, int, str, int]] = []
    for h in headings:
        if deduped and h[0] < deduped[-1][1]:
            # Overlapping — keep the one with lower depth (more structural)
            if h[3] < deduped[-1][3]:
                deduped[-1] = h
            continue
        deduped.append(h)

    # Build spans between headings
    sections: list[TextSpan] = []

    # Text before first heading (preamble)
    if deduped[0][0] > 0:
        preamble_text = text[:deduped[0][0]].strip()
        if len(preamble_text) > 50:
            sections.append(TextSpan(
                start=0, end=deduped[0][0],
                text=preamble_text,
                heading="Preamble",
                depth=0,
            ))

    # Each heading → next heading
    for i, (start, end, heading, depth) in enumerate(deduped):
        if i + 1 < len(deduped):
            section_end = deduped[i + 1][0]
        else:
            section_end = len(text)

        section_text = text[start:section_end].strip()
        if section_text:
            sections.append(TextSpan(
                start=start, end=section_end,
                text=section_text,
                heading=heading,
                depth=depth,
            ))

    return sections


def _split_paragraphs(text: str, max_chars: int = 4000) -> list[TextSpan]:
    """Fall back to paragraph-level splitting with character tracking."""
    parts = _PARAGRAPH_SPLIT.split(text)
    spans: list[TextSpan] = []
    pos = 0

    for part in parts:
        part_stripped = part.strip()
        if not part_stripped:
            pos = text.find(part, pos) + len(part) if part else pos
            continue

        actual_start = text.find(part_stripped, pos)
        if actual_start < 0:
            actual_start = pos
        actual_end = actual_start + len(part_stripped)

        # If paragraph is too long, split further
        if len(part_stripped) > max_chars:
            for chunk_start in range(0, len(part_stripped), max_chars):
                chunk = part_stripped[chunk_start:chunk_start + max_chars]
                if chunk.strip():
                    cs = actual_start + chunk_start
                    ce = cs + len(chunk)
                    first_line = chunk.split('\n')[0][:80].strip()
                    spans.append(TextSpan(
                        start=cs, end=ce,
                        text=chunk,
                        heading=first_line,
                        depth=0,
                    ))
        else:
            first_line = part_stripped.split('\n')[0][:80].strip()
            spans.append(TextSpan(
                start=actual_start, end=actual_end,
                text=part_stripped,
                heading=first_line,
                depth=0,
            ))

        pos = actual_end

    return spans
