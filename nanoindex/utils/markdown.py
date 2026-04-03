"""Markdown heading parser and content extractor.

Parses structured markdown (as returned by Nanonets OCR) into a flat list
of heading nodes with their text content, ready for tree building.

Also understands ``<!-- nanoindex:page:N -->`` markers injected by the
page-parallel extractor so that each heading carries a source page number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_PAGE_MARKER_RE = re.compile(r"<!--\s*nanoindex:page:(\d+)\s*-->")
_PAGE_HEADING_RE = re.compile(r"^#{1,3}\s+Page\s+(\d+)\s*$", re.IGNORECASE)


@dataclass
class HeadingNode:
    """A heading extracted from markdown, with its body text."""

    title: str
    level: int
    line_number: int
    page: int = 0
    text_content: str = ""
    children: list["HeadingNode"] = field(default_factory=list)


def parse_markdown_headings(markdown: str) -> list[HeadingNode]:
    """Extract headings from *markdown* and assign text content to each.

    Returns a **flat** list ordered by document position.  The caller
    (tree_builder) is responsible for nesting by level.

    Page markers (``<!-- nanoindex:page:N -->``) are tracked so each
    heading gets a ``page`` attribute indicating which PDF page it
    originated from.
    """
    lines = markdown.split("\n")
    headings: list[HeadingNode] = []
    heading_positions: list[int] = []

    current_page = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        pm = _PAGE_MARKER_RE.search(stripped)
        if pm:
            current_page = int(pm.group(1))
            continue

        # Treat "## Page N" as a page marker, not a real heading
        ph = _PAGE_HEADING_RE.match(stripped)
        if ph:
            current_page = int(ph.group(1))
            continue

        m = _HEADING_RE.match(stripped)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            headings.append(HeadingNode(
                title=title, level=level, line_number=i, page=current_page,
            ))
            heading_positions.append(i)

    for idx, node in enumerate(headings):
        start = node.line_number + 1
        end = heading_positions[idx + 1] if idx + 1 < len(heading_positions) else len(lines)
        body_lines = [
            ln for ln in lines[start:end]
            if not _PAGE_MARKER_RE.search(ln)
        ]
        node.text_content = "\n".join(body_lines).strip()

    return headings


def extract_text_between(markdown: str, start_heading: str, end_heading: str | None) -> str:
    """Return the markdown body between two heading titles."""
    lines = markdown.split("\n")
    capturing = False
    captured: list[str] = []

    for line in lines:
        m = _HEADING_RE.match(line.strip())
        if m:
            title = m.group(2).strip()
            if not capturing and title == start_heading:
                capturing = True
                continue
            if capturing and end_heading and title == end_heading:
                break
        if capturing:
            captured.append(line)

    return "\n".join(captured).strip()
