"""Build a document tree from Nanonets extraction output — zero LLM calls.

Fallback cascade (first match wins):

  1. **TOC** — ``table-of-contents`` from Nanonets gives a rich, multi-level
     hierarchy with page numbers and parent IDs.  Best quality.
  2. **Hierarchy JSON** — ``sections`` with explicit levels from Nanonets.
  3. **Markdown headings** — regex-parsed ``#``/``##``/``###`` hierarchy.
  4. **Page-based** — one node per page (for tables-heavy documents with
     sparse headings).
  5. **Single root** — fallback for 1-page documents with no structure.
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher

from nanoindex.config import NanoIndexConfig
from nanoindex.models import (
    BoundingBox,
    DocumentTree,
    ExtractionResult,
    HierarchySection,
    HierarchyTable,
    TOCEntry,
    TreeNode,
)
from nanoindex.utils.markdown import HeadingNode, parse_markdown_headings
from nanoindex.utils.tokens import count_tokens
from nanoindex.utils.tree_ops import assign_node_ids

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Step 1: Convert hierarchy sections → flat TreeNode list
# ------------------------------------------------------------------

def _hierarchy_to_nodes(sections: list[HierarchySection], depth: int = 1) -> list[TreeNode]:
    """Recursively convert ``HierarchySection`` objects into ``TreeNode`` objects."""
    nodes: list[TreeNode] = []
    for sec in sections:
        node = TreeNode(
            title=sec.title or f"Section (level {depth})",
            level=sec.level or depth,
            text=sec.content or None,
        )
        if sec.subsections:
            node.nodes = _hierarchy_to_nodes(sec.subsections, depth=depth + 1)
        nodes.append(node)
    return nodes


# ------------------------------------------------------------------
# Step 2: Convert markdown headings → flat TreeNode list
# ------------------------------------------------------------------

def _headings_to_flat_nodes(headings: list[HeadingNode]) -> list[TreeNode]:
    """Convert parsed markdown headings into a flat ``TreeNode`` list.

    If headings carry ``page`` info (from page-parallel extraction markers),
    the page number is propagated to ``start_index`` / ``end_index``.
    """
    nodes: list[TreeNode] = []
    for i, h in enumerate(headings):
        end_page = h.page
        for j in range(i + 1, len(headings)):
            if headings[j].page and headings[j].page > h.page:
                end_page = headings[j].page - 1
                break
            if headings[j].page:
                end_page = headings[j].page
        if end_page < h.page:
            end_page = h.page

        nodes.append(TreeNode(
            title=h.title,
            level=h.level,
            text=h.text_content or None,
            start_index=h.page,
            end_index=end_page,
        ))
    return nodes


def _nest_flat_nodes(flat: list[TreeNode]) -> list[TreeNode]:
    """Convert a flat list (ordered by document position) into a nested tree
    using heading levels.  Lower level numbers are parents of higher ones."""
    if not flat:
        return []

    root: list[TreeNode] = []
    stack: list[TreeNode] = []

    for node in flat:
        while stack and stack[-1].level >= node.level:
            stack.pop()
        if stack:
            stack[-1].nodes.append(node)
        else:
            root.append(node)
        stack.append(node)

    return root


# ------------------------------------------------------------------
# Step 3: Merge hierarchy + headings
# ------------------------------------------------------------------

def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _enrich_from_headings(
    hierarchy_nodes: list[TreeNode],
    heading_nodes: list[HeadingNode],
) -> None:
    """If a hierarchy node's text is empty, try to fill it from a matching heading."""
    heading_map: dict[str, HeadingNode] = {}
    for h in heading_nodes:
        heading_map[h.title.lower().strip()] = h

    for node in _iter_all(hierarchy_nodes):
        if node.text:
            continue
        key = node.title.lower().strip()
        match = heading_map.get(key)
        if match and match.text_content:
            node.text = match.text_content
            continue
        # Fuzzy fallback
        for hkey, hnode in heading_map.items():
            if _similarity(key, hkey) > 0.85 and hnode.text_content:
                node.text = hnode.text_content
                break


def _iter_all(nodes: list[TreeNode]):
    for n in nodes:
        yield n
        yield from _iter_all(n.nodes)


# ------------------------------------------------------------------
# Step 4: Attach bounding boxes and assign pages
# ------------------------------------------------------------------

def _attach_bboxes(nodes: list[TreeNode], bboxes: list[BoundingBox]) -> None:
    """Attach bounding boxes to tree nodes by matching content text."""
    if not bboxes:
        return

    for node in _iter_all(nodes):
        title_lower = node.title.lower().strip()
        for bb in bboxes:
            if not bb.text:
                continue
            bb_text = bb.text.lower().strip().lstrip("#").strip()
            if bb_text == title_lower or _similarity(bb_text, title_lower) > 0.8:
                node.bounding_boxes.append(bb)
                break


def _assign_pages(nodes: list[TreeNode], page_count: int) -> None:
    """Assign start_index / end_index to every node.

    Uses three signals, in priority order:

    1. **Pre-set values** — ``_headings_to_flat_nodes`` may have already
       assigned page numbers from ``<!-- nanoindex:page:N -->`` markers.
    2. **Bounding boxes** — if a node has matched bboxes, use their page
       numbers.
    3. **Propagation** — parents span the union of their children;
       remaining gaps are filled sequentially.
    """
    for node in _iter_all(nodes):
        if node.start_index > 0:
            continue
        if node.bounding_boxes:
            pages = [bb.page for bb in node.bounding_boxes]
            node.start_index = min(pages)
            node.end_index = max(pages)

    _propagate_pages_up(nodes, page_count)


def _propagate_pages_up(nodes: list[TreeNode], page_count: int) -> None:
    for node in nodes:
        if node.nodes:
            _propagate_pages_up(node.nodes, page_count)
            child_starts = [c.start_index for c in node.nodes if c.start_index > 0]
            child_ends = [c.end_index for c in node.nodes if c.end_index > 0]
            if child_starts and (node.start_index == 0):
                node.start_index = min(child_starts)
            if child_ends and (node.end_index == 0):
                node.end_index = max(child_ends)
            if child_ends and node.end_index < max(child_ends):
                node.end_index = max(child_ends)

    prev_end = 1
    for node in _iter_all(nodes):
        if node.start_index == 0:
            node.start_index = prev_end
        if node.end_index == 0:
            node.end_index = node.start_index
        prev_end = node.end_index


# ------------------------------------------------------------------
# Step 5: Confidence filtering
# ------------------------------------------------------------------

def _filter_low_confidence(nodes: list[TreeNode], threshold: float) -> list[TreeNode]:
    """Remove tree nodes whose average bbox confidence is below *threshold*."""
    kept: list[TreeNode] = []
    for node in nodes:
        node.nodes = _filter_low_confidence(node.nodes, threshold)
        if node.bounding_boxes:
            avg = sum(bb.confidence for bb in node.bounding_boxes) / len(node.bounding_boxes)
            node.confidence = round(avg, 4)
        if node.confidence >= threshold or node.nodes:
            kept.append(node)
    return kept


# ------------------------------------------------------------------
# Step 6: Split oversized nodes
# ------------------------------------------------------------------

def _split_large_nodes(nodes: list[TreeNode], max_tokens: int) -> list[TreeNode]:
    """Recursively split nodes whose text exceeds *max_tokens*."""
    result: list[TreeNode] = []
    for node in nodes:
        node.nodes = _split_large_nodes(node.nodes, max_tokens)
        if node.text and count_tokens(node.text) > max_tokens:
            chunks = _chunk_text(node.text, max_tokens)
            for i, chunk in enumerate(chunks):
                child = TreeNode(
                    title=f"{node.title} (part {i + 1})",
                    level=node.level + 1,
                    text=chunk,
                    start_index=node.start_index,
                    end_index=node.end_index,
                )
                node.nodes.append(child)
            node.text = None
        result.append(node)
    return result


def _chunk_text(text: str, max_tokens: int) -> list[str]:
    """Split *text* into chunks of roughly *max_tokens* each, on paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)
        if current_tokens + para_tokens > max_tokens and current:
            chunks.append("\n\n".join(current))
            current = []
            current_tokens = 0
        current.append(para)
        current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))
    return chunks if chunks else [text]


# ------------------------------------------------------------------
# Step 7: Attach tables
# ------------------------------------------------------------------

def _attach_tables(nodes: list[TreeNode], tables: list[HierarchyTable]) -> None:
    """Attach hierarchy tables to the most relevant tree node."""
    if not tables or not nodes:
        return
    node_list = list(_iter_all(nodes))
    for table in tables:
        best_node = node_list[0]
        best_score = 0.0
        for node in node_list:
            score = _similarity(table.title, node.title) if table.title else 0.0
            if score > best_score:
                best_score = score
                best_node = node
        best_node.tables.append(table)


# ------------------------------------------------------------------
# TOC-based tree building (preferred strategy)
# ------------------------------------------------------------------

_PAGE_MARKER_RE = re.compile(r"<!--\s*nanoindex:page:(\d+)\s*-->")
_PAGE_HEADER_RE = re.compile(r"^## Page (\d+)\s*$", re.MULTILINE)


def _split_markdown_by_page(markdown: str, page_count: int) -> dict[int, str]:
    """Split merged markdown into per-page text using page markers."""
    page_texts: dict[int, list[str]] = {p: [] for p in range(1, page_count + 1)}
    current_page = 1

    for line in markdown.split("\n"):
        pm = _PAGE_MARKER_RE.search(line)
        if pm:
            current_page = int(pm.group(1))
            continue
        ph = _PAGE_HEADER_RE.match(line.strip())
        if ph:
            current_page = int(ph.group(1))
            continue
        page_texts.setdefault(current_page, []).append(line)

    return {p: "\n".join(lines).strip() for p, lines in page_texts.items()}


def _build_from_toc(
    toc: list[TOCEntry],
    markdown: str,
    page_count: int,
) -> list[TreeNode]:
    """Build a nested tree from TOC entries, filling each node with text
    from the pages it spans.

    TOC entries have ``parent_ids`` which define the nesting.  We compute
    each node's page range from its own page and its next sibling's page.
    """
    if not toc:
        return []

    page_texts = _split_markdown_by_page(markdown, page_count)

    id_to_node: dict[str, TreeNode] = {}

    for i, entry in enumerate(toc):
        next_page = page_count
        for j in range(i + 1, len(toc)):
            if toc[j].level <= entry.level:
                next_page = toc[j].page - 1
                break
        if next_page < entry.page:
            next_page = entry.page

        text_parts = [
            page_texts.get(p, "")
            for p in range(entry.page, next_page + 1)
        ]
        text = "\n\n".join(t for t in text_parts if t).strip()

        node = TreeNode(
            title=entry.title or f"Section (page {entry.page})",
            level=entry.level,
            text=text or None,
            start_index=entry.page,
            end_index=next_page,
        )
        id_to_node[entry.id] = node

    roots: list[TreeNode] = []
    for entry in toc:
        node = id_to_node[entry.id]
        parent_id = entry.parent_ids[-1] if entry.parent_ids else None
        if parent_id and parent_id in id_to_node:
            id_to_node[parent_id].nodes.append(node)
        else:
            roots.append(node)

    return roots


# ------------------------------------------------------------------
# Running-header detection
# ------------------------------------------------------------------

_BOILERPLATE_TITLES = {
    "table of contents", "contents", "index", "page",
}

_PAGE_TITLE_RE = re.compile(r"^page\s+\d+$", re.IGNORECASE)


def _is_boilerplate(title: str) -> bool:
    """True if this heading is a known boilerplate/running header."""
    t = title.strip().lower()
    return t in _BOILERPLATE_TITLES or bool(_PAGE_TITLE_RE.match(t))


def _find_running_headers(headings: list[HeadingNode]) -> set[str]:
    """Detect running headers and page markers to strip.

    Returns the set of title strings to remove.  Catches both:
    - Titles repeated on >30% of headings (e.g. "Table of Contents")
    - Known boilerplate patterns (e.g. "Page 42")
    """
    to_remove: set[str] = set()
    if len(headings) < 4:
        return to_remove

    from collections import Counter
    title_counts = Counter(h.title.strip() for h in headings)
    threshold = len(headings) * 0.3

    for title, count in title_counts.items():
        if count >= threshold or _is_boilerplate(title):
            to_remove.add(title)

    return to_remove


def _strip_boilerplate(
    headings: list[HeadingNode],
    to_remove: set[str],
) -> list[HeadingNode]:
    """Remove boilerplate headings, keeping only real section headings."""
    return [h for h in headings if h.title.strip() not in to_remove and not _is_boilerplate(h.title)]


# ------------------------------------------------------------------
# Heading level normalization
# ------------------------------------------------------------------

_PART_RE = re.compile(r"^PART\s+[IVX]+\b", re.IGNORECASE)
_ITEM_RE = re.compile(r"^Item\s+\d+[A-Z]?\b\.?", re.IGNORECASE)


def _reassign_page_text(
    nodes: list[TreeNode],
    markdown: str,
    page_count: int,
) -> None:
    """Replace heading-parser text with full per-page text for leaf nodes.

    The markdown heading parser only captures text between heading lines.
    Content not under any heading (financial tables, footers, etc.) gets
    lost.  This function reassigns each leaf node's text to the full
    markdown content of its page range, guaranteeing complete coverage.
    """
    page_texts = _split_markdown_by_page(markdown, page_count)

    for node in _iter_all(nodes):
        if node.nodes:
            continue
        if node.start_index <= 0:
            continue
        parts = [
            page_texts.get(p, "")
            for p in range(node.start_index, node.end_index + 1)
        ]
        full_text = "\n\n".join(t for t in parts if t).strip()
        if full_text:
            node.text = full_text


def _recover_orphan_pages(
    nodes: list[TreeNode],
    markdown: str,
    page_count: int,
) -> list[TreeNode]:
    """Create nodes for pages not covered by any existing tree node.

    After ``_assign_pages``, some pages may be "orphans" — present in the
    extraction markdown but not assigned to any node.  This typically
    happens when markdown headings only appear in part of the document
    (e.g. exhibits get headings but financial statement tables don't).

    Groups contiguous orphan pages into sections and appends them to
    the root of the tree so they're available for retrieval.
    """
    if page_count <= 0:
        return nodes

    covered: set[int] = set()
    for node in _iter_all(nodes):
        if node.start_index and node.end_index:
            for p in range(node.start_index, node.end_index + 1):
                covered.add(p)

    all_pages = set(range(1, page_count + 1))
    orphan_pages = sorted(all_pages - covered)
    if not orphan_pages:
        return nodes

    page_texts = _split_markdown_by_page(markdown, page_count)

    ranges: list[tuple[int, int]] = []
    start = orphan_pages[0]
    prev = start
    for p in orphan_pages[1:]:
        if p == prev + 1:
            prev = p
        else:
            ranges.append((start, prev))
            start = p
            prev = p
    ranges.append((start, prev))

    new_nodes: list[TreeNode] = []
    for rng_start, rng_end in ranges:
        parts = [page_texts.get(p, "") for p in range(rng_start, rng_end + 1)]
        text = "\n\n".join(t for t in parts if t).strip()
        if not text or len(text) < 20:
            continue

        first_text = page_texts.get(rng_start, "")
        title = f"Pages {rng_start}-{rng_end}" if rng_start != rng_end else f"Page {rng_start}"
        for line in first_text.split("\n"):
            cleaned = line.strip().lstrip("#").strip().strip("*").strip()
            if cleaned and len(cleaned) > 3 and not _is_boilerplate(cleaned):
                title = cleaned[:100]
                break

        new_nodes.append(TreeNode(
            title=title,
            level=1,
            text=text,
            start_index=rng_start,
            end_index=rng_end,
        ))

    if new_nodes:
        logger.info(
            "Recovered %d orphan page ranges (%d pages) as new tree nodes",
            len(new_nodes), len(orphan_pages),
        )
        nodes.extend(new_nodes)

    return nodes


def _deduplicate_parent_text(nodes: list[TreeNode]) -> None:
    """Remove text from parent nodes that is duplicated in their children.

    After nesting, a parent's ``text`` may contain the full content of its
    page range while its children already carry the same text in smaller
    sections.  This trims the parent to at most a short prefix (the text
    before the first child's content begins), preventing the retriever
    from pulling redundant megabytes when selecting a parent node.
    """
    for node in nodes:
        if not node.nodes:
            continue
        _deduplicate_parent_text(node.nodes)

        if not node.text:
            continue

        child_texts = {c.text for c in node.nodes if c.text}
        if not child_texts:
            continue

        # If the parent text fully contains any child text, it's duplicated.
        # Keep only the prefix before the first child's content starts.
        first_child_text = node.nodes[0].text
        if first_child_text and first_child_text[:200] in node.text:
            idx = node.text.find(first_child_text[:200])
            prefix = node.text[:idx].strip() if idx > 0 else ""
            node.text = prefix if prefix else None


def _deduplicate_sibling_branches(nodes: list[TreeNode]) -> list[TreeNode]:
    """Remove duplicate top-level branches with high text overlap.

    Some tree-building strategies produce near-identical sibling nodes
    (e.g. from duplicated TOC entries or redundant headings).  This
    removes later siblings whose text overlaps >80% with an earlier one.
    """
    if len(nodes) <= 1:
        return nodes

    def _branch_text(node: TreeNode) -> str:
        parts = [node.text or ""]
        for child in (node.nodes or []):
            parts.append(_branch_text(child))
        return "\n".join(parts)

    kept: list[TreeNode] = []
    kept_texts: list[str] = []

    for node in nodes:
        branch = _branch_text(node)
        if not branch.strip():
            kept.append(node)
            kept_texts.append("")
            continue

        is_dup = False
        for prev_text in kept_texts:
            if not prev_text:
                continue
            shorter = min(len(branch), len(prev_text))
            if shorter < 100:
                continue
            sample_len = min(shorter, 2000)
            overlap = sum(
                1 for a, b in zip(branch[:sample_len], prev_text[:sample_len])
                if a == b
            )
            if overlap / sample_len > 0.80:
                is_dup = True
                logger.info(
                    "Removing duplicate branch '%s' (%.0f%% overlap with earlier sibling)",
                    node.title[:60], (overlap / sample_len) * 100,
                )
                break

        if not is_dup:
            kept.append(node)
            kept_texts.append(branch)

    if len(kept) < len(nodes):
        logger.info(
            "Sibling dedup removed %d duplicate branches (%d → %d)",
            len(nodes) - len(kept), len(nodes), len(kept),
        )
    return kept


def _normalize_heading_levels(headings: list[HeadingNode]) -> list[HeadingNode]:
    """Reassign heading levels using document structure patterns.

    Financial filings (10-K, 10-Q) typically emit all headings at the same
    markdown level (or with inconsistent nesting).  When PART / Item
    patterns are detected, levels are **always** reassigned so that
    ``_nest_flat_nodes`` produces a proper hierarchy:

      - PART I / PART II / … → level 1
      - Item 1. / Item 1A. / Item 7. / … → level 2
      - Everything else → level 3
    """
    if len(headings) < 3:
        return headings

    has_parts = any(_PART_RE.match(h.title.strip()) for h in headings)
    has_items = any(_ITEM_RE.match(h.title.strip()) for h in headings)

    if not (has_parts or has_items):
        return headings

    result: list[HeadingNode] = []
    for h in headings:
        title = h.title.strip()
        if _PART_RE.match(title):
            new_level = 1
        elif _ITEM_RE.match(title):
            new_level = 2
        else:
            new_level = 3
        result.append(HeadingNode(
            title=h.title,
            level=new_level,
            line_number=h.line_number,
            page=h.page,
            text_content=h.text_content,
        ))

    logger.info(
        "Normalized heading levels: %d at L1, %d at L2, %d at L3",
        sum(1 for h in result if h.level == 1),
        sum(1 for h in result if h.level == 2),
        sum(1 for h in result if h.level == 3),
    )
    return result


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def build_document_tree(
    extraction: ExtractionResult,
    doc_name: str,
    config: NanoIndexConfig,
) -> DocumentTree:
    """Build a ``DocumentTree`` from an ``ExtractionResult``.

    Fallback cascade (first match wins):
      1. TOC (>=2 entries) — best structure, has page numbers and nesting
      2. Hierarchy JSON (>=2 sections with nesting)
      3. Dense *informative* markdown headings (running headers stripped)
      4. Section-grouped page tree (real headings become parents of pages)
      5. Page-based tree (one node per page)
      6. Single root node
    """
    has_toc = len(extraction.toc) >= 2

    hierarchy_nodes = _hierarchy_to_nodes(extraction.hierarchy_sections)
    heading_nodes = parse_markdown_headings(extraction.markdown)

    # Detect and strip running headers (e.g. "Table of Contents" on every page)
    boilerplate = _find_running_headers(heading_nodes)
    if boilerplate:
        real_headings = _strip_boilerplate(heading_nodes, boilerplate)
        logger.info(
            "Stripped %d boilerplate headings (%s) — %d real headings remain",
            len(heading_nodes) - len(real_headings),
            ", ".join(sorted(boilerplate)[:3]),
            len(real_headings),
        )
    else:
        real_headings = heading_nodes

    # Normalize heading levels for financial filings (PART/Item patterns)
    real_headings = _normalize_heading_levels(real_headings)

    heading_tree_nodes = _headings_to_flat_nodes(real_headings)

    hierarchy_has_structure = _has_structure(hierarchy_nodes)
    distinct_heading_levels = len({h.level for h in real_headings})
    headings_are_dense = (
        len(real_headings) > 1
        and extraction.page_count > 0
        and len(real_headings) >= extraction.page_count / 5
    )

    if has_toc:
        logger.info("Using TOC for tree structure (%d entries)", len(extraction.toc))
        tree_nodes = _build_from_toc(extraction.toc, extraction.markdown, extraction.page_count)
    elif hierarchy_has_structure:
        logger.info("Using hierarchy JSON for tree structure")
        tree_nodes = hierarchy_nodes
        _enrich_from_headings(tree_nodes, heading_nodes)
    elif len(real_headings) >= 3 and (headings_are_dense or distinct_heading_levels >= 2):
        heading_page_coverage = (
            len({h.page for h in real_headings}) / extraction.page_count
            if extraction.page_count else 0
        )
        if heading_page_coverage >= 0.15:
            logger.info(
                "Using markdown headings for tree (%d headings / %d levels / %d pages, %.0f%% page coverage)",
                len(real_headings), distinct_heading_levels, extraction.page_count,
                heading_page_coverage * 100,
            )
            tree_nodes = _nest_flat_nodes(heading_tree_nodes)
        elif extraction.page_count > 1:
            logger.info(
                "Headings cover only %.0f%% of pages — falling through to section-grouped tree",
                heading_page_coverage * 100,
            )
            tree_nodes = _build_section_grouped_tree(
                extraction.markdown, extraction.page_count, real_headings,
            )
        else:
            logger.info(
                "Headings cover only %.0f%% of pages (single page) — using headings anyway",
                heading_page_coverage * 100,
            )
            tree_nodes = _nest_flat_nodes(heading_tree_nodes)
    elif len(real_headings) >= 3 and extraction.page_count > 1:
        logger.info(
            "Using section-grouped tree (%d section headings across %d pages)",
            len(real_headings), extraction.page_count,
        )
        tree_nodes = _build_section_grouped_tree(
            extraction.markdown, extraction.page_count, real_headings,
        )
    elif extraction.page_count > 1:
        logger.info(
            "Headings too sparse (%d headings for %d pages) — building page-based tree",
            len(real_headings), extraction.page_count,
        )
        tree_nodes = _build_page_based_tree(
            extraction.markdown, extraction.page_count, real_headings,
        )
    else:
        logger.info("No structure detected — creating single root node")
        tree_nodes = [TreeNode(
            title=doc_name,
            level=1,
            text=extraction.markdown or None,
        )]

    _attach_bboxes(tree_nodes, extraction.bounding_boxes)
    _assign_pages(tree_nodes, extraction.page_count)
    tree_nodes = _recover_orphan_pages(
        tree_nodes, extraction.markdown, extraction.page_count,
    )
    _reassign_page_text(tree_nodes, extraction.markdown, extraction.page_count)
    _attach_tables(tree_nodes, extraction.hierarchy_tables)

    tree_nodes = _filter_low_confidence(tree_nodes, config.confidence_threshold)
    _deduplicate_parent_text(tree_nodes)
    tree_nodes = _deduplicate_sibling_branches(tree_nodes)

    assign_node_ids(tree_nodes)

    return DocumentTree(
        doc_name=doc_name,
        extraction_metadata={
            "extractor": "nanonets",
            "pages_processed": extraction.page_count,
            "processing_time": round(extraction.processing_time, 2),
        },
        structure=tree_nodes,
        all_bounding_boxes=extraction.bounding_boxes,
        page_dimensions=extraction.page_dimensions,
    )


def _build_section_grouped_tree(
    markdown: str,
    page_count: int,
    real_headings: list[HeadingNode],
) -> list[TreeNode]:
    """Group pages into sections defined by the real (non-running-header) headings.

    Each real heading becomes a parent node.  All pages from that heading
    until the next heading (at the same or higher level) become children.
    This gives documents like 10-K filings a meaningful tree:

        Item 1. Business (pp. 4-9)
          ├── Page 4
          ├── Page 5
          └── ...
        Item 1A. Risk Factors (pp. 10-15)
          └── ...
    """
    page_texts = _split_markdown_by_page(markdown, page_count)

    headings_sorted = sorted(real_headings, key=lambda h: h.page)

    sections: list[TreeNode] = []
    for i, heading in enumerate(headings_sorted):
        start_page = heading.page
        if i + 1 < len(headings_sorted):
            end_page = headings_sorted[i + 1].page - 1
        else:
            end_page = page_count
        end_page = max(end_page, start_page)

        section_text_parts = [
            page_texts.get(p, "") for p in range(start_page, end_page + 1)
        ]
        section_text = "\n\n".join(t for t in section_text_parts if t).strip()

        section_node = TreeNode(
            title=heading.title,
            level=heading.level,
            text=section_text or None,
            start_index=start_page,
            end_index=end_page,
        )

        span = end_page - start_page + 1
        if span > 1:
            for pg in range(start_page, end_page + 1):
                pg_text = page_texts.get(pg, "").strip()
                if not pg_text:
                    continue
                pg_title = _best_page_title([], pg_text, pg)
                section_node.nodes.append(TreeNode(
                    title=pg_title,
                    level=heading.level + 1,
                    text=pg_text,
                    start_index=pg,
                    end_index=pg,
                ))
            if section_node.nodes:
                section_node.text = None

        sections.append(section_node)

    # Handle pages before the first heading
    if headings_sorted and headings_sorted[0].page > 1:
        pre_pages = list(range(1, headings_sorted[0].page))
        pre_parts = [page_texts.get(p, "") for p in pre_pages]
        pre_text = "\n\n".join(t for t in pre_parts if t).strip()
        if pre_text:
            min_level = min(h.level for h in headings_sorted)
            preamble = TreeNode(
                title="Preamble",
                level=min_level,
                text=pre_text,
                start_index=1,
                end_index=headings_sorted[0].page - 1,
            )
            sections.insert(0, preamble)

    return _nest_flat_nodes(sections)


def _best_page_title(
    headings_on_page: list[HeadingNode],
    text: str,
    page_num: int,
) -> str:
    """Pick the best title for a page node, skipping boilerplate."""
    for h in headings_on_page:
        if h.title.strip().lower() not in _BOILERPLATE_TITLES:
            return h.title.strip()
    # No non-boilerplate heading — use first non-empty content line
    for line in text.split("\n"):
        cleaned = line.strip().lstrip("#").strip().strip("*").strip()
        if cleaned and cleaned.lower() not in _BOILERPLATE_TITLES and len(cleaned) > 3:
            return cleaned[:100]
    return f"Page {page_num}"


def _build_page_based_tree(
    markdown: str,
    page_count: int,
    heading_nodes: list[HeadingNode],
) -> list[TreeNode]:
    """Build a tree with one node per page, using headings as children where found.

    This handles documents where OCR detects few headings (financial reports,
    tables-heavy PDFs) — every page gets a node so no content is lost.
    """
    import re

    page_marker_re = re.compile(r"<!--\s*nanoindex:page:(\d+)\s*-->")
    page_header_re = re.compile(r"^## Page (\d+)\s*$", re.MULTILINE)

    page_texts: dict[int, list[str]] = {p: [] for p in range(1, page_count + 1)}

    current_page = 1
    for line in markdown.split("\n"):
        pm = page_marker_re.search(line)
        if pm:
            current_page = int(pm.group(1))
            continue
        ph = page_header_re.match(line.strip())
        if ph:
            current_page = int(ph.group(1))
            continue
        page_texts.setdefault(current_page, []).append(line)

    heading_by_page: dict[int, list[HeadingNode]] = {}
    for h in heading_nodes:
        heading_by_page.setdefault(h.page, []).append(h)

    nodes: list[TreeNode] = []
    for pg in range(1, page_count + 1):
        text = "\n".join(page_texts.get(pg, [])).strip()
        if not text:
            continue

        headings_on_page = heading_by_page.get(pg, [])
        title = _best_page_title(headings_on_page, text, pg)

        node = TreeNode(
            title=title,
            level=1,
            text=text,
            start_index=pg,
            end_index=pg,
        )

        for h in headings_on_page[1:]:
            node.nodes.append(TreeNode(
                title=h.title,
                level=2,
                text=h.text_content or None,
                start_index=pg,
                end_index=pg,
            ))

        nodes.append(node)

    return nodes


def _has_structure(nodes: list[TreeNode]) -> bool:
    """Return True if the hierarchy has meaningful nesting (not just one flat level)."""
    if not nodes:
        return False
    if len(nodes) >= 2:
        return True
    if nodes and any(n.nodes for n in nodes):
        return True
    return False
