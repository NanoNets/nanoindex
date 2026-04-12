"""Validate tree quality after building.

Checks structural integrity, page coverage, and content completeness.
"""

from __future__ import annotations

import logging
from nanoindex.models import DocumentTree
from nanoindex.utils.tree_ops import iter_nodes

logger = logging.getLogger(__name__)


class TreeValidationResult:
    def __init__(self):
        self.passed = True
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.stats: dict = {}

    def warn(self, msg: str):
        self.warnings.append(msg)

    def error(self, msg: str):
        self.errors.append(msg)
        self.passed = False


def validate_tree(tree: DocumentTree) -> TreeValidationResult:
    """Validate a tree for structural integrity and content quality."""
    result = TreeValidationResult()
    nodes = list(iter_nodes(tree.structure))

    result.stats["total_nodes"] = len(nodes)
    result.stats["doc_name"] = tree.doc_name

    # 1. Must have at least one node
    if not nodes:
        result.error("Tree has zero nodes")
        return result

    # 2. Page coverage — every page should be in at least one node
    all_pages = set()
    for n in nodes:
        for p in range(n.start_index, n.end_index + 1):
            all_pages.add(p)

    if all_pages:
        min_page = min(all_pages)
        max_page = max(all_pages)
        expected_pages = set(range(min_page, max_page + 1))
        missing_pages = expected_pages - all_pages
        if missing_pages:
            result.warn(f"Pages not covered by any node: {sorted(missing_pages)[:10]}")
        result.stats["page_coverage"] = len(all_pages) / max(len(expected_pages), 1)
        result.stats["total_pages"] = max_page
    else:
        result.warn("No page ranges assigned to nodes")
        result.stats["page_coverage"] = 0

    # 3. Node IDs should be unique
    node_ids = [n.node_id for n in nodes if n.node_id]
    if len(node_ids) != len(set(node_ids)):
        result.error("Duplicate node IDs found")

    # 4. Nodes should have text content
    empty_nodes = [n for n in nodes if not (n.text or "").strip()]
    if empty_nodes:
        pct = len(empty_nodes) / len(nodes)
        if pct > 0.5:
            result.warn(f"{len(empty_nodes)}/{len(nodes)} nodes have no text ({pct:.0%})")
        result.stats["empty_nodes"] = len(empty_nodes)

    # 5. Tree depth — should have some hierarchy for docs > 5 pages
    max_level = max(n.level for n in nodes)
    result.stats["max_depth"] = max_level
    if max_level == 1 and len(nodes) > 10:
        result.warn(
            "Flat tree (all nodes at level 1) for a large document — hierarchy may be missing"
        )

    # 6. Summaries — check if enrichment ran
    nodes_with_summary = sum(1 for n in nodes if n.summary)
    result.stats["nodes_with_summary"] = nodes_with_summary
    result.stats["summary_coverage"] = nodes_with_summary / len(nodes) if nodes else 0

    # 7. Bounding boxes
    total_bboxes = len(tree.all_bounding_boxes)
    result.stats["bounding_boxes"] = total_bboxes
    node_bboxes = sum(len(n.bounding_boxes) for n in nodes)
    result.stats["node_bounding_boxes"] = node_bboxes

    # 8. Very large nodes (potential splitting needed)
    large_nodes = [n for n in nodes if len(n.text or "") > 50000]
    if large_nodes:
        result.warn(f"{len(large_nodes)} nodes have >50K chars — may need further splitting")

    # Log results
    if result.errors:
        logger.warning("Tree validation FAILED for %s: %s", tree.doc_name, result.errors)
    elif result.warnings:
        logger.info(
            "Tree validation PASSED with %d warnings for %s", len(result.warnings), tree.doc_name
        )
    else:
        logger.info(
            "Tree validation PASSED for %s (%d nodes, %.0f%% page coverage)",
            tree.doc_name,
            len(nodes),
            result.stats.get("page_coverage", 0) * 100,
        )

    return result
