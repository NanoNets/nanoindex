"""Build citation objects from retrieved nodes.

Single shared implementation used by both agentic.py and generator.py.
"""

from __future__ import annotations

from nanoindex.models import (
    BoundingBox,
    Citation,
    DocumentTree,
    PageDimensions,
    RetrievedNode,
)


def build_citations(
    nodes: list[RetrievedNode],
    tree: DocumentTree | None = None,
    include_metadata: bool = False,
) -> list[Citation]:
    """Build citations from retrieved nodes.

    Each citation carries:
    - The node's bounding boxes (content-level, from page overlap)
    - Page dimensions (when include_metadata=True)

    The citation_resolver narrows bboxes to only those matching the answer
    text AFTER the answer is generated. This function just collects them.
    """
    citations: list[Citation] = []
    for rn in nodes:
        pages = (
            list(range(rn.node.start_index, rn.node.end_index + 1)) if rn.node.start_index else []
        )

        # Use bboxes from the node (content-level after _attach_content_bboxes)
        bboxes: list[BoundingBox] = list(rn.node.bounding_boxes)

        # If node has no bboxes, try tree-level bboxes for its pages
        if not bboxes and pages and tree and tree.all_bounding_boxes:
            page_set = set(pages)
            bboxes = [bb for bb in tree.all_bounding_boxes if bb.page in page_set]

        # Page dimensions (only when metadata requested)
        dims: list[PageDimensions] = []
        if include_metadata and pages and tree:
            page_set = set(pages)
            dims = [pd for pd in tree.page_dimensions if pd.page in page_set]

        citations.append(
            Citation(
                node_id=rn.node.node_id,
                title=rn.node.title,
                doc_name=rn.doc_name,
                pages=pages,
                bounding_boxes=bboxes,
                page_dimensions=dims,
            )
        )
    return citations
