"""Pydantic data models shared across the NanoIndex pipeline.

Every structured object flowing between modules is defined here so that
the entire codebase has a single source of truth for shapes and validation.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Bounding box / layout metadata
# ---------------------------------------------------------------------------

class BoundingBox(BaseModel):
    """Normalised coordinates for a single detected region on a page."""

    page: int
    x: float
    y: float
    width: float
    height: float
    confidence: float = 1.0
    region_type: str = "unknown"  # heading, paragraph, table, image, …
    text: str | None = None


class PageDimensions(BaseModel):
    page: int
    width: int
    height: int


# ---------------------------------------------------------------------------
# Extraction result (output of Nanonets API)
# ---------------------------------------------------------------------------

class HierarchySection(BaseModel):
    """A single section from the ``hierarchy_output`` JSON."""

    id: str = ""
    title: str = ""
    level: int = 1
    content: str = ""
    subsections: list[HierarchySection] = Field(default_factory=list)


class HierarchyTable(BaseModel):
    id: str = ""
    title: str = ""
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)


class HierarchyKVPair(BaseModel):
    key: str
    value: str


class TOCEntry(BaseModel):
    """A single entry from the ``table-of-contents`` JSON."""

    id: str = ""
    title: str = ""
    level: int = 1
    page: int = 0
    parent_ids: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """Normalised output from a Nanonets extraction call."""

    markdown: str = ""
    toc: list[TOCEntry] = Field(default_factory=list)
    hierarchy_sections: list[HierarchySection] = Field(default_factory=list)
    hierarchy_tables: list[HierarchyTable] = Field(default_factory=list)
    hierarchy_kv_pairs: list[HierarchyKVPair] = Field(default_factory=list)
    bounding_boxes: list[BoundingBox] = Field(default_factory=list)
    page_dimensions: list[PageDimensions] = Field(default_factory=list)
    page_count: int = 0
    processing_time: float = 0.0
    record_id: str = ""


# ---------------------------------------------------------------------------
# Parsed document (universal output from any parser)
# ---------------------------------------------------------------------------

class ModalContent(BaseModel):
    """A non-text element extracted from a document."""

    content_type: str  # "image", "table", "equation"
    page: int = 0
    content: str = ""  # raw content (markdown table, latex, etc.)
    image_path: str | None = None  # path to extracted image file
    caption: str = ""
    surrounding_text: str = ""  # context from nearby text
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    """Universal output from any document parser."""

    markdown: str = ""
    pages: list[str] = Field(default_factory=list)  # per-page text
    page_count: int = 0
    modal_contents: list[ModalContent] = Field(default_factory=list)
    toc: list[TOCEntry] = Field(default_factory=list)
    hierarchy_sections: list[HierarchySection] = Field(default_factory=list)
    bounding_boxes: list[BoundingBox] = Field(default_factory=list)
    page_dimensions: list[PageDimensions] = Field(default_factory=list)
    processing_time: float = 0.0
    parser_name: str = ""

    def to_extraction_result(self) -> ExtractionResult:
        """Convert to an ExtractionResult for backward compatibility with the tree builder."""
        return ExtractionResult(
            markdown=self.markdown,
            toc=self.toc,
            hierarchy_sections=self.hierarchy_sections,
            bounding_boxes=self.bounding_boxes,
            page_dimensions=self.page_dimensions,
            page_count=self.page_count,
            processing_time=self.processing_time,
        )


# ---------------------------------------------------------------------------
# Tree nodes (the core index structure)
# ---------------------------------------------------------------------------

class TreeNode(BaseModel):
    """A single node in the document tree index."""

    title: str
    node_id: str = ""
    start_index: int = 0  # 1-based page number
    end_index: int = 0
    level: int = 1
    summary: str | None = None
    text: str | None = None
    confidence: float = 1.0
    bounding_boxes: list[BoundingBox] = Field(default_factory=list)
    tables: list[HierarchyTable] = Field(default_factory=list)
    nodes: list[TreeNode] = Field(default_factory=list)


class DocumentTree(BaseModel):
    """Top-level output: the complete indexed document tree."""

    doc_name: str
    doc_description: str | None = None
    extraction_metadata: dict[str, Any] = Field(default_factory=dict)
    structure: list[TreeNode] = Field(default_factory=list)
    all_bounding_boxes: list[BoundingBox] = Field(default_factory=list)
    page_dimensions: list[PageDimensions] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Retrieval / generation models
# ---------------------------------------------------------------------------

class RetrievedNode(BaseModel):
    """A tree node selected during tree search, enriched with content."""

    node: TreeNode
    text: str = ""
    page_images: list[str] = Field(default_factory=list)  # file paths
    doc_name: str | None = None


class Citation(BaseModel):
    node_id: str
    title: str
    doc_name: str | None = None
    pages: list[int] = Field(default_factory=list)
    bounding_boxes: list[BoundingBox] = Field(default_factory=list)
    page_dimensions: list[PageDimensions] = Field(default_factory=list)


class Answer(BaseModel):
    """Structured answer returned by the generation step."""

    content: str
    reasoning: str = ""
    citations: list[Citation] = Field(default_factory=list)
    mode: str = "text"  # "text" | "vision"


# ---------------------------------------------------------------------------
# Graph models (entity-relationship layer)
# ---------------------------------------------------------------------------

class Entity(BaseModel):
    """An entity extracted from one or more tree nodes."""

    name: str
    entity_type: str = "Other"
    description: str = ""
    source_node_ids: list[str] = Field(default_factory=list)


class Relationship(BaseModel):
    """A relationship between two entities."""

    source: str
    target: str
    keywords: str = ""
    description: str = ""
    source_node_ids: list[str] = Field(default_factory=list)


class DocumentGraph(BaseModel):
    """Entity-relationship graph extracted from a document tree."""

    doc_name: str = ""
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
