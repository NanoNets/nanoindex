"""Pydantic data models shared across the NanoIndex pipeline.

Every structured object flowing between modules is defined here so that
the entire codebase has a single source of truth for shapes and validation.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


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

    markdown: Any = ""

    @model_validator(mode="before")
    @classmethod
    def _coerce_markdown(cls, data: Any) -> Any:
        """Ensure markdown is always a string, even if API returns a dict."""
        if isinstance(data, dict):
            md = data.get("markdown", "")
            while isinstance(md, dict):
                md = md.get("content", "") or md.get("markdown", "") or md.get("text", "") or ""
            if not isinstance(md, str):
                md = str(md) if md else ""
            data["markdown"] = md
        return data
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


class ExtractedTable(BaseModel):
    """A structured table extracted from a document."""

    name: str = ""
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    source_pages: list[int] = Field(default_factory=list)
    totals: dict[str, float] = Field(default_factory=dict)
    validation: dict[str, Any] = Field(default_factory=dict)


class ExtractedForm(BaseModel):
    """Key-value fields extracted from a form/template document."""

    schema_name: str = ""
    fields: dict[str, Any] = Field(default_factory=dict)
    source_pages: list[int] = Field(default_factory=list)
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Knowledge Base models
# ---------------------------------------------------------------------------

class KBDocument(BaseModel):
    """Metadata for a document tracked by the KnowledgeBase."""

    doc_id: str
    doc_name: str
    source_path: str
    added_at: str  # ISO timestamp
    tree_path: str
    graph_path: str | None = None
    embeddings_path: str | None = None
    content_hash: str = ""  # MD5 of tree JSON at time of ingestion


class KBConfig(BaseModel):
    """Persistent configuration for a wiki directory."""

    version: str = "1"
    created_at: str = ""
    documents: list[KBDocument] = Field(default_factory=list)
    concept_index: dict[str, list[str]] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Self-correcting extraction models
# ---------------------------------------------------------------------------

class ValidationResult(BaseModel):
    """Result of validating extracted data against document anchors."""

    passed: bool = False
    row_count_match: bool | None = None
    row_count_expected: int | None = None
    row_count_actual: int | None = None
    total_mismatches: list[dict[str, Any]] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)


class ExtractionResult2(BaseModel):
    """Result of structured extraction (table or form mode)."""

    rows: list[dict[str, Any]] = Field(default_factory=list)
    fields: dict[str, Any] = Field(default_factory=dict)
    columns: list[str] = Field(default_factory=list)
    validation: ValidationResult = Field(default_factory=ValidationResult)
    corrections: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    mode: str = ""  # "table" or "form"
    source_pages: list[int] = Field(default_factory=list)
    doc_name: str = ""

    def to_csv(self, path: str) -> None:
        """Write extracted rows to a CSV file."""
        import csv

        with open(path, "w", newline="") as f:
            if self.rows:
                fieldnames = self.columns or list(self.rows[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.rows)

    def to_json(self, path: str) -> None:
        """Write the full extraction result to a JSON file."""
        import json

        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)
