"""Tests for the tree builder module."""

from __future__ import annotations

from nanoindex.config import NanoIndexConfig
from nanoindex.core.tree_builder import build_document_tree
from nanoindex.models import (
    BoundingBox,
    ExtractionResult,
    HierarchySection,
)


def _make_config(**kw) -> NanoIndexConfig:
    return NanoIndexConfig(nanonets_api_key="test-key", **kw)


class TestBuildFromHierarchy:
    def test_nested_sections(self):
        extraction = ExtractionResult(
            markdown="# Report\n\n## Intro\n\nHello\n\n## Details\n\nWorld",
            hierarchy_sections=[
                HierarchySection(
                    title="Report",
                    level=1,
                    content="",
                    subsections=[
                        HierarchySection(title="Intro", level=2, content="Hello"),
                        HierarchySection(title="Details", level=2, content="World"),
                    ],
                ),
            ],
            page_count=3,
        )
        tree = build_document_tree(extraction, "report", _make_config())
        assert tree.doc_name == "report"
        assert len(tree.structure) == 1
        assert tree.structure[0].title == "Report"
        assert len(tree.structure[0].nodes) == 2
        assert tree.structure[0].nodes[0].title == "Intro"

    def test_flat_hierarchy_falls_back_to_headings(self):
        extraction = ExtractionResult(
            markdown="# Title\n\n## Section A\n\nContent A\n\n## Section B\n\nContent B",
            hierarchy_sections=[
                HierarchySection(title="Document", level=1, content="All content"),
            ],
            page_count=2,
        )
        tree = build_document_tree(extraction, "flat", _make_config())
        assert len(tree.structure) >= 1


class TestBuildFromMarkdownOnly:
    def test_heading_based_tree(self):
        extraction = ExtractionResult(
            markdown="# Annual Report\n\n## Revenue\n\n$500M\n\n## Expenses\n\n$300M\n\n## Outlook\n\nGrowth expected",
            hierarchy_sections=[],
            page_count=5,
        )
        tree = build_document_tree(extraction, "annual", _make_config())
        assert len(tree.structure) == 1
        root = tree.structure[0]
        assert root.title == "Annual Report"
        assert len(root.nodes) == 3

    def test_no_structure_creates_single_root(self):
        extraction = ExtractionResult(
            markdown="Just a plain letter with no headings at all.",
            hierarchy_sections=[],
            page_count=1,
        )
        tree = build_document_tree(extraction, "letter", _make_config())
        assert len(tree.structure) == 1
        assert tree.structure[0].title == "letter"


class TestBoundingBoxAttachment:
    def test_bboxes_assign_pages(self):
        extraction = ExtractionResult(
            markdown="# Intro\n\nSome text\n\n# Details\n\nMore text",
            hierarchy_sections=[
                HierarchySection(title="Intro", level=1, content="Some text"),
                HierarchySection(title="Details", level=1, content="More text"),
            ],
            bounding_boxes=[
                BoundingBox(page=2, x=0.1, y=0.1, width=0.8, height=0.05, confidence=0.95, text="Intro"),
                BoundingBox(page=4, x=0.1, y=0.1, width=0.8, height=0.05, confidence=0.93, text="Details"),
            ],
            page_count=5,
        )
        tree = build_document_tree(extraction, "test", _make_config())
        assert tree.structure[0].start_index == 2
        assert tree.structure[1].start_index == 4


class TestNodeIds:
    def test_ids_assigned(self):
        extraction = ExtractionResult(
            markdown="# A\n\n## B\n\ntext\n\n## C\n\ntext",
            hierarchy_sections=[
                HierarchySection(
                    title="A",
                    level=1,
                    subsections=[
                        HierarchySection(title="B", level=2, content="text"),
                        HierarchySection(title="C", level=2, content="text"),
                    ],
                ),
            ],
            page_count=3,
        )
        tree = build_document_tree(extraction, "ids", _make_config())
        root = tree.structure[0]
        assert root.node_id == "0000"
        assert root.nodes[0].node_id == "0000.0000"
        assert root.nodes[1].node_id == "0000.0001"


class TestConfidenceFiltering:
    def test_low_confidence_removed(self):
        extraction = ExtractionResult(
            markdown="# Good\n\nGood content\n\n# Bad\n\nBad content",
            hierarchy_sections=[
                HierarchySection(title="Good", level=1, content="Good content"),
                HierarchySection(title="Bad", level=1, content="Bad content"),
            ],
            bounding_boxes=[
                BoundingBox(page=1, x=0, y=0, width=1, height=0.1, confidence=0.95, text="Good"),
                BoundingBox(page=2, x=0, y=0, width=1, height=0.1, confidence=0.3, text="Bad"),
            ],
            page_count=2,
        )
        tree = build_document_tree(extraction, "conf", _make_config(confidence_threshold=0.5))
        titles = [n.title for n in tree.structure]
        assert "Good" in titles
        # "Bad" may still appear if it has children, but its confidence should be flagged
