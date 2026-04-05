"""Tests for Pydantic models: serialization, validation, nesting."""

from __future__ import annotations

from nanoindex.models import (
    Answer,
    BoundingBox,
    Citation,
    DocumentTree,
    ExtractionResult,
    HierarchySection,
    RetrievedNode,
    TreeNode,
)


class TestTreeNode:
    def test_minimal(self):
        node = TreeNode(title="Intro")
        assert node.title == "Intro"
        assert node.nodes == []
        assert node.level == 1

    def test_nested(self):
        child = TreeNode(title="Sub", level=2)
        parent = TreeNode(title="Main", level=1, nodes=[child])
        assert len(parent.nodes) == 1
        assert parent.nodes[0].title == "Sub"

    def test_round_trip(self):
        node = TreeNode(
            title="Revenue",
            node_id="0001",
            start_index=3,
            end_index=5,
            summary="Revenue details",
            confidence=0.95,
        )
        data = node.model_dump()
        restored = TreeNode.model_validate(data)
        assert restored.title == node.title
        assert restored.confidence == 0.95


class TestDocumentTree:
    def test_full_tree(self):
        tree = DocumentTree(
            doc_name="test.pdf",
            structure=[
                TreeNode(
                    title="Chapter 1",
                    node_id="0000",
                    start_index=1,
                    end_index=10,
                    nodes=[
                        TreeNode(title="Section 1.1", node_id="0000.0000", level=2),
                    ],
                ),
            ],
        )
        assert tree.doc_name == "test.pdf"
        assert len(tree.structure) == 1
        assert len(tree.structure[0].nodes) == 1

    def test_serialization_excludes_none(self):
        tree = DocumentTree(doc_name="test.pdf", structure=[])
        data = tree.model_dump(exclude_none=True)
        assert "doc_description" not in data


class TestBoundingBox:
    def test_defaults(self):
        bb = BoundingBox(page=1, x=0.1, y=0.2, width=0.5, height=0.3)
        assert bb.confidence == 1.0
        assert bb.region_type == "unknown"


class TestExtractionResult:
    def test_empty(self):
        r = ExtractionResult()
        assert r.markdown == ""
        assert r.page_count == 0

    def test_with_sections(self):
        sec = HierarchySection(title="Intro", level=1, content="Hello")
        r = ExtractionResult(hierarchy_sections=[sec])
        assert len(r.hierarchy_sections) == 1


class TestAnswer:
    def test_with_citations(self):
        a = Answer(
            content="Revenue was $500M",
            citations=[Citation(node_id="0001", title="Revenue", pages=[3, 4])],
            mode="text",
        )
        assert a.mode == "text"
        assert a.citations[0].pages == [3, 4]


class TestRetrievedNode:
    def test_basic(self):
        node = TreeNode(title="Test")
        rn = RetrievedNode(node=node, text="Sample text")
        assert rn.text == "Sample text"
        assert rn.page_images == []


def test_kb_document():
    from nanoindex.models import KBDocument
    doc = KBDocument(doc_id="abc", doc_name="test", source_path="/tmp/test.pdf", added_at="2026-01-01", tree_path="trees/test.json")
    assert doc.doc_name == "test"


def test_kb_config():
    from nanoindex.models import KBConfig
    config = KBConfig(created_at="2026-01-01")
    assert config.version == "1"
    assert config.documents == []
