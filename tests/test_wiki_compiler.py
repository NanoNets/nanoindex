"""Tests for the wiki compiler module."""

from pathlib import Path

import pytest

from nanoindex.core.wiki_compiler import (
    compile_concept_page,
    compile_document_page,
    compile_index_page,
    compile_query_page,
    incremental_update,
)
from nanoindex.models import (
    Citation,
    DocumentGraph,
    DocumentTree,
    Entity,
    KBConfig,
    KBDocument,
    Relationship,
    TreeNode,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_tree() -> DocumentTree:
    """Create a sample tree with 3 nodes."""
    return DocumentTree(
        doc_name="Annual Report 2025",
        doc_description="Company annual report",
        structure=[
            TreeNode(
                title="Introduction",
                node_id="0001",
                start_index=1,
                end_index=3,
                level=1,
            ),
            TreeNode(
                title="Financial Statements",
                node_id="0002",
                start_index=4,
                end_index=8,
                level=1,
                nodes=[
                    TreeNode(
                        title="Balance Sheet",
                        node_id="0002.0001",
                        start_index=4,
                        end_index=5,
                        level=2,
                    ),
                ],
            ),
            TreeNode(
                title="Appendix",
                node_id="0003",
                start_index=9,
                end_index=10,
                level=1,
            ),
        ],
    )


def _make_graph() -> DocumentGraph:
    """Create a sample graph with entities and relationships."""
    return DocumentGraph(
        doc_name="Annual Report 2025",
        entities=[
            Entity(
                name="Acme Corp",
                entity_type="Organization",
                description="A multinational conglomerate",
                source_node_ids=["0001"],
            ),
            Entity(
                name="Revenue",
                entity_type="Financial Metric",
                description="Total revenue for the fiscal year",
                source_node_ids=["0002"],
            ),
        ],
        relationships=[
            Relationship(
                source="Acme Corp",
                target="Revenue",
                keywords="reports",
                description="Acme Corp reports Revenue",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCompileDocumentPage:
    def test_basic_output(self):
        tree = _make_tree()
        graph = _make_graph()
        md = compile_document_page(tree, graph)

        # Title
        assert "# Annual Report 2025" in md

        # Node names present
        assert "Introduction" in md
        assert "Financial Statements" in md
        assert "Balance Sheet" in md
        assert "Appendix" in md

        # Page numbers in outline
        assert "1" in md
        assert "10" in md

        # Stats table
        assert "Nodes" in md
        assert "4" in md  # 4 nodes total (3 top-level + 1 child)
        assert "Entities" in md
        assert "2" in md  # 2 entities

    def test_no_graph(self):
        tree = _make_tree()
        md = compile_document_page(tree, None)

        assert "# Annual Report 2025" in md
        assert "Entities | 0" in md
        assert "## Entities" not in md

    def test_entity_backlinks(self):
        tree = _make_tree()
        graph = _make_graph()
        md = compile_document_page(tree, graph)

        assert "[[concepts/acme-corp|Acme Corp]]" in md
        assert "[[concepts/revenue|Revenue]]" in md


class TestCompileConceptPage:
    def test_basic_output(self):
        md = compile_concept_page(
            name="Acme Corp",
            entity_type="Organization",
            descriptions=["A large company", "A multinational conglomerate headquartered in NY"],
            source_docs=[
                ("Annual Report 2025", "annual-report-2025"),
                ("Q1 Earnings", "q1-earnings"),
            ],
            relationships=[
                ("Revenue", "revenue", "reports"),
                ("CEO", "ceo", "led by"),
            ],
        )

        assert "# Acme Corp" in md
        assert "**Type:** Organization" in md

        # Backlinks in [[]] format
        assert "[[documents/annual-report-2025|Annual Report 2025]]" in md
        assert "[[documents/q1-earnings|Q1 Earnings]]" in md
        assert "[[concepts/revenue|Revenue]]" in md
        assert "[[concepts/ceo|CEO]]" in md

    def test_longest_description_first(self):
        md = compile_concept_page(
            name="Test",
            entity_type="Other",
            descriptions=["short", "this is the longest description of them all"],
            source_docs=[],
            relationships=[],
        )

        # Longest should appear before the shorter one
        long_pos = md.index("this is the longest description of them all")
        short_pos = md.index("short")
        assert long_pos < short_pos

    def test_empty_descriptions(self):
        md = compile_concept_page(
            name="Empty",
            entity_type="Other",
            descriptions=[],
            source_docs=[],
            relationships=[],
        )
        assert "# Empty" in md


class TestCompileIndexPage:
    def test_basic_output(self):
        config = KBConfig(
            version="1",
            created_at="2025-01-01",
            documents=[
                KBDocument(
                    doc_id="d1",
                    doc_name="Report A",
                    source_path="/a.pdf",
                    added_at="2025-01-01",
                    tree_path="/trees/a.json",
                ),
                KBDocument(
                    doc_id="d2",
                    doc_name="Report B",
                    source_path="/b.pdf",
                    added_at="2025-01-02",
                    tree_path="/trees/b.json",
                ),
            ],
            concept_index={
                "Acme Corp": ["d1"],
                "Revenue": ["d1", "d2"],
            },
        )

        md = compile_index_page(config, query_count=5)

        assert "# Knowledge Base Index" in md
        # Document links
        assert "[[documents/report-a|Report A]]" in md
        assert "[[documents/report-b|Report B]]" in md
        # Concept links
        assert "[[concepts/acme-corp|Acme Corp]]" in md
        assert "[[concepts/revenue|Revenue]]" in md
        # Counts
        assert "**Documents:** 2" in md
        assert "**Concepts:** 2" in md
        assert "**Recent queries:** 5" in md


class TestCompileQueryPage:
    def test_basic_output(self):
        citations = [
            Citation(
                node_id="0001",
                title="Introduction",
                doc_name="Annual Report 2025",
                pages=[1, 2],
            ),
            Citation(
                node_id="0002",
                title="Financials",
                doc_name="Annual Report 2025",
                pages=[4, 5, 6],
            ),
        ]
        concepts = [
            ("Acme Corp", "acme-corp"),
            ("Revenue", "revenue"),
        ]

        md = compile_query_page(
            question="What was the revenue?",
            answer_content="The revenue was $1B.",
            citations=citations,
            concepts=concepts,
        )

        # Question and answer
        assert "# Query: What was the revenue?" in md
        assert "The revenue was $1B." in md

        # Citation backlinks
        assert "[[documents/annual-report-2025|Introduction]]" in md
        assert "node 0001" in md
        assert "pp. 1, 2" in md

        # Concept backlinks
        assert "[[concepts/acme-corp|Acme Corp]]" in md
        assert "[[concepts/revenue|Revenue]]" in md

    def test_no_citations(self):
        md = compile_query_page(
            question="Test?",
            answer_content="Answer.",
            citations=[],
            concepts=[],
        )
        assert "# Query: Test?" in md
        assert "## Citations" not in md


class TestIncrementalUpdate:
    def test_files_created(self, tmp_path: Path):
        wiki_path = tmp_path / "wiki"
        wiki_path.mkdir()

        new_doc = KBDocument(
            doc_id="d1",
            doc_name="Annual Report 2025",
            source_path="/report.pdf",
            added_at="2025-01-01",
            tree_path="/trees/d1.json",
        )
        tree = _make_tree()
        graph = _make_graph()

        config = KBConfig(
            version="1",
            created_at="2025-01-01",
            documents=[new_doc],
            concept_index={},
        )

        all_graphs = {"Annual Report 2025": graph}

        incremental_update(
            wiki_path=wiki_path,
            new_doc=new_doc,
            new_tree=tree,
            new_graph=graph,
            config=config,
            all_graphs=all_graphs,
        )

        # Document page created
        doc_file = wiki_path / "documents" / "annual-report-2025.md"
        assert doc_file.exists()
        doc_content = doc_file.read_text()
        assert "# Annual Report 2025" in doc_content

        # Concept pages created
        acme_file = wiki_path / "concepts" / "acme-corp.md"
        assert acme_file.exists()
        acme_content = acme_file.read_text()
        assert "[[documents/annual-report-2025|Annual Report 2025]]" in acme_content

        revenue_file = wiki_path / "concepts" / "revenue.md"
        assert revenue_file.exists()

        # Index page created
        index_file = wiki_path / "_index.md"
        assert index_file.exists()
        index_content = index_file.read_text()
        assert "[[documents/annual-report-2025|Annual Report 2025]]" in index_content

        # concept_index updated
        assert "Acme Corp" in config.concept_index
        assert "d1" in config.concept_index["Acme Corp"]
        assert "Revenue" in config.concept_index

    def test_no_graph(self, tmp_path: Path):
        wiki_path = tmp_path / "wiki"
        wiki_path.mkdir()

        new_doc = KBDocument(
            doc_id="d1",
            doc_name="Simple Doc",
            source_path="/simple.pdf",
            added_at="2025-01-01",
            tree_path="/trees/d1.json",
        )
        tree = DocumentTree(
            doc_name="Simple Doc",
            structure=[
                TreeNode(title="Section 1", node_id="0001", start_index=1, end_index=2),
            ],
        )
        config = KBConfig(
            version="1",
            created_at="2025-01-01",
            documents=[new_doc],
        )

        incremental_update(
            wiki_path=wiki_path,
            new_doc=new_doc,
            new_tree=tree,
            new_graph=None,
            config=config,
            all_graphs={},
        )

        doc_file = wiki_path / "documents" / "simple-doc.md"
        assert doc_file.exists()

        # No concept files
        concepts_dir = wiki_path / "concepts"
        assert list(concepts_dir.iterdir()) == []

        # Index still created
        assert (wiki_path / "_index.md").exists()
