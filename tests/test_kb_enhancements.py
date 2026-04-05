"""Tests for wiki enhancements: log.md, content hash, type-specific templates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from nanoindex.core.wiki_compiler import compile_concept_page
from nanoindex.kb import KnowledgeBase
from nanoindex.models import DocumentTree, KBDocument, TreeNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tree(doc_name: str = "test-doc") -> DocumentTree:
    """Create a minimal DocumentTree for testing."""
    return DocumentTree(
        doc_name=doc_name,
        doc_description="A test document.",
        structure=[
            TreeNode(
                title="Section 1",
                node_id="0001",
                start_index=1,
                end_index=2,
                summary="First section",
                nodes=[
                    TreeNode(
                        title="Sub 1.1",
                        node_id="0001.0001",
                        start_index=1,
                        end_index=1,
                    ),
                ],
            ),
            TreeNode(
                title="Section 2",
                node_id="0002",
                start_index=3,
                end_index=4,
                summary="Second section",
            ),
        ],
    )


def _setup_kb_with_doc(tmp_path: Path) -> KnowledgeBase:
    """Create a KB and manually add a document (bypasses NanoIndex pipeline)."""
    kb = KnowledgeBase(tmp_path / "kb")

    tree = _make_tree("report")
    tree_rel = "trees/report.json"
    tree_path = kb._data_dir / tree_rel
    tree_path.parent.mkdir(parents=True, exist_ok=True)
    tree_data = tree.model_dump(exclude_none=True)
    with open(tree_path, "w") as f:
        json.dump(tree_data, f, indent=2, ensure_ascii=False)

    tree_json = json.dumps(tree_data, sort_keys=True)
    content_hash = hashlib.md5(tree_json.encode()).hexdigest()

    doc = KBDocument(
        doc_id="report",
        doc_name="report",
        source_path="/tmp/report.pdf",
        added_at="2026-04-06T00:00:00+00:00",
        tree_path=tree_rel,
        content_hash=content_hash,
    )
    kb._config.documents.append(doc)
    kb._config.concept_index["revenue"] = ["report"]
    kb._save_config()
    return kb


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLogMd:
    """Tests for log.md append-only activity log."""

    def test_log_created_on_add(self, tmp_path):
        """Adding a doc should create log.md with an 'ingest' entry."""
        kb = KnowledgeBase(tmp_path / "kb")
        tree = _make_tree("report")

        # Patch async_index to return our tree directly
        with patch.object(kb._ni, "async_index", new_callable=AsyncMock, return_value=tree):
            kb.add(tmp_path / "report.pdf")

        log_path = tmp_path / "kb" / "log.md"
        assert log_path.exists(), "log.md should be created after add()"
        content = log_path.read_text(encoding="utf-8")
        assert "ingest" in content
        assert "report.pdf" in content

    def test_log_created_on_lint(self, tmp_path):
        """Running lint should append a 'lint' entry to log.md."""
        kb = KnowledgeBase(tmp_path / "kb")
        kb.lint()

        log_path = tmp_path / "kb" / "log.md"
        assert log_path.exists(), "log.md should be created after lint()"
        content = log_path.read_text(encoding="utf-8")
        assert "lint" in content
        assert "health check" in content


class TestContentHash:
    """Tests for source provenance hashing."""

    def test_content_hash(self, tmp_path):
        """After add(), KBDocument.content_hash should be a non-empty MD5."""
        kb = KnowledgeBase(tmp_path / "kb")
        tree = _make_tree("report")

        with patch.object(kb._ni, "async_index", new_callable=AsyncMock, return_value=tree):
            doc = kb.add(tmp_path / "report.pdf")

        assert doc.content_hash != "", "content_hash should be non-empty"
        assert len(doc.content_hash) == 32, "MD5 hex digest should be 32 chars"

    def test_lint_detects_stale_hash(self, tmp_path):
        """lint() should warn when tree file changes after ingestion."""
        kb = _setup_kb_with_doc(tmp_path)

        # Tamper with the tree file
        tree_path = kb._data_dir / "trees" / "report.json"
        with open(tree_path) as f:
            data = json.load(f)
        data["doc_description"] = "tampered!"
        with open(tree_path, "w") as f:
            json.dump(data, f)

        warnings = kb.lint()
        stale_warnings = [w for w in warnings if "Stale" in w]
        assert len(stale_warnings) == 1, f"Expected 1 stale warning, got: {warnings}"


class TestTypeSpecificTemplates:
    """Tests for type-specific concept page templates."""

    def test_organization_template(self):
        output = compile_concept_page(
            name="Acme Corp",
            entity_type="Organization",
            descriptions=["A large corporation."],
            source_docs=[("report", "report")],
            relationships=[("Bob Smith", "bob-smith", "CEO of")],
        )
        assert "## Key People" in output
        assert "## Financial Data" in output
        assert "## Related Orgs" in output
        assert "## Description" in output

    def test_person_template(self):
        output = compile_concept_page(
            name="Bob Smith",
            entity_type="Person",
            descriptions=["CEO of Acme Corp."],
            source_docs=[("report", "report")],
            relationships=[("Acme Corp", "acme-corp", "CEO of")],
        )
        assert "## Role" in output
        assert "## Affiliations" in output
        assert "## Mentions" in output
        assert "## Description" in output

    def test_financial_item_template(self):
        output = compile_concept_page(
            name="Revenue",
            entity_type="FinancialItem",
            descriptions=["Total revenue for FY2025."],
            source_docs=[("report", "report")],
            relationships=[],
        )
        assert "## Values" in output
        assert "## Trends" in output
        assert "## Sources" in output
        assert "## Description" in output

    def test_metric_template(self):
        output = compile_concept_page(
            name="Net Income",
            entity_type="Metric",
            descriptions=["Bottom line profit."],
            source_docs=[("report", "report")],
            relationships=[],
        )
        assert "## Values" in output
        assert "## Trends" in output
        assert "## Sources" in output

    def test_time_period_template(self):
        output = compile_concept_page(
            name="FY2025",
            entity_type="TimePeriod",
            descriptions=["Fiscal year ending Dec 2025."],
            source_docs=[("report", "report")],
            relationships=[("FY2024", "fy2024", "previous period")],
        )
        assert "## Events" in output
        assert "## Related Periods" in output
        assert "## Description" in output

    def test_default_template(self):
        output = compile_concept_page(
            name="SomeEntity",
            entity_type="Other",
            descriptions=["Some description."],
            source_docs=[("report", "report")],
            relationships=[("X", "x", "related")],
        )
        # Default should NOT have type-specific sections
        assert "## Key People" not in output
        assert "## Role" not in output
        assert "## Values" not in output
        assert "## Events" not in output
        # But should have standard sections
        assert "## Source Documents" in output
        assert "## Related Concepts" in output
