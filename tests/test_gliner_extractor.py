"""Tests for GLiNER extractor and entity resolution."""

from __future__ import annotations

import pytest

from nanoindex.models import DocumentGraph, Entity, Relationship


# ---------------------------------------------------------------------------
# Domain detection tests
# ---------------------------------------------------------------------------

class TestDetectDomain:
    def test_financial_domain(self):
        from nanoindex.core.gliner_extractor import _detect_domain
        text = "The company reported revenue of $5B in fiscal 2024. Earnings per share (EPS) grew 12%. EBITDA margin improved."
        assert _detect_domain(text) == "financial"

    def test_legal_domain(self):
        from nanoindex.core.gliner_extractor import _detect_domain
        text = "The plaintiff filed a motion in court against the defendant. The statute of limitations applies. The jurisdiction covers the filing."
        assert _detect_domain(text) == "legal"

    def test_medical_domain(self):
        from nanoindex.core.gliner_extractor import _detect_domain
        text = "The patient was admitted to the hospital with a diagnosis of pneumonia. Treatment included dosage adjustments. The symptom improved after clinical review."
        assert _detect_domain(text) == "medical"

    def test_generic_domain(self):
        from nanoindex.core.gliner_extractor import _detect_domain
        text = "This is a general document about various topics including technology and innovation."
        assert _detect_domain(text) == "generic"

    def test_insurance_domain(self):
        from nanoindex.core.gliner_extractor import _detect_domain
        text = "The policy covers the insured for premium payments. The claim was filed with a deductible of $500. Coverage type is comprehensive."
        assert _detect_domain(text) == "insurance"


# ---------------------------------------------------------------------------
# Entity resolution tests
# ---------------------------------------------------------------------------

class TestEntityResolution:
    def test_exact_match(self):
        """'Google' and 'google' (case-insensitive) should merge."""
        from nanoindex.core.entity_resolver import resolve_entities

        graph = DocumentGraph(
            doc_name="test",
            entities=[
                Entity(name="Google", entity_type="Organization", source_node_ids=["n1"]),
                Entity(name="google", entity_type="Organization", source_node_ids=["n2"]),
            ],
            relationships=[],
        )
        result = resolve_entities(graph)
        assert len(result.entities) == 1
        # Merged entity should have both node ids
        assert set(result.entities[0].source_node_ids) == {"n1", "n2"}

    def test_suffix_match(self):
        """'Google Inc.' and 'Google' should merge."""
        from nanoindex.core.entity_resolver import resolve_entities

        graph = DocumentGraph(
            doc_name="test",
            entities=[
                Entity(name="Google", entity_type="Organization", source_node_ids=["n1", "n2"]),
                Entity(name="Google Inc.", entity_type="Organization", source_node_ids=["n3"]),
            ],
            relationships=[],
        )
        result = resolve_entities(graph)
        assert len(result.entities) == 1
        assert set(result.entities[0].source_node_ids) == {"n1", "n2", "n3"}

    def test_levenshtein_match(self):
        """'Gogle' and 'Google' should merge (edit distance 1)."""
        from nanoindex.core.entity_resolver import resolve_entities

        graph = DocumentGraph(
            doc_name="test",
            entities=[
                Entity(name="Google", entity_type="Organization", source_node_ids=["n1", "n2"]),
                Entity(name="Gogle", entity_type="Organization", source_node_ids=["n3"]),
            ],
            relationships=[],
        )
        result = resolve_entities(graph)
        assert len(result.entities) == 1
        assert set(result.entities[0].source_node_ids) == {"n1", "n2", "n3"}

    def test_preserves_unique(self):
        """Distinct entities should not be merged."""
        from nanoindex.core.entity_resolver import resolve_entities

        graph = DocumentGraph(
            doc_name="test",
            entities=[
                Entity(name="Google", entity_type="Organization", source_node_ids=["n1"]),
                Entity(name="Microsoft", entity_type="Organization", source_node_ids=["n2"]),
                Entity(name="New York", entity_type="Location", source_node_ids=["n3"]),
            ],
            relationships=[],
        )
        result = resolve_entities(graph)
        assert len(result.entities) == 3
        names = {e.name for e in result.entities}
        assert names == {"Google", "Microsoft", "New York"}

    def test_relationship_dedup(self):
        """Duplicate relationships should be removed after entity resolution."""
        from nanoindex.core.entity_resolver import resolve_entities

        graph = DocumentGraph(
            doc_name="test",
            entities=[
                Entity(name="Google", entity_type="Organization", source_node_ids=["n1", "n2"]),
                Entity(name="Google Inc.", entity_type="Organization", source_node_ids=["n3"]),
                Entity(name="Alphabet", entity_type="Organization", source_node_ids=["n4"]),
            ],
            relationships=[
                Relationship(source="Google", target="Alphabet", keywords="subsidiary of", source_node_ids=["n1"]),
                Relationship(source="Google Inc.", target="Alphabet", keywords="subsidiary of", source_node_ids=["n3"]),
            ],
        )
        result = resolve_entities(graph)
        # Both relationships should map to the same canonical entities and dedup
        assert len(result.relationships) == 1
        assert result.relationships[0].keywords == "subsidiary of"

    def test_self_referencing_relationships_removed(self):
        """Relationships where source == target after resolution should be removed."""
        from nanoindex.core.entity_resolver import resolve_entities

        graph = DocumentGraph(
            doc_name="test",
            entities=[
                Entity(name="Google", entity_type="Organization", source_node_ids=["n1", "n2"]),
                Entity(name="Google Inc.", entity_type="Organization", source_node_ids=["n3"]),
            ],
            relationships=[
                Relationship(source="Google", target="Google Inc.", keywords="also known as", source_node_ids=["n1"]),
            ],
        )
        result = resolve_entities(graph)
        # After merging, source == target, so relationship should be removed
        assert len(result.relationships) == 0


# ---------------------------------------------------------------------------
# Levenshtein helper tests
# ---------------------------------------------------------------------------

class TestLevenshtein:
    def test_identical(self):
        from nanoindex.core.entity_resolver import _levenshtein
        assert _levenshtein("hello", "hello") == 0

    def test_single_edit(self):
        from nanoindex.core.entity_resolver import _levenshtein
        assert _levenshtein("hello", "helo") == 1

    def test_empty(self):
        from nanoindex.core.entity_resolver import _levenshtein
        assert _levenshtein("", "abc") == 3
        assert _levenshtein("abc", "") == 3


# ---------------------------------------------------------------------------
# GLiNER2 integration tests (skip if gliner2 not installed)
# ---------------------------------------------------------------------------

def _gliner_available() -> bool:
    try:
        import gliner2  # noqa: F401
        return True
    except ImportError:
        return False


class TestGLiNERIntegration:
    def test_gliner_import(self):
        """Verify the extractor module can be imported."""
        from nanoindex.core.gliner_extractor import extract_entities_gliner, DOMAIN_LABELS
        assert "generic" in DOMAIN_LABELS
        assert "financial" in DOMAIN_LABELS

    @pytest.mark.skipif(
        not _gliner_available(),
        reason="gliner package not installed",
    )
    def test_gliner_extraction(self):
        """End-to-end GLiNER extraction (requires gliner package)."""
        from nanoindex.core.gliner_extractor import extract_entities_gliner
        from nanoindex.models import DocumentTree, TreeNode

        tree = DocumentTree(
            doc_name="test",
            structure=[
                TreeNode(
                    title="Section 1",
                    node_id="n1",
                    text="Apple Inc. reported revenue of $394 billion in fiscal year 2023. "
                         "CEO Tim Cook announced new product lines during the earnings call.",
                ),
            ],
        )
        graph = extract_entities_gliner(tree)
        assert isinstance(graph, DocumentGraph)
        # Should find at least some entities
        assert len(graph.entities) > 0
