"""Tests for entity extractor and multimodal entity integration."""

from __future__ import annotations

import asyncio

import pytest

from nanoindex.models import (
    DocumentGraph,
    DocumentTree,
    Entity,
    ModalContent,
    ParsedDocument,
    Relationship,
    TreeNode,
)


# ------------------------------------------------------------------
# DocumentGraph model tests
# ------------------------------------------------------------------


def test_document_graph_empty():
    """An empty DocumentGraph can be created with no entities or relationships."""
    graph = DocumentGraph(doc_name="test")
    assert graph.doc_name == "test"
    assert graph.entities == []
    assert graph.relationships == []


def test_document_graph_with_entities_and_relationships():
    """DocumentGraph correctly holds entities and relationships."""
    entities = [
        Entity(name="Acme Corp", entity_type="Organization", description="A company"),
        Entity(name="Revenue", entity_type="FinancialItem", description="Total revenue"),
    ]
    relationships = [
        Relationship(
            source="Acme Corp",
            target="Revenue",
            keywords="reports",
            description="Acme Corp reports Revenue",
        ),
    ]
    graph = DocumentGraph(
        doc_name="report",
        entities=entities,
        relationships=relationships,
    )
    assert len(graph.entities) == 2
    assert len(graph.relationships) == 1
    assert graph.entities[0].name == "Acme Corp"
    assert graph.relationships[0].source == "Acme Corp"


def test_document_graph_extend():
    """Entities and relationships lists can be extended after creation."""
    graph = DocumentGraph(doc_name="test")
    graph.entities.append(
        Entity(name="Image1", entity_type="Image", description="A chart")
    )
    graph.relationships.append(
        Relationship(source="Image1", target="0001", keywords="belongs_to")
    )
    assert len(graph.entities) == 1
    assert len(graph.relationships) == 1


# ------------------------------------------------------------------
# Import smoke test for extract_multimodal_entities
# ------------------------------------------------------------------


def test_extract_multimodal_entities_importable():
    """extract_multimodal_entities can be imported from entity_extractor."""
    from nanoindex.core.entity_extractor import extract_multimodal_entities

    assert callable(extract_multimodal_entities)


def test_get_processor_importable():
    """get_processor can be imported from modal_processors."""
    from nanoindex.core.modal_processors import get_processor

    assert callable(get_processor)
    # image processor should be registered
    proc = get_processor("image")
    assert proc is not None
    assert proc.content_type == "image"
    # unknown type returns None
    assert get_processor("unknown_type") is None


# ------------------------------------------------------------------
# extract_multimodal_entities with no modal content
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_multimodal_entities_empty():
    """Returns empty lists when parsed has no modal_contents."""
    from nanoindex.core.entity_extractor import extract_multimodal_entities

    parsed = ParsedDocument(markdown="hello", modal_contents=[])
    tree = DocumentTree(
        doc_name="test",
        structure=[TreeNode(title="Root", node_id="0001", start_index=1, end_index=1)],
    )

    # LLM should not be called, so passing None is fine
    entities, rels = await extract_multimodal_entities(parsed, tree, None)  # type: ignore[arg-type]
    assert entities == []
    assert rels == []
