"""Tests for community detection, contradiction detection, and multi-hop expansion."""

from nanoindex.models import DocumentGraph, Entity, Relationship


def _make_tech_graph():
    """Create a graph with two clear clusters: Apple and Google."""
    entities = [
        Entity(name="Apple", entity_type="Company", description="Technology company", source_node_ids=["0001"]),
        Entity(name="Tim Cook", entity_type="Person", description="CEO of Apple", source_node_ids=["0001"]),
        Entity(name="iPhone", entity_type="Product", description="Smartphone by Apple", source_node_ids=["0001"]),
        Entity(name="Google", entity_type="Company", description="Search and tech company", source_node_ids=["0002"]),
        Entity(name="Sundar Pichai", entity_type="Person", description="CEO of Google", source_node_ids=["0002"]),
        Entity(name="Android", entity_type="Product", description="Mobile OS by Google", source_node_ids=["0002"]),
    ]
    relationships = [
        Relationship(source="Tim Cook", target="Apple", keywords="CEO of"),
        Relationship(source="iPhone", target="Apple", keywords="made by"),
        Relationship(source="Sundar Pichai", target="Google", keywords="CEO of"),
        Relationship(source="Android", target="Google", keywords="made by"),
    ]
    return DocumentGraph(doc_name="test", entities=entities, relationships=relationships)


def test_detect_communities():
    from nanoindex.core.community_detector import detect_communities

    graph = _make_tech_graph()
    communities = detect_communities(graph)
    assert len(communities) == 2  # Apple cluster and Google cluster

    # Each community should have 3 members
    sizes = sorted(len(c.entity_names) for c in communities)
    assert sizes == [3, 3]


def test_detect_communities_empty():
    from nanoindex.core.community_detector import detect_communities

    empty = DocumentGraph(doc_name="empty", entities=[], relationships=[])
    assert detect_communities(empty) == []


def test_detect_communities_no_relationships():
    from nanoindex.core.community_detector import detect_communities

    graph = DocumentGraph(
        doc_name="no_rels",
        entities=[Entity(name="A", entity_type="X"), Entity(name="B", entity_type="Y")],
        relationships=[],
    )
    assert detect_communities(graph) == []


def test_auto_summarize():
    from nanoindex.core.community_detector import detect_communities, auto_summarize_community

    graph = _make_tech_graph()
    communities = detect_communities(graph)
    assert len(communities) >= 1

    summary = auto_summarize_community(communities[0], graph)
    assert "Community:" in summary
    assert "Entities" in summary
    # Should mention at least one entity from the community
    assert any(name in summary for name in communities[0].entity_names)


def test_contradiction_detection():
    from nanoindex.core.contradiction_detector import find_contradictions

    g1 = DocumentGraph(
        doc_name="doc1",
        entities=[Entity(name="Revenue", entity_type="Metric", description="Revenue was $100M")],
    )
    g2 = DocumentGraph(
        doc_name="doc2",
        entities=[Entity(name="Revenue", entity_type="Metric", description="Revenue was $150M")],
    )

    results = find_contradictions({"doc1": g1, "doc2": g2})
    assert len(results) >= 1
    assert results[0]["entity"] == "revenue"
    assert results[0]["type"] == "numeric_discrepancy"


def test_contradiction_detection_no_conflict():
    from nanoindex.core.contradiction_detector import find_contradictions

    g1 = DocumentGraph(
        doc_name="doc1",
        entities=[Entity(name="CEO", entity_type="Person", description="John Smith is the CEO")],
    )
    g2 = DocumentGraph(
        doc_name="doc2",
        entities=[Entity(name="CEO", entity_type="Person", description="John Smith is the CEO")],
    )

    results = find_contradictions({"doc1": g1, "doc2": g2})
    assert len(results) == 0


def test_multi_hop_expansion():
    """Test 2-hop traversal finds indirect connections."""
    from nanoindex.core.graph_builder import build_nx_graph, build_entity_to_nodes, graph_expand

    # Chain: A -> B -> C -> D
    entities = [
        Entity(name="A", entity_type="X", source_node_ids=["n1"]),
        Entity(name="B", entity_type="X", source_node_ids=["n2"]),
        Entity(name="C", entity_type="X", source_node_ids=["n3"]),
        Entity(name="D", entity_type="X", source_node_ids=["n4"]),
    ]
    relationships = [
        Relationship(source="A", target="B", keywords="linked"),
        Relationship(source="B", target="C", keywords="linked"),
        Relationship(source="C", target="D", keywords="linked"),
    ]
    graph = DocumentGraph(doc_name="chain", entities=entities, relationships=relationships)

    nx_graph = build_nx_graph(graph)
    entity_to_nodes = build_entity_to_nodes(graph)

    # 1-hop from n1 (entity A) should reach B (n2)
    result_1hop = graph_expand(nx_graph, {"n1"}, entity_to_nodes, hops=1)
    assert "n1" in result_1hop
    assert "n2" in result_1hop
    assert "n3" not in result_1hop

    # 2-hop from n1 should reach C (n3)
    result_2hop = graph_expand(nx_graph, {"n1"}, entity_to_nodes, hops=2)
    assert "n3" in result_2hop
    assert "n4" not in result_2hop

    # 3-hop from n1 should reach D (n4)
    result_3hop = graph_expand(nx_graph, {"n1"}, entity_to_nodes, hops=3)
    assert "n4" in result_3hop
