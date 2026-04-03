"""Build and query a NetworkX graph from extracted entities and relationships.

The graph connects entities as nodes and relationships as edges.
Each entity node stores which tree node_ids it appears in, enabling
efficient entity→tree_node lookup for retrieval.
"""

from __future__ import annotations

import logging
from typing import Any

from nanoindex.models import DocumentGraph

logger = logging.getLogger(__name__)

try:
    import networkx as nx
except ImportError:
    nx = None  # type: ignore[assignment]


def _require_nx():
    if nx is None:
        raise ImportError("pip install networkx — required for graph mode")


def build_nx_graph(graph_data: DocumentGraph) -> Any:
    """Build a NetworkX graph from extracted entities and relationships.

    Returns a networkx.Graph with:
      - Nodes: entity names (attributes: type, description, source_node_ids)
      - Edges: relationships (attributes: keywords, description, source_node_ids)
    """
    _require_nx()
    G = nx.Graph()

    for ent in graph_data.entities:
        G.add_node(
            ent.name,
            entity_type=ent.entity_type,
            description=ent.description,
            source_node_ids=set(ent.source_node_ids),
        )

    for rel in graph_data.relationships:
        # Ensure both endpoints exist
        if not G.has_node(rel.source):
            G.add_node(rel.source, entity_type="Other", description="", source_node_ids=set())
        if not G.has_node(rel.target):
            G.add_node(rel.target, entity_type="Other", description="", source_node_ids=set())

        G.add_edge(
            rel.source,
            rel.target,
            keywords=rel.keywords,
            description=rel.description,
            source_node_ids=set(rel.source_node_ids),
        )

    logger.info("Built graph: %d entity nodes, %d relationship edges", G.number_of_nodes(), G.number_of_edges())
    return G


def build_entity_to_nodes(graph_data: DocumentGraph) -> dict[str, set[str]]:
    """Build inverted index: entity name (lowercased) → set of tree node_ids."""
    index: dict[str, set[str]] = {}
    for ent in graph_data.entities:
        key = ent.name.lower()
        index.setdefault(key, set()).update(ent.source_node_ids)
    return index


def entity_keyword_match(query: str, entity_to_nodes: dict[str, set[str]]) -> set[str]:
    """Find tree nodes matching entity names in the query.

    Simple substring matching — fast, zero LLM cost.
    """
    query_lower = query.lower()
    matched_node_ids: set[str] = set()

    for entity_name, node_ids in entity_to_nodes.items():
        # Check if entity name appears in query (minimum 3 chars to avoid noise)
        if len(entity_name) >= 3 and entity_name in query_lower:
            matched_node_ids.update(node_ids)

    return matched_node_ids


def graph_expand(
    nx_graph: Any,
    seed_node_ids: set[str],
    entity_to_nodes: dict[str, set[str]],
    hops: int = 1,
) -> set[str]:
    """Expand seed tree nodes via entity graph traversal.

    1. Find entities mentioned in seed tree nodes
    2. Traverse graph edges to find neighbor entities
    3. Map neighbor entities back to their tree node_ids
    4. Return expanded set of tree node_ids
    """
    _require_nx()

    if not nx_graph or nx_graph.number_of_nodes() == 0:
        return set()

    # Step 1: Find entities in seed nodes
    seed_entities: set[str] = set()
    for entity_name, data in nx_graph.nodes(data=True):
        node_ids = data.get("source_node_ids", set())
        if node_ids & seed_node_ids:
            seed_entities.add(entity_name)

    if not seed_entities:
        return set()

    # Step 2: BFS expansion on entity graph
    expanded_entities: set[str] = set(seed_entities)
    frontier = set(seed_entities)
    for _ in range(hops):
        next_frontier: set[str] = set()
        for ent in frontier:
            if nx_graph.has_node(ent):
                for neighbor in nx_graph.neighbors(ent):
                    if neighbor not in expanded_entities:
                        next_frontier.add(neighbor)
                        expanded_entities.add(neighbor)
        frontier = next_frontier

    # Step 3: Map expanded entities back to tree node_ids
    expanded_node_ids: set[str] = set()
    for entity_name in expanded_entities:
        key = entity_name.lower()
        if key in entity_to_nodes:
            expanded_node_ids.update(entity_to_nodes[key])

    new_nodes = expanded_node_ids - seed_node_ids
    if new_nodes:
        logger.info("Graph expansion: %d seed entities → %d expanded → %d new tree nodes",
                    len(seed_entities), len(expanded_entities), len(new_nodes))

    return expanded_node_ids
