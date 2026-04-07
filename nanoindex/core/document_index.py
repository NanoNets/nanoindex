"""DocumentIndex — unified navigation layer merging tree + graph.

The tree gives structure (hierarchy, page ranges, text).
The graph gives semantics (entities, relationships, communities).
The DocumentIndex bridges them:

  - node_id → entities (what entities live in this section?)
  - entity → node_ids (where in the document is this entity discussed?)
  - entity → related entities → their node_ids (follow relationships to jump sections)
  - query → entity match → seed nodes → graph expand → ranked candidates

This is the single object agents use to navigate a document.
"""

from __future__ import annotations

import logging
from typing import Any

from nanoindex.models import (
    DocumentGraph,
    DocumentTree,
    Entity,
    Relationship,
    TreeNode,
)
from nanoindex.utils.tree_ops import iter_nodes

logger = logging.getLogger(__name__)

try:
    import networkx as nx
except ImportError:
    nx = None  # type: ignore[assignment]


class DocumentIndex:
    """Unified document navigation: tree structure + entity graph.

    Usage::

        idx = DocumentIndex(tree, graph)

        # What entities are in this node?
        entities = idx.entities_in_node("0001.0024.0020")
        # → [Entity(name="Revenue", type="Revenue"), Entity(name="3M", type="Company")]

        # Where is "Revenue" discussed?
        nodes = idx.nodes_for_entity("Revenue")
        # → {"0001.0024.0020", "0001.0004.0003"}

        # Follow relationships from "Revenue" to find related sections
        related = idx.related_nodes("Revenue", hops=1)
        # → {"0001.0024.0020", "0001.0004.0003", "0001.0024.0037"} (via 3M, Segments, etc.)

        # Find nodes relevant to a query (entity match + graph expansion)
        candidates = idx.query_nodes("What was 3M's revenue in FY2018?")
        # → ranked list of (node_id, score) tuples
    """

    def __init__(self, tree: DocumentTree, graph: DocumentGraph | None = None):
        self.tree = tree
        self.graph = graph or DocumentGraph(doc_name=tree.doc_name)

        # Build indices
        self._node_map: dict[str, TreeNode] = {}
        self._node_to_entities: dict[str, list[Entity]] = {}
        self._entity_to_nodes: dict[str, set[str]] = {}
        self._entity_by_name: dict[str, Entity] = {}
        self._nx_graph: Any = None

        self._build_indices()

    def _build_indices(self):
        """Build bidirectional node ↔ entity indices."""
        # Node map
        for node in iter_nodes(self.tree.structure):
            self._node_map[node.node_id] = node

        # Entity → nodes (forward index)
        for ent in self.graph.entities:
            key = ent.name.lower()
            self._entity_by_name[key] = ent
            self._entity_to_nodes.setdefault(key, set()).update(ent.source_node_ids)

        # Nodes → entities (reverse index)
        for ent in self.graph.entities:
            for nid in ent.source_node_ids:
                self._node_to_entities.setdefault(nid, []).append(ent)

        # NetworkX graph (lazy)
        if nx and self.graph.entities:
            self._nx_graph = self._build_nx()

        logger.info(
            "DocumentIndex built: %d nodes, %d entities, %d relationships, %d node↔entity links",
            len(self._node_map),
            len(self._entity_by_name),
            len(self.graph.relationships),
            sum(len(ents) for ents in self._node_to_entities.values()),
        )

    def _build_nx(self) -> Any:
        G = nx.Graph()
        for ent in self.graph.entities:
            G.add_node(ent.name, entity_type=ent.entity_type, source_node_ids=set(ent.source_node_ids))
        for rel in self.graph.relationships:
            if not G.has_node(rel.source):
                G.add_node(rel.source, entity_type="Other", source_node_ids=set())
            if not G.has_node(rel.target):
                G.add_node(rel.target, entity_type="Other", source_node_ids=set())
            G.add_edge(rel.source, rel.target, keywords=rel.keywords)
        return G

    # ------------------------------------------------------------------
    # Lookup methods
    # ------------------------------------------------------------------

    def entities_in_node(self, node_id: str) -> list[Entity]:
        """What entities appear in this tree node?"""
        return self._node_to_entities.get(node_id, [])

    def nodes_for_entity(self, entity_name: str) -> set[str]:
        """Which tree nodes mention this entity?"""
        return self._entity_to_nodes.get(entity_name.lower(), set())

    def get_entity(self, name: str) -> Entity | None:
        """Look up an entity by name."""
        return self._entity_by_name.get(name.lower())

    def related_entities(self, entity_name: str, hops: int = 1) -> list[Entity]:
        """Find entities connected to this one via graph relationships."""
        if not self._nx_graph or not self._nx_graph.has_node(entity_name):
            # Try case-insensitive lookup
            for node_name in self._nx_graph.nodes() if self._nx_graph else []:
                if node_name.lower() == entity_name.lower():
                    entity_name = node_name
                    break
            else:
                return []

        visited = {entity_name}
        frontier = {entity_name}
        for _ in range(hops):
            next_frontier = set()
            for ent in frontier:
                if self._nx_graph.has_node(ent):
                    for neighbor in self._nx_graph.neighbors(ent):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            next_frontier.add(neighbor)
            frontier = next_frontier

        result = []
        for name in visited - {entity_name}:
            ent = self._entity_by_name.get(name.lower())
            if ent:
                result.append(ent)
        return result

    def related_nodes(self, entity_name: str, hops: int = 1) -> set[str]:
        """Follow entity relationships to find tree nodes in related sections."""
        related = self.related_entities(entity_name, hops=hops)
        node_ids = set()
        for ent in related:
            node_ids.update(ent.source_node_ids)
        # Also include direct nodes
        node_ids.update(self.nodes_for_entity(entity_name))
        return node_ids

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def query_nodes(
        self,
        query: str,
        *,
        decomposition: dict | None = None,
        max_results: int = 30,
        hops: int = 2,
    ) -> list[tuple[str, float]]:
        """Find tree nodes relevant to a query using entity matching + graph expansion.

        Returns list of (node_id, score) sorted by relevance.
        Score is based on: direct entity match (1.0) > graph-expanded (0.5) > keyword (0.25).
        """
        query_lower = query.lower()
        scores: dict[str, float] = {}

        # Add decomposition data points to search
        search_texts = [query_lower]
        if decomposition:
            for dp in decomposition.get("data_points", []):
                search_texts.append(dp.lower())

        # Direct entity match — entities whose names appear in the query
        matched_entities: set[str] = set()
        for entity_name in self._entity_to_nodes:
            if len(entity_name) >= 3:
                for text in search_texts:
                    if entity_name in text:
                        matched_entities.add(entity_name)
                        for nid in self._entity_to_nodes[entity_name]:
                            scores[nid] = max(scores.get(nid, 0), 1.0)
                        break

        # Graph expansion — follow relationships from matched entities
        if matched_entities and self._nx_graph:
            for entity_name in list(matched_entities):
                related = self.related_entities(entity_name, hops=hops)
                for ent in related:
                    for nid in ent.source_node_ids:
                        if nid not in scores:
                            scores[nid] = max(scores.get(nid, 0), 0.5)

        # Keyword fallback — scan node titles/summaries
        if not scores:
            keywords = [w for w in query_lower.split() if len(w) > 3]
            for node in iter_nodes(self.tree.structure):
                title_lower = (node.title or "").lower()
                summary_lower = (node.summary or "").lower()
                hits = sum(1 for kw in keywords if kw in title_lower or kw in summary_lower)
                if hits > 0:
                    scores[node.node_id] = 0.25 * hits / max(len(keywords), 1)

        # Sort by score, limit results
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:max_results]

    def node_context(self, node_id: str) -> str:
        """Get a rich context string for a node: title + entities + relationships."""
        node = self._node_map.get(node_id)
        if not node:
            return ""

        parts = [f"[{node_id}] {node.title}"]
        if node.summary:
            parts.append(f"  Summary: {node.summary[:200]}")
        if node.start_index:
            parts.append(f"  Pages: {node.start_index}-{node.end_index}")

        entities = self.entities_in_node(node_id)
        if entities:
            ent_strs = [f"{e.name} ({e.entity_type})" for e in entities[:10]]
            parts.append(f"  Entities: {', '.join(ent_strs)}")

        return "\n".join(parts)

    def candidate_outline(self, node_ids: set[str]) -> str:
        """Build a rich outline of candidate nodes with entity annotations."""
        lines = []
        for node in iter_nodes(self.tree.structure):
            if node.node_id in node_ids:
                lines.append(self.node_context(node.node_id))
        return "\n\n".join(lines)
