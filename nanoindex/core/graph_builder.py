"""Build and query a NetworkX graph from extracted entities and relationships.

The graph connects entities as nodes and relationships as edges.
Each entity node stores which tree node_ids it appears in, enabling
efficient entity->tree_node lookup for retrieval.

Two build paths:
  1. **From API entities** (``build_graph_from_hierarchy``) - uses entities
     and relationships extracted by the hierarchy API during indexing.
     Fast (milliseconds), high quality (Gemini-extracted semantic relations).
  2. **From local NER** (GLiNER/spaCy in ``entity_extractor.py``) - runs
     NER locally on every node's text.  Slower, used as fallback when
     hierarchy API entities are not available.
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Any

from nanoindex.models import (
    DocumentGraph,
    Entity,
    HierarchySection,
    Relationship,
)

logger = logging.getLogger(__name__)

try:
    import networkx as nx
except ImportError:
    nx = None  # type: ignore[assignment]


def _require_nx():
    if nx is None:
        raise ImportError("pip install networkx — required for graph mode")


# ------------------------------------------------------------------
# Relationship type normalization
# ------------------------------------------------------------------

_REL_SYNONYMS: dict[str, str] = {
    "incorporated_in": "operates_in",
    "incorporated in": "operates_in",
    "located_at": "located_in",
    "located at": "located_in",
    "is_title": "holds_title",
    "has_title": "holds_title",
    "is_president_of": "holds_title",
    "is_vice_president_cfo_and_treasurer_of": "holds_title",
    "representative_of": "represents",
    "submits to jurisdiction of": "submits_to_jurisdiction",
    "operates as": "acts_as",
    "serves_as_trustee_for": "acts_as",
    "formerly known as": "formerly_known_as",
    "dated as of": "in_period",
    "title": "holds_title",
    "is_officer_of": "works_for",
    "employed_by": "works_for",
    "is_cfo_of": "works_for",
    "is_vice_president_of": "works_for",
}

_DROP_REL_TYPES = {"?", "related_to", ""}


def _normalize_rel_type(raw: str) -> str | None:
    """Normalize a relationship type string. Returns None if it should be dropped."""
    cleaned = raw.strip().lower().replace(" ", "_")
    if cleaned in _DROP_REL_TYPES:
        return None
    return _REL_SYNONYMS.get(raw.strip().lower(), cleaned)


def _fuzzy_match(a: str, b: str) -> bool:
    """Check if two entity names are near-duplicates (e.g. 'PepsiCo' vs 'PepsiCo, Inc.')."""
    a_clean = re.sub(r"[,.\s]+(Inc|Corp|LLC|Ltd|plc|Co)\.?$", "", a.strip(), flags=re.IGNORECASE)
    b_clean = re.sub(r"[,.\s]+(Inc|Corp|LLC|Ltd|plc|Co)\.?$", "", b.strip(), flags=re.IGNORECASE)
    if a_clean.lower() == b_clean.lower():
        return True
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() > 0.9


# ------------------------------------------------------------------
# Build graph from hierarchy API entities (fast path)
# ------------------------------------------------------------------


def build_graph_from_hierarchy(
    sections: list[HierarchySection],
    doc_name: str,
) -> DocumentGraph:
    """Build an entity graph directly from API-extracted entities and relationships.

    Walks all sections recursively, merges entities by name (with fuzzy
    dedup for suffixes like ', Inc.'), normalizes relationship types,
    and drops noise relationships ('related_to', '?').

    Returns a ``DocumentGraph`` ready for graph search and retrieval.
    """
    raw_entities: dict[str, Entity] = {}  # lowercase name -> Entity
    raw_relationships: list[Relationship] = []
    # For fuzzy dedup: canonical name lookup
    canonical: dict[str, str] = {}  # lowercase_variant -> canonical_name

    def _canonical_name(name: str) -> str:
        """Find or register the canonical form of an entity name."""
        key = name.strip().lower()
        if key in canonical:
            return canonical[key]
        # Check fuzzy match against existing names
        for existing_key, existing_canonical in canonical.items():
            if _fuzzy_match(name, existing_canonical):
                canonical[key] = existing_canonical
                return existing_canonical
        # New entity
        canonical[key] = name.strip()
        return name.strip()

    def _walk(secs: list[HierarchySection], parent_node_id: str = "") -> None:
        for sec in secs:
            node_id = sec.id or parent_node_id

            # Collect entities
            for e in sec.entities:
                canon = _canonical_name(e.name)
                key = canon.lower()
                if key in raw_entities:
                    if node_id and node_id not in raw_entities[key].source_node_ids:
                        raw_entities[key].source_node_ids.append(node_id)
                    # Keep longer description
                    if len(e.value) > len(raw_entities[key].description):
                        raw_entities[key].description = e.value
                else:
                    raw_entities[key] = Entity(
                        name=canon,
                        entity_type=e.entity_type,
                        description=e.value,
                        source_node_ids=[node_id] if node_id else [],
                    )

            # Collect relationships
            for r in sec.relationships:
                rel_type = _normalize_rel_type(r.rel_type)
                if rel_type is None:
                    continue  # drop noise
                src = _canonical_name(r.source)
                tgt = _canonical_name(r.target)
                if src and tgt and src.lower() != tgt.lower():
                    raw_relationships.append(
                        Relationship(
                            source=src,
                            target=tgt,
                            keywords=rel_type,
                            source_node_ids=[node_id] if node_id else [],
                        )
                    )

            _walk(sec.subsections, node_id)

    _walk(sections)

    # Deduplicate relationships (same source+target+type)
    seen_rels: dict[tuple[str, str, str], Relationship] = {}
    for r in raw_relationships:
        key = (r.source.lower(), r.target.lower(), r.keywords)
        if key in seen_rels:
            for nid in r.source_node_ids:
                if nid not in seen_rels[key].source_node_ids:
                    seen_rels[key].source_node_ids.append(nid)
        else:
            seen_rels[key] = r

    entities = list(raw_entities.values())
    relationships = list(seen_rels.values())

    logger.info(
        "Graph from API: %d entities, %d relationships (from %d sections)",
        len(entities),
        len(relationships),
        len(canonical),
    )
    return DocumentGraph(
        doc_name=doc_name,
        entities=entities,
        relationships=relationships,
    )


# ------------------------------------------------------------------
# Cross-reference resolution
# ------------------------------------------------------------------

# Patterns that match internal document references
_XREF_PATTERNS = [
    re.compile(r"\bSection\s+(\d+(?:\.\d+)*(?:\([a-z]\))?)", re.IGNORECASE),
    re.compile(r"\bClause\s+(\d+(?:\.\d+)*(?:\([a-z]\))?)", re.IGNORECASE),
    re.compile(r"\bArticle\s+(\d+(?:\.\d+)*)", re.IGNORECASE),
    re.compile(r"\bItem\s+(\d+[A-Z]?(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"\bNote\s+(\d+)", re.IGNORECASE),
    re.compile(r"\bExhibit\s+(\d+(?:\.\d+)*)", re.IGNORECASE),
    re.compile(r"\bSchedule\s+(\d+(?:\.\d+)*)", re.IGNORECASE),
    re.compile(r"\bPart\s+([IVX]+)", re.IGNORECASE),
]


def add_cross_references(
    graph: DocumentGraph,
    tree_nodes: list,
) -> DocumentGraph:
    """Scan tree node text for internal references and add graph edges.

    Detects patterns like "Section 3.1", "Clause 5(b)", "Article 19",
    "Item 1A", "Note 7", "Exhibit 4.6" and creates ``references``
    relationships between the node containing the reference and the
    node whose title matches the referenced section.

    Modifies *graph* in-place and returns it.
    """
    from nanoindex.utils.tree_ops import iter_nodes

    all_nodes = list(iter_nodes(tree_nodes))

    # Build lookup: normalized reference label -> node_id
    # e.g. "section 3.1" -> "0004", "item 1a" -> "0012"
    label_to_node: dict[str, str] = {}
    for node in all_nodes:
        title = (node.title or "").strip()
        for pattern in _XREF_PATTERNS:
            m = pattern.match(title)
            if m:
                label = pattern.pattern.split(r"\s+")[0].strip("\\b").lower()
                ref_id = f"{label} {m.group(1)}".lower()
                label_to_node[ref_id] = node.node_id
                break

    if not label_to_node:
        return graph

    # Scan all node text for references
    xref_count = 0
    seen: set[tuple[str, str]] = set()

    for node in all_nodes:
        text = node.text or ""
        if not text:
            continue

        for pattern in _XREF_PATTERNS:
            for m in pattern.finditer(text):
                label = pattern.pattern.split(r"\s+")[0].strip("\\b").lower()
                ref_id = f"{label} {m.group(1)}".lower()
                target_node_id = label_to_node.get(ref_id)

                if target_node_id and target_node_id != node.node_id:
                    edge_key = (node.node_id, target_node_id)
                    if edge_key not in seen:
                        seen.add(edge_key)
                        graph.relationships.append(
                            Relationship(
                                source=node.title,
                                target=next(
                                    (n.title for n in all_nodes if n.node_id == target_node_id),
                                    ref_id,
                                ),
                                keywords="references",
                                source_node_ids=[node.node_id, target_node_id],
                            )
                        )
                        xref_count += 1

    if xref_count:
        logger.info("Added %d cross-reference edges to graph", xref_count)

    return graph


# ------------------------------------------------------------------
# Build NetworkX graph from DocumentGraph
# ------------------------------------------------------------------


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

    logger.info(
        "Built graph: %d entity nodes, %d relationship edges",
        G.number_of_nodes(),
        G.number_of_edges(),
    )
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
        logger.info(
            "Graph expansion: %d seed entities → %d expanded → %d new tree nodes",
            len(seed_entities),
            len(expanded_entities),
            len(new_nodes),
        )

    return expanded_node_ids
