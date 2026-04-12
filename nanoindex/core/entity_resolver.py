"""Fuzzy entity resolution to merge duplicate entities."""

from __future__ import annotations

import logging

from nanoindex.models import DocumentGraph, Entity, Relationship

logger = logging.getLogger(__name__)


def resolve_entities(graph: DocumentGraph) -> DocumentGraph:
    """Merge entities that refer to the same real-world thing."""
    entities = list(graph.entities)
    merged: dict[str, Entity] = {}  # canonical_key -> Entity
    key_map: dict[str, str] = {}  # original_key -> canonical_key

    for entity in sorted(entities, key=lambda e: len(e.source_node_ids), reverse=True):
        name_lower = entity.name.lower().strip()

        # Check if this matches any existing canonical entity
        canonical = _find_match(name_lower, merged)

        if canonical:
            # Merge into existing
            key_map[name_lower] = canonical
            existing = merged[canonical]
            existing.source_node_ids = sorted(
                set(existing.source_node_ids) | set(entity.source_node_ids)
            )
            if len(entity.description) > len(existing.description):
                existing.description = entity.description
        else:
            # New canonical entity
            merged[name_lower] = entity
            key_map[name_lower] = name_lower

    # Remap relationships
    new_rels: list[Relationship] = []
    seen_rels: set[tuple[str, str, str]] = set()
    for rel in graph.relationships:
        src_key = key_map.get(rel.source.lower(), rel.source.lower())
        tgt_key = key_map.get(rel.target.lower(), rel.target.lower())
        if src_key in merged and tgt_key in merged:
            new_src = merged[src_key].name
            new_tgt = merged[tgt_key].name
            rel_key = (new_src, new_tgt, rel.keywords)
            if rel_key not in seen_rels and new_src != new_tgt:
                seen_rels.add(rel_key)
                new_rels.append(
                    Relationship(
                        source=new_src,
                        target=new_tgt,
                        keywords=rel.keywords,
                        source_node_ids=rel.source_node_ids,
                    )
                )

    logger.info(
        "Entity resolution: %d -> %d entities, %d -> %d relationships",
        len(entities),
        len(merged),
        len(graph.relationships),
        len(new_rels),
    )

    return DocumentGraph(
        doc_name=graph.doc_name, entities=list(merged.values()), relationships=new_rels
    )


def _find_match(name: str, existing: dict[str, Entity]) -> str | None:
    """Find if name matches any existing entity via fuzzy matching."""
    # Exact match
    if name in existing:
        return name

    # Strip common suffixes
    for suffix in [" inc", " inc.", " corp", " corp.", " llc", " ltd", " co", " co."]:
        stripped = name.rstrip(".").removesuffix(suffix)
        if stripped != name and stripped in existing:
            return stripped
        # Also check the reverse
        for key in existing:
            if key.rstrip(".").removesuffix(suffix) == stripped and stripped:
                return key

    # Substring containment (shorter is canonical if it's a real entity name)
    for key in existing:
        if len(key) >= 3 and len(name) >= 3:
            if key in name and len(key) >= len(name) * 0.5:
                return key
            if name in key and len(name) >= len(key) * 0.5:
                return key

    # Levenshtein distance for typos (only for short names)
    if len(name) <= 20:
        for key in existing:
            if len(key) <= 20 and _levenshtein(name, key) <= 2:
                return key

    return None


def _levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]
