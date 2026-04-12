"""Extract entities and relationships from tree nodes.

Runs at index time after summaries are generated. Uses an LLM to identify
named entities and their relationships within each tree node, then merges
duplicates across nodes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from nanoindex.core.llm import LLMClient
from nanoindex.models import (
    DocumentGraph,
    DocumentTree,
    Entity,
    ModalContent,
    ParsedDocument,
    Relationship,
)
from nanoindex.utils.tree_ops import iter_nodes

logger = logging.getLogger(__name__)

_MAX_CONCURRENT = 3

_ENTITY_TYPES = (
    "Organization, Person, Metric, TimePeriod, Location, Document, "
    "Concept, Product, Event, FinancialItem, LegalTerm, Other"
)

_EXTRACT_PROMPT = """\
Extract entities and relationships from this document section.

ENTITY TYPES: {entity_types}

For each entity output one line:
ENTITY|name|type|short description

For each relationship output one line:
REL|source_entity|target_entity|keywords|short description

Rules:
- Use title case for entity names (e.g., "Total Revenue", "3M Company")
- Merge obvious duplicates (e.g., "Revenue" and "Total Revenue" → "Total Revenue")
- Only extract entities clearly mentioned in the text
- Keep descriptions under 20 words
- Output DONE when finished

Section title: {title}
Section summary: {summary}
Section content (first 5000 chars):
{content}

Extract:"""


async def extract_entities(
    tree: DocumentTree,
    llm: LLMClient,
) -> DocumentGraph:
    """Extract entities and relationships from all tree nodes."""
    all_nodes = list(iter_nodes(tree.structure))
    sem = asyncio.Semaphore(_MAX_CONCURRENT)

    raw_entities: list[tuple[str, str, str, str]] = []  # (name, type, desc, node_id)
    raw_relationships: list[tuple[str, str, str, str, str]] = []  # (src, tgt, kw, desc, node_id)

    async def _extract_node(node):
        content = node.text or ""
        if len(content) < 100:
            return

        prompt = _EXTRACT_PROMPT.format(
            entity_types=_ENTITY_TYPES,
            title=node.title,
            summary=node.summary or "",
            content=content[:5000],
        )

        async with sem:
            try:
                resp = await llm.chat(
                    [{"role": "user", "content": prompt}],
                    max_tokens=1024,
                    temperature=0.0,
                )
                _parse_response(resp, node.node_id, raw_entities, raw_relationships)
            except Exception:
                logger.warning("Entity extraction failed for node '%s'", node.title, exc_info=True)

    logger.info("Extracting entities from %d nodes", len(all_nodes))
    await asyncio.gather(*[_extract_node(n) for n in all_nodes])

    # Merge duplicates
    entities = _merge_entities(raw_entities)
    relationships = _merge_relationships(raw_relationships)

    logger.info("Extracted %d entities, %d relationships", len(entities), len(relationships))
    return DocumentGraph(
        doc_name=tree.doc_name,
        entities=entities,
        relationships=relationships,
    )


def _parse_response(
    text: str,
    node_id: str,
    entities: list[tuple[str, str, str, str]],
    relationships: list[tuple[str, str, str, str, str]],
) -> None:
    """Parse delimiter-based entity/relationship extraction output."""
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.upper() == "DONE":
            continue

        parts = [p.strip() for p in line.split("|")]

        if parts[0].upper() == "ENTITY" and len(parts) >= 4:
            name, etype, desc = parts[1], parts[2], parts[3]
            entities.append((name, etype, desc, node_id))
        elif parts[0].upper() == "REL" and len(parts) >= 5:
            src, tgt, kw, desc = parts[1], parts[2], parts[3], parts[4]
            relationships.append((src, tgt, kw, desc, node_id))


def _normalize_name(name: str) -> str:
    """Normalize entity name for deduplication."""
    return re.sub(r"\s+", " ", name.strip().lower())


def _merge_entities(raw: list[tuple[str, str, str, str]]) -> list[Entity]:
    """Merge duplicate entities across nodes."""
    merged: dict[str, Entity] = {}

    for name, etype, desc, node_id in raw:
        key = _normalize_name(name)
        if key in merged:
            ent = merged[key]
            if node_id not in ent.source_node_ids:
                ent.source_node_ids.append(node_id)
            # Keep longer description
            if len(desc) > len(ent.description):
                ent.description = desc
        else:
            merged[key] = Entity(
                name=name.strip(),
                entity_type=etype.strip(),
                description=desc.strip(),
                source_node_ids=[node_id],
            )

    return list(merged.values())


def _merge_relationships(raw: list[tuple[str, str, str, str, str]]) -> list[Relationship]:
    """Merge duplicate relationships across nodes."""
    merged: dict[str, Relationship] = {}

    for src, tgt, kw, desc, node_id in raw:
        # Normalize key (undirected)
        k1, k2 = _normalize_name(src), _normalize_name(tgt)
        key = (min(k1, k2), max(k1, k2))

        if key in merged:
            rel = merged[key]
            if node_id not in rel.source_node_ids:
                rel.source_node_ids.append(node_id)
            if len(desc) > len(rel.description):
                rel.description = desc
        else:
            merged[key] = Relationship(
                source=src.strip(),
                target=tgt.strip(),
                keywords=kw.strip(),
                description=desc.strip(),
                source_node_ids=[node_id],
            )

    return list(merged.values())


# ---------------------------------------------------------------------------
# Multimodal entity extraction
# ---------------------------------------------------------------------------


async def extract_multimodal_entities(
    parsed: ParsedDocument,
    tree: DocumentTree,
    llm: LLMClient,
) -> tuple[list[Entity], list[Relationship]]:
    """Extract entities from non-text content (images, tables, etc.).

    Iterates over ``parsed.modal_contents``, dispatches each item to its
    modal processor, and collects the resulting entities and relationships.
    """
    from nanoindex.core.modal_processors import get_processor

    if not parsed.modal_contents:
        return [], []

    # Build a page -> node_id mapping from the tree.
    # Each tree node covers pages [start_index, end_index]; we map each page
    # to the *deepest* (most specific) node that contains it.
    page_to_node: dict[int, str] = {}
    for node in iter_nodes(tree.structure):
        if node.start_index and node.end_index:
            for pg in range(node.start_index, node.end_index + 1):
                # Later (deeper) nodes overwrite earlier ones — that's fine,
                # iter_nodes is depth-first so children come after parents.
                page_to_node[pg] = node.node_id

    entities: list[Entity] = []
    relationships: list[Relationship] = []
    sem = asyncio.Semaphore(_MAX_CONCURRENT)

    async def _process_item(item: ModalContent) -> None:
        processor = get_processor(item.content_type)
        if processor is None:
            logger.debug("No processor for content_type=%r, skipping", item.content_type)
            return

        parent_node_id = page_to_node.get(item.page, "root")

        async with sem:
            try:
                result = await processor.process(item, llm, parent_node_id)
            except Exception:
                logger.warning(
                    "Modal processing failed for %s on page %d",
                    item.content_type,
                    item.page,
                    exc_info=True,
                )
                return

        if result is not None:
            entities.append(result.entity)
            relationships.extend(result.relationships)

    await asyncio.gather(*[_process_item(item) for item in parsed.modal_contents])

    logger.info(
        "Multimodal extraction: %d entities, %d relationships from %d items",
        len(entities),
        len(relationships),
        len(parsed.modal_contents),
    )
    return entities, relationships


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_graph(graph: DocumentGraph, path) -> None:
    """Save graph to JSON file."""
    from pathlib import Path

    with open(Path(path), "w") as f:
        json.dump(graph.model_dump(), f, ensure_ascii=False, indent=2)
    logger.info(
        "Saved graph (%d entities, %d rels) to %s",
        len(graph.entities),
        len(graph.relationships),
        path,
    )


def load_graph(path) -> DocumentGraph:
    """Load graph from JSON file."""
    from pathlib import Path

    with open(Path(path)) as f:
        return DocumentGraph.model_validate(json.load(f))
