"""Wiki compiler: produces Obsidian-compatible markdown files with [[backlinks]].

Takes document trees and entity graphs and generates interconnected markdown
pages for a knowledge base wiki.
"""

from __future__ import annotations

from pathlib import Path

from nanoindex.models import (
    Citation,
    DocumentGraph,
    DocumentTree,
    KBConfig,
    KBDocument,
    TreeNode,
)
from nanoindex.utils.slug import slugify
from nanoindex.utils.tree_ops import iter_nodes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _outline_md(nodes: list[TreeNode], indent: int = 0) -> list[str]:
    """Render a tree outline as markdown bullet list lines."""
    lines: list[str] = []
    prefix = "  " * indent
    for node in nodes:
        page_info = ""
        if node.start_index and node.end_index:
            page_info = f" (pp. {node.start_index}\u2013{node.end_index})"
        elif node.start_index:
            page_info = f" (p. {node.start_index})"
        lines.append(f"{prefix}- **{node.title}**{page_info}")
        lines.extend(_outline_md(node.nodes, indent + 1))
    return lines


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compile_document_page(
    tree: DocumentTree,
    graph: DocumentGraph | None = None,
) -> str:
    """Return Obsidian-compatible markdown for a single document page.

    Includes title, stats, tree outline, and an entity table with backlinks.
    """
    all_nodes = list(iter_nodes(tree.structure))
    node_count = len(all_nodes)

    page_nums = set()
    for n in all_nodes:
        if n.start_index:
            page_nums.add(n.start_index)
        if n.end_index:
            page_nums.add(n.end_index)
    page_count = max(page_nums) if page_nums else 0

    entity_count = len(graph.entities) if graph else 0

    lines: list[str] = []
    lines.append(f"# {tree.doc_name}")
    lines.append("")
    if tree.doc_description:
        lines.append(f"> {tree.doc_description}")
        lines.append("")

    # Stats
    lines.append("| Stat | Value |")
    lines.append("|------|-------|")
    lines.append(f"| Pages | {page_count} |")
    lines.append(f"| Nodes | {node_count} |")
    lines.append(f"| Entities | {entity_count} |")
    lines.append("")

    # Tree outline
    lines.append("## Structure")
    lines.append("")
    lines.extend(_outline_md(tree.structure))
    lines.append("")

    # Entity table
    if graph and graph.entities:
        lines.append("## Entities")
        lines.append("")
        lines.append("| Entity | Type |")
        lines.append("|--------|------|")
        for ent in graph.entities:
            slug = slugify(ent.name)
            link = f"[[concepts/{slug}|{ent.name}]]"
            lines.append(f"| {link} | {ent.entity_type} |")
        lines.append("")

    return "\n".join(lines)


def _description_block(descriptions: list[str]) -> list[str]:
    """Render the description section from a list of descriptions."""
    lines: list[str] = []
    if descriptions:
        sorted_descs = sorted(descriptions, key=len, reverse=True)
        lines.append(sorted_descs[0])
        lines.append("")
        if len(sorted_descs) > 1:
            lines.append("### Additional Notes")
            lines.append("")
            for desc in sorted_descs[1:]:
                if desc:
                    lines.append(f"- {desc}")
            lines.append("")
    return lines


def _source_docs_block(source_docs: list[tuple[str, str]]) -> list[str]:
    """Render the source documents section."""
    lines: list[str] = []
    if source_docs:
        lines.append("## Source Documents")
        lines.append("")
        for doc_name, doc_slug in source_docs:
            lines.append(f"- [[documents/{doc_slug}|{doc_name}]]")
        lines.append("")
    return lines


def _relationships_block(relationships: list[tuple[str, str, str]]) -> list[str]:
    """Render the related concepts section."""
    lines: list[str] = []
    if relationships:
        lines.append("## Related Concepts")
        lines.append("")
        lines.append("| Concept | Relationship |")
        lines.append("|---------|-------------|")
        for target_name, target_slug, keywords in relationships:
            link = f"[[concepts/{target_slug}|{target_name}]]"
            lines.append(f"| {link} | {keywords} |")
        lines.append("")
    return lines


def compile_concept_page(
    name: str,
    entity_type: str,
    descriptions: list[str],
    source_docs: list[tuple[str, str]],
    relationships: list[tuple[str, str, str]],
) -> str:
    """Return markdown for a concept (entity) page with backlinks.

    Uses type-specific templates based on ``entity_type``:

    - **Organization**: Description, Key People, Financial Data, Related Orgs
    - **Person**: Description, Role, Affiliations, Mentions
    - **FinancialItem / Metric**: Description, Values, Trends, Sources
    - **TimePeriod**: Description, Events, Related Periods
    - **Default**: Description, Source Documents, Related Concepts

    Parameters
    ----------
    name:
        Entity name.
    entity_type:
        Entity type string.
    descriptions:
        Descriptions gathered from different documents.  The longest is used
        as the primary description; others are appended as additional notes.
    source_docs:
        List of ``(doc_name, doc_slug)`` tuples.
    relationships:
        List of ``(target_name, target_slug, keywords)`` tuples.
    """
    lines: list[str] = []
    lines.append(f"# {name}")
    lines.append("")
    lines.append(f"**Type:** {entity_type}")
    lines.append("")

    etype = entity_type.lower()

    if etype == "organization":
        # --- Organization template ---
        lines.append("## Description")
        lines.append("")
        lines.extend(_description_block(descriptions))

        lines.append("## Key People")
        lines.append("")
        lines.append("_No people linked yet._")
        lines.append("")

        lines.append("## Financial Data")
        lines.append("")
        lines.append("_No financial data linked yet._")
        lines.append("")

        lines.append("## Related Orgs")
        lines.append("")
        lines.extend(_relationships_block(relationships))
        lines.extend(_source_docs_block(source_docs))

    elif etype == "person":
        # --- Person template ---
        lines.append("## Description")
        lines.append("")
        lines.extend(_description_block(descriptions))

        lines.append("## Role")
        lines.append("")
        lines.append("_No role information yet._")
        lines.append("")

        lines.append("## Affiliations")
        lines.append("")
        lines.extend(_relationships_block(relationships))

        lines.append("## Mentions")
        lines.append("")
        lines.extend(_source_docs_block(source_docs))

    elif etype in ("financialitem", "metric"):
        # --- FinancialItem / Metric template ---
        lines.append("## Description")
        lines.append("")
        lines.extend(_description_block(descriptions))

        lines.append("## Values")
        lines.append("")
        lines.append("_No values recorded yet._")
        lines.append("")

        lines.append("## Trends")
        lines.append("")
        lines.append("_No trend data yet._")
        lines.append("")

        lines.append("## Sources")
        lines.append("")
        lines.extend(_source_docs_block(source_docs))
        lines.extend(_relationships_block(relationships))

    elif etype == "timeperiod":
        # --- TimePeriod template ---
        lines.append("## Description")
        lines.append("")
        lines.extend(_description_block(descriptions))

        lines.append("## Events")
        lines.append("")
        lines.append("_No events linked yet._")
        lines.append("")

        lines.append("## Related Periods")
        lines.append("")
        lines.extend(_relationships_block(relationships))
        lines.extend(_source_docs_block(source_docs))

    else:
        # --- Default template (original format) ---
        lines.extend(_description_block(descriptions))
        lines.extend(_source_docs_block(source_docs))
        lines.extend(_relationships_block(relationships))

    return "\n".join(lines)


def compile_index_page(config: KBConfig, query_count: int = 0) -> str:
    """Return markdown for the master index page.

    Lists all documents and concepts with links, plus recent query count.
    """
    lines: list[str] = []
    lines.append("# Knowledge Base Index")
    lines.append("")

    lines.append(f"**Documents:** {len(config.documents)}  ")
    lines.append(f"**Concepts:** {len(config.concept_index)}  ")
    lines.append(f"**Recent queries:** {query_count}")
    lines.append("")

    # Document listing
    lines.append("## Documents")
    lines.append("")
    for doc in config.documents:
        slug = slugify(doc.doc_name)
        lines.append(f"- [[documents/{slug}|{doc.doc_name}]]")
    lines.append("")

    # Concept listing
    if config.concept_index:
        lines.append("## Concepts")
        lines.append("")
        for concept_name in sorted(config.concept_index.keys()):
            slug = slugify(concept_name)
            lines.append(f"- [[concepts/{slug}|{concept_name}]]")
        lines.append("")

    return "\n".join(lines)


def compile_query_page(
    question: str,
    answer_content: str,
    citations: list[Citation],
    concepts: list[tuple[str, str]],
) -> str:
    """Return markdown for a query result page.

    Parameters
    ----------
    question:
        The user's question.
    answer_content:
        The answer text.
    citations:
        List of ``Citation`` objects (has ``title``, ``pages``, ``node_id``).
    concepts:
        List of ``(name, slug)`` tuples for related concepts.
    """
    lines: list[str] = []
    lines.append(f"# Query: {question}")
    lines.append("")
    lines.append("## Answer")
    lines.append("")
    lines.append(answer_content)
    lines.append("")

    # Citations
    if citations:
        lines.append("## Citations")
        lines.append("")
        for cit in citations:
            pages_str = ", ".join(str(p) for p in cit.pages) if cit.pages else "N/A"
            doc_slug = slugify(cit.doc_name) if cit.doc_name else ""
            if doc_slug:
                lines.append(
                    f"- [[documents/{doc_slug}|{cit.title}]] "
                    f"(pp. {pages_str}, node {cit.node_id})"
                )
            else:
                lines.append(
                    f"- {cit.title} (pp. {pages_str}, node {cit.node_id})"
                )
        lines.append("")

    # Related concepts
    if concepts:
        lines.append("## Related Concepts")
        lines.append("")
        for name, slug in concepts:
            lines.append(f"- [[concepts/{slug}|{name}]]")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Incremental update
# ---------------------------------------------------------------------------

def _collect_entity_data(
    entity_name: str,
    all_graphs: dict[str, DocumentGraph],
    all_doc_slugs: dict[str, str],
) -> tuple[str, list[str], list[tuple[str, str]], list[tuple[str, str, str]]]:
    """Gather merged data for one entity across all graphs.

    Returns (entity_type, descriptions, source_docs, relationships).
    """
    entity_type = "Other"
    descriptions: list[str] = []
    source_docs: list[tuple[str, str]] = []
    relationships: list[tuple[str, str, str]] = []
    seen_docs: set[str] = set()
    seen_rels: set[tuple[str, str]] = set()

    for doc_name, graph in all_graphs.items():
        for ent in graph.entities:
            if ent.name == entity_name:
                entity_type = ent.entity_type
                if ent.description and ent.description not in descriptions:
                    descriptions.append(ent.description)
                if doc_name not in seen_docs:
                    seen_docs.add(doc_name)
                    doc_slug = all_doc_slugs.get(doc_name, slugify(doc_name))
                    source_docs.append((doc_name, doc_slug))

        for rel in graph.relationships:
            if rel.source == entity_name:
                key = (rel.target, rel.keywords)
                if key not in seen_rels:
                    seen_rels.add(key)
                    relationships.append(
                        (rel.target, slugify(rel.target), rel.keywords)
                    )
            elif rel.target == entity_name:
                key = (rel.source, rel.keywords)
                if key not in seen_rels:
                    seen_rels.add(key)
                    relationships.append(
                        (rel.source, slugify(rel.source), rel.keywords)
                    )

    return entity_type, descriptions, source_docs, relationships


def incremental_update(
    wiki_path: Path,
    new_doc: KBDocument,
    new_tree: DocumentTree,
    new_graph: DocumentGraph | None,
    config: KBConfig,
    all_graphs: dict[str, DocumentGraph],
) -> None:
    """Incrementally update the wiki after adding a new document.

    1. Creates ``documents/<slug>.md`` for the new document.
    2. For each entity in *new_graph*: creates or updates
       ``concepts/<slug>.md`` by merging data from *all_graphs*.
    3. Updates ``config.concept_index``.
    4. Regenerates ``_index.md``.
    """
    docs_dir = wiki_path / "documents"
    concepts_dir = wiki_path / "concepts"
    docs_dir.mkdir(parents=True, exist_ok=True)
    concepts_dir.mkdir(parents=True, exist_ok=True)

    # Build doc slug lookup from config
    all_doc_slugs: dict[str, str] = {}
    for doc in config.documents:
        all_doc_slugs[doc.doc_name] = slugify(doc.doc_name)

    # 1. Document page
    doc_slug = slugify(new_doc.doc_name)
    doc_md = compile_document_page(new_tree, new_graph)
    (docs_dir / f"{doc_slug}.md").write_text(doc_md, encoding="utf-8")

    # 2. Concept pages
    if new_graph:
        for ent in new_graph.entities:
            ent_slug = slugify(ent.name)

            # Update concept_index
            if ent.name not in config.concept_index:
                config.concept_index[ent.name] = []
            if new_doc.doc_id not in config.concept_index[ent.name]:
                config.concept_index[ent.name].append(new_doc.doc_id)

            # Merge from all graphs
            entity_type, descriptions, source_docs, relationships = (
                _collect_entity_data(ent.name, all_graphs, all_doc_slugs)
            )

            concept_md = compile_concept_page(
                name=ent.name,
                entity_type=entity_type,
                descriptions=descriptions,
                source_docs=source_docs,
                relationships=relationships,
            )
            (concepts_dir / f"{ent_slug}.md").write_text(
                concept_md, encoding="utf-8"
            )

    # 3. Regenerate index
    index_md = compile_index_page(config, query_count=0)
    (wiki_path / "_index.md").write_text(index_md, encoding="utf-8")
