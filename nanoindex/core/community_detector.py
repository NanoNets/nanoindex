"""Community detection for entity graphs.

Groups densely-connected entities into communities for global queries.
Communities enable global queries like "What are the main themes?"
"""

import logging
from nanoindex.models import DocumentGraph

logger = logging.getLogger(__name__)


class Community:
    """A group of related entities."""

    def __init__(self, id: int, entity_names: list[str], label: str = ""):
        self.id = id
        self.entity_names = entity_names
        self.label = label  # auto-generated or LLM-generated
        self.summary = ""  # LLM-generated summary


def detect_communities(graph: DocumentGraph) -> list[Community]:
    """Detect entity communities using Louvain algorithm."""
    import networkx as nx

    if not graph.entities or not graph.relationships:
        return []

    G = nx.Graph()
    for e in graph.entities:
        G.add_node(e.name)
    for r in graph.relationships:
        if G.has_node(r.source) and G.has_node(r.target):
            G.add_edge(r.source, r.target, weight=1)

    # Remove isolated nodes (no relationships)
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    if len(G.nodes) < 2:
        return []

    # Louvain community detection
    from networkx.algorithms.community import louvain_communities

    try:
        communities_sets = louvain_communities(G, resolution=1.0, seed=42)
    except Exception:
        return []

    communities = []

    for idx, members in enumerate(communities_sets):
        if len(members) < 2:
            continue

        member_list = sorted(members)

        # Auto-generate label from top entities (by connection count)
        degrees = [(n, G.degree(n)) for n in members]
        degrees.sort(key=lambda x: -x[1])
        top_names = [n for n, _ in degrees[:3]]
        label = " / ".join(top_names)

        communities.append(
            Community(
                id=idx,
                entity_names=member_list,
                label=label,
            )
        )

    logger.info("Detected %d communities from %d entities", len(communities), len(G.nodes))
    return communities


def auto_summarize_community(community: Community, graph: DocumentGraph) -> str:
    """Generate a text summary of a community without LLM."""
    entity_map = {e.name.lower(): e for e in graph.entities}

    lines = [f"Community: {community.label}"]
    lines.append(f"Entities ({len(community.entity_names)}):")

    for name in community.entity_names[:10]:
        ent = entity_map.get(name.lower())
        if ent:
            lines.append(f"  - [{ent.entity_type}] {ent.name}: {ent.description[:60]}")
        else:
            lines.append(f"  - {name}")

    if len(community.entity_names) > 10:
        lines.append(f"  ... and {len(community.entity_names) - 10} more")

    # Add key relationships within community
    member_set = {n.lower() for n in community.entity_names}
    internal_rels = [
        r
        for r in graph.relationships
        if r.source.lower() in member_set and r.target.lower() in member_set
    ]

    if internal_rels:
        lines.append(f"\nRelationships ({len(internal_rels)}):")
        for r in internal_rels[:5]:
            lines.append(f"  {r.source} --{r.keywords}--> {r.target}")

    return "\n".join(lines)


async def llm_summarize_community(community: Community, graph: DocumentGraph, llm) -> str:
    """Generate LLM summary for a community."""
    context = auto_summarize_community(community, graph)
    prompt = f"""Summarize this group of related entities in 2-3 sentences.
What is the theme or topic that connects them?

{context}

Summary:"""
    try:
        return await llm.chat(
            [{"role": "user", "content": prompt}], max_tokens=200, temperature=0.0
        )
    except Exception:
        return auto_summarize_community(community, graph)
