"""Tree traversal, node lookup, and serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from nanoindex.models import DocumentTree, TreeNode

if TYPE_CHECKING:
    from nanoindex.models import DocumentGraph


def iter_nodes(nodes: list[TreeNode]) -> Iterator[TreeNode]:
    """Depth-first iteration over a tree (or forest) of ``TreeNode`` objects."""
    for node in nodes:
        yield node
        yield from iter_nodes(node.nodes)


def find_node(nodes: list[TreeNode], node_id: str) -> TreeNode | None:
    """Find a node by ``node_id`` anywhere in the tree."""
    for node in iter_nodes(nodes):
        if node.node_id == node_id:
            return node
    return None


def collect_text(node: TreeNode) -> str:
    """Recursively collect all text from *node* and its descendants."""
    parts: list[str] = []
    if node.text:
        parts.append(node.text)
    for child in node.nodes:
        parts.append(collect_text(child))
    return "\n\n".join(p for p in parts if p)


def tree_to_outline(nodes: list[TreeNode], indent: int = 0) -> str:
    """Render a human-readable outline string for display / LLM prompts."""
    lines: list[str] = []
    prefix = "  " * indent
    for node in nodes:
        line = f"{prefix}- [{node.node_id}] {node.title}"
        if node.summary:
            line += f": {node.summary}"
        if node.start_index and node.end_index:
            line += f" (pp. {node.start_index}-{node.end_index})"
        lines.append(line)
        lines.append(tree_to_outline(node.nodes, indent + 1))
    return "\n".join(line for line in lines if line)


def assign_node_ids(nodes: list[TreeNode], prefix: str = "") -> None:
    """Assign zero-padded, depth-first node IDs in-place."""
    for i, node in enumerate(nodes):
        node.node_id = f"{prefix}{i:04d}" if not prefix else f"{prefix}.{i:04d}"
        assign_node_ids(node.nodes, prefix=node.node_id)


def _node_to_dict(node: TreeNode) -> dict:
    """Convert a TreeNode to a clean dict for LLM consumption (no text)."""
    d: dict = {"node_id": node.node_id, "title": node.title}
    if node.summary:
        d["summary"] = node.summary
    if node.start_index and node.end_index:
        d["start_page"] = node.start_index
        d["end_page"] = node.end_index
    if node.nodes:
        d["nodes"] = [_node_to_dict(c) for c in node.nodes]
    return d


def tree_to_json_outline(nodes: list[TreeNode]) -> str:
    """Render the tree as a compact JSON string (no text), for LLM prompts.

    Matches PageIndex's approach of showing `json.dumps(tree_without_text)`.
    """
    return json.dumps([_node_to_dict(n) for n in nodes], indent=2)


def find_siblings(
    structure: list[TreeNode],
    node_id: str,
    max_each_side: int = 2,
) -> list[TreeNode]:
    """Return up to *max_each_side* siblings on each side of *node_id*.

    Searches the tree for the parent that contains *node_id* among its
    children, then returns adjacent siblings (excluding the node itself).
    Returns an empty list if the node is a root or not found.
    """

    def _search(nodes: list[TreeNode]) -> list[TreeNode] | None:
        for parent in nodes:
            for idx, child in enumerate(parent.nodes):
                if child.node_id == node_id:
                    lo = max(0, idx - max_each_side)
                    hi = min(len(parent.nodes), idx + max_each_side + 1)
                    return [parent.nodes[j] for j in range(lo, hi) if j != idx]
            result = _search(parent.nodes)
            if result is not None:
                return result
        return None

    return _search(structure) or []


def save_tree(tree: DocumentTree, path: str | Path) -> None:
    """Serialise a ``DocumentTree`` to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(tree.model_dump(exclude_none=True), f, indent=2, ensure_ascii=False)


def load_tree(path: str | Path) -> DocumentTree:
    """Deserialise a ``DocumentTree`` from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return DocumentTree.model_validate(data)


def load_graph(path: str | Path) -> "DocumentGraph":
    """Deserialise a ``DocumentGraph`` from a JSON file."""
    from nanoindex.models import DocumentGraph

    with open(path) as f:
        data = json.load(f)
    return DocumentGraph.model_validate(data)
