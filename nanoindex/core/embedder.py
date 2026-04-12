"""Embed tree node summaries and perform cosine similarity search.

Used at index time to create node embeddings, and at query time
for cheap (single API call) candidate retrieval.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from nanoindex.models import DocumentTree
from nanoindex.utils.tree_ops import iter_nodes

logger = logging.getLogger(__name__)


async def embed_texts(
    texts: list[str], api_key: str, model: str, base_url: str
) -> list[list[float]]:
    """Embed a batch of texts using OpenAI-compatible API or local model.

    If model starts with 'local:' (e.g. 'local:all-MiniLM-L6-v2'), uses
    sentence-transformers locally. Otherwise uses OpenAI-compatible API.
    """
    if model.startswith("local:"):
        return _embed_local(texts, model.removeprefix("local:"))

    import httpx

    url = f"{base_url.rstrip('/')}/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    all_embeddings: list[list[float]] = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                url,
                headers=headers,
                json={"model": model, "input": batch},
            )
            resp.raise_for_status()
            data = resp.json()
            batch_embs = [
                item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])
            ]
            all_embeddings.extend(batch_embs)

    return all_embeddings


def _embed_local(texts: list[str], model_name: str) -> list[list[float]]:
    """Embed using sentence-transformers locally (no API key needed)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("pip install sentence-transformers — required for local embeddings")

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.tolist()


async def embed_tree(
    tree: DocumentTree,
    api_key: str,
    model: str = "text-embedding-3-small",
    base_url: str = "https://api.openai.com/v1",
) -> dict[str, list[float]]:
    """Embed all node summaries. Returns node_id -> embedding vector."""
    nodes = list(iter_nodes(tree.structure))

    # Build texts: title + summary for each node
    node_ids: list[str] = []
    texts: list[str] = []
    for node in nodes:
        text = node.title
        if node.summary:
            text += f": {node.summary}"
        node_ids.append(node.node_id)
        texts.append(text[:8000])  # Cap to avoid token limits

    if not texts:
        return {}

    logger.info("Embedding %d node summaries with %s", len(texts), model)
    embeddings = await embed_texts(texts, api_key, model, base_url)

    return dict(zip(node_ids, embeddings))


async def embed_query(
    query: str,
    api_key: str,
    model: str = "text-embedding-3-small",
    base_url: str = "https://api.openai.com/v1",
) -> list[float]:
    """Embed a single query string."""
    result = await embed_texts([query], api_key, model, base_url)
    return result[0]


def cosine_search(
    query_vec: list[float],
    node_embeddings: dict[str, list[float]],
    top_k: int = 20,
) -> list[tuple[str, float]]:
    """Pure numpy cosine similarity search. Returns (node_id, score) pairs."""
    if not node_embeddings:
        return []

    node_ids = list(node_embeddings.keys())
    matrix = np.array([node_embeddings[nid] for nid in node_ids])
    qvec = np.array(query_vec)

    # Cosine similarity
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(qvec)
    norms = np.where(norms == 0, 1e-10, norms)
    scores = matrix @ qvec / norms

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(node_ids[i], float(scores[i])) for i in top_indices]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_embeddings(embeddings: dict[str, list[float]], path: Path) -> None:
    """Save embeddings as compressed numpy archive."""
    node_ids = list(embeddings.keys())
    vectors = np.array([embeddings[nid] for nid in node_ids])
    np.savez_compressed(
        path,
        node_ids=np.array(node_ids),
        vectors=vectors,
    )
    logger.info("Saved %d embeddings to %s", len(node_ids), path)


def load_embeddings(path: Path) -> dict[str, list[float]]:
    """Load embeddings from numpy archive."""
    data = np.load(path, allow_pickle=True)
    node_ids = data["node_ids"].tolist()
    vectors = data["vectors"].tolist()
    return dict(zip(node_ids, vectors))
