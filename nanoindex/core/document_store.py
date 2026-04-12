"""Multi-document store with document selection strategies.

Manages multiple indexed ``DocumentTree`` instances and provides
three strategies for selecting relevant documents at query time:

- **metadata**: filter by structured key-value metadata
- **description**: LLM picks relevant docs from their descriptions
- **direct**: explicit list of doc IDs (or all)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from nanoindex.core.llm import LLMClient
from nanoindex.models import DocumentTree
from nanoindex.utils.tree_ops import save_tree, load_tree

logger = logging.getLogger(__name__)

_MANIFEST_FILE = "index.json"


class DocumentEntry(BaseModel):
    """A single document in the store with its tree, metadata, and description."""

    doc_id: str
    doc_name: str
    tree: DocumentTree
    metadata: dict[str, Any] = Field(default_factory=dict)
    description: str | None = None


class DocumentStore:
    """In-memory store of indexed documents with persistence and selection."""

    def __init__(self) -> None:
        self._entries: dict[str, DocumentEntry] = {}

    @property
    def count(self) -> int:
        return len(self._entries)

    def add(
        self,
        tree: DocumentTree,
        *,
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        description: str | None = None,
    ) -> str:
        """Add a document tree to the store. Returns the ``doc_id``."""
        did = doc_id or tree.doc_name
        if did in self._entries:
            logger.warning("Overwriting existing document '%s'", did)

        desc = description or tree.doc_description
        entry = DocumentEntry(
            doc_id=did,
            doc_name=tree.doc_name,
            tree=tree,
            metadata=metadata or {},
            description=desc,
        )
        self._entries[did] = entry
        logger.info("Added document '%s' (%d nodes)", did, len(tree.structure))
        return did

    def remove(self, doc_id: str) -> None:
        if doc_id not in self._entries:
            raise KeyError(f"Document '{doc_id}' not found in store")
        del self._entries[doc_id]

    def get(self, doc_id: str) -> DocumentEntry:
        if doc_id not in self._entries:
            raise KeyError(f"Document '{doc_id}' not found in store")
        return self._entries[doc_id]

    def list_documents(self) -> list[DocumentEntry]:
        return list(self._entries.values())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """Save the entire store to a directory (one JSON per tree + manifest)."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)

        manifest: list[dict[str, Any]] = []
        for entry in self._entries.values():
            tree_file = f"{entry.doc_id}.json"
            save_tree(entry.tree, d / tree_file)
            manifest.append(
                {
                    "doc_id": entry.doc_id,
                    "doc_name": entry.doc_name,
                    "tree_file": tree_file,
                    "metadata": entry.metadata,
                    "description": entry.description,
                }
            )

        with open(d / _MANIFEST_FILE, "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        logger.info("Saved %d documents to %s", len(manifest), d)

    @classmethod
    def load(cls, directory: str | Path) -> DocumentStore:
        """Load a store from a directory previously created by ``save()``."""
        d = Path(directory)
        manifest_path = d / _MANIFEST_FILE
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest found at {manifest_path}")

        with open(manifest_path) as f:
            manifest = json.load(f)

        store = cls()
        for item in manifest:
            tree = load_tree(d / item["tree_file"])
            store.add(
                tree,
                doc_id=item["doc_id"],
                metadata=item.get("metadata", {}),
                description=item.get("description"),
            )
        logger.info("Loaded %d documents from %s", store.count, d)
        return store

    # ------------------------------------------------------------------
    # Selection strategies
    # ------------------------------------------------------------------

    def select_by_metadata(
        self,
        filters: dict[str, Any],
    ) -> list[DocumentEntry]:
        """Filter documents whose metadata matches all key-value pairs.

        Supports exact match and basic comparison operators::

            {"company": "3M"}                   # exact match
            {"year__gte": 2020}                 # year >= 2020
            {"year__lte": 2022}                 # year <= 2022
            {"doc_type__in": ["10-K", "10-Q"]}  # membership
        """
        results: list[DocumentEntry] = []
        for entry in self._entries.values():
            if _matches_filters(entry.metadata, filters):
                results.append(entry)
        return results

    async def select_by_description(
        self,
        query: str,
        llm: LLMClient,
        *,
        max_docs: int = 5,
    ) -> list[DocumentEntry]:
        """Use an LLM to select relevant documents by comparing the query
        against each document's description."""
        entries_with_desc = [e for e in self._entries.values() if e.description]
        if not entries_with_desc:
            logger.warning("No documents have descriptions — returning all")
            return list(self._entries.values())[:max_docs]

        docs_payload = json.dumps(
            [
                {
                    "doc_id": e.doc_id,
                    "doc_name": e.doc_name,
                    "doc_description": e.description,
                }
                for e in entries_with_desc
            ],
            indent=2,
        )

        prompt = _DOC_SELECT_PROMPT.format(query=query, documents=docs_payload)
        messages = [{"role": "user", "content": prompt}]
        resp = await llm.chat(messages, temperature=0.0, max_tokens=512)

        selected_ids = _parse_doc_ids(resp)
        if not selected_ids:
            logger.warning("LLM returned no doc IDs — falling back to all")
            return entries_with_desc[:max_docs]

        results = [self._entries[did] for did in selected_ids if did in self._entries]
        return results[:max_docs]

    def select_direct(
        self,
        doc_ids: list[str] | None = None,
    ) -> list[DocumentEntry]:
        """Return specific documents by ID, or all if no IDs given."""
        if doc_ids is None:
            return list(self._entries.values())
        results = []
        for did in doc_ids:
            if did in self._entries:
                results.append(self._entries[did])
            else:
                logger.warning("Document '%s' not found in store — skipping", did)
        return results


# ------------------------------------------------------------------
# Document selection prompt
# ------------------------------------------------------------------

_DOC_SELECT_PROMPT = """\
You are given a list of documents with their IDs, file names, and descriptions. \
Select the documents that may contain information relevant to answering the query.

Query: {query}

Documents:
{documents}

Response format (JSON only, no other text):
{{
    "thinking": "<your reasoning for document selection>",
    "doc_ids": ["doc_id1", "doc_id2"]
}}

Return an empty list if no documents are relevant."""


def _parse_doc_ids(response: str) -> list[str]:
    """Extract doc_ids from the LLM response."""
    try:
        data = json.loads(response)
        if isinstance(data, dict) and "doc_ids" in data:
            return data["doc_ids"]
    except json.JSONDecodeError:
        pass

    match = re.search(r'"doc_ids"\s*:\s*\[(.*?)\]', response, re.DOTALL)
    if match:
        return re.findall(r'"([^"]+)"', match.group(1))
    return []


# ------------------------------------------------------------------
# Metadata filter matching
# ------------------------------------------------------------------


def _matches_filters(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key, value in filters.items():
        if "__" in key:
            field, op = key.rsplit("__", 1)
            meta_val = metadata.get(field)
            if meta_val is None:
                return False
            if op == "gte" and not (meta_val >= value):
                return False
            elif op == "lte" and not (meta_val <= value):
                return False
            elif op == "gt" and not (meta_val > value):
                return False
            elif op == "lt" and not (meta_val < value):
                return False
            elif op == "in" and meta_val not in value:
                return False
            elif op == "contains" and value not in str(meta_val):
                return False
        else:
            if metadata.get(key) != value:
                return False
    return True
