"""KnowledgeBase — persistent, multi-document knowledge store.

Wraps :class:`NanoIndex` and :class:`DocumentStore` to provide a
directory-backed knowledge base that survives across sessions.

Usage::

    from nanoindex.kb import KnowledgeBase

    kb = KnowledgeBase("./my_kb")
    kb.add("report.pdf")
    answer = kb.ask("What was the revenue?")
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nanoindex.models import DocumentGraph, DocumentTree

# numpy imported lazily in methods that need it

from nanoindex import NanoIndex, _run
from nanoindex.core.document_store import DocumentStore
from nanoindex.models import (
    Answer,
    DocumentGraph,
    KBConfig,
    KBDocument,
)
from nanoindex.utils.slug import slugify
from nanoindex.utils.tree_ops import load_tree, save_tree

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """Open or create a persistent knowledge base at *path*."""

    def __init__(self, path: str | Path, **kwargs: Any) -> None:
        self.path = Path(path)
        self._data_dir = self.path / ".nanoindex"
        self._ni = NanoIndex(**kwargs)
        self._store = DocumentStore()
        self._config: KBConfig

        if (self._data_dir / "config.json").exists():
            self._config = self._load_config()
            self._load_all()
        else:
            # Create directory structure
            for subdir in ("trees", "graphs", "embeddings", "queries"):
                (self._data_dir / subdir).mkdir(parents=True, exist_ok=True)
            self._config = KBConfig(
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            self._save_config()

    # ------------------------------------------------------------------
    # Log
    # ------------------------------------------------------------------

    def _append_log(self, action: str, detail: str) -> None:
        """Append an entry to ``{wiki_path}/log.md``."""
        log_path = self.path / "log.md"
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        entry = f"\n## [{date_str}] {action} | {detail}\n"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(entry)

    # ------------------------------------------------------------------
    # Add document
    # ------------------------------------------------------------------

    def add(self, pdf_path: str | Path) -> KBDocument:
        """Add a document by indexing a PDF (sync wrapper)."""
        return _run(self.async_add(pdf_path))

    def add_tree(
        self,
        tree: "DocumentTree",
        graph: "DocumentGraph | None" = None,
        source_path: str = "",
    ) -> KBDocument:
        """Add a pre-built tree (and optional graph) directly to the wiki.

        Use this when you already have cached trees and graphs from
        benchmarks or prior indexing runs::

            from nanoindex.utils.tree_ops import load_tree
            tree = load_tree("benchmarks/cache_v3/3M_2018_10K.json")
            graph = DocumentGraph(**json.load(open("benchmarks/graphs_v4/3M_2018_10K.json")))
            kb.add_tree(tree, graph)
        """
        doc_name = tree.doc_name

        # Save tree
        tree_rel = f"trees/{doc_name}.json"
        (self._data_dir / "trees").mkdir(parents=True, exist_ok=True)
        save_tree(tree, self._data_dir / tree_rel)

        # Save graph
        graph_rel: str | None = None
        if graph is not None:
            graph_rel = f"graphs/{doc_name}.json"
            (self._data_dir / "graphs").mkdir(parents=True, exist_ok=True)
            with open(self._data_dir / graph_rel, "w") as f:
                json.dump(graph.model_dump(), f, ensure_ascii=False)
            self._ni._graphs[doc_name] = graph

        # Add to document store
        self._store.add(tree, doc_id=doc_name)

        # Create KB record
        doc_id = slugify(doc_name)
        now = datetime.now(timezone.utc).isoformat()
        tree_json = json.dumps(tree.model_dump(exclude_none=True), sort_keys=True)
        content_hash = hashlib.md5(tree_json.encode()).hexdigest()
        kb_doc = KBDocument(
            doc_id=doc_id,
            doc_name=doc_name,
            source_path=source_path,
            added_at=now,
            tree_path=tree_rel,
            graph_path=graph_rel,
            content_hash=content_hash,
        )
        self._config.documents.append(kb_doc)

        # Wiki compile
        try:
            from nanoindex.core import wiki_compiler

            wiki_compiler.incremental_update(
                wiki_path=self.path,
                new_doc=kb_doc,
                new_tree=tree,
                new_graph=graph,
                config=self._config,
                all_graphs={
                    d.doc_id: self._load_graph(d)
                    for d in self._config.documents
                    if self._graph_path(d).exists()
                },
            )
        except Exception:
            logger.debug("Wiki update failed; skipping", exc_info=True)

        self._save_config()

        # Log
        from nanoindex.utils.tree_ops import iter_nodes

        node_count = len(list(iter_nodes(tree.structure)))
        entity_count = len(graph.entities) if graph else 0
        self._append_log(
            "ingest",
            f"{doc_name} (from cached tree)\nAdded {node_count} nodes, {entity_count} entities.\n",
        )
        return kb_doc

    async def async_add(self, pdf_path: str | Path) -> KBDocument:
        """Index a PDF, persist artifacts, and update the knowledge base."""
        pdf_path = Path(pdf_path)
        tree = await self._ni.async_index(pdf_path)

        doc_name = tree.doc_name

        # 1. Save tree
        tree_rel = f"trees/{doc_name}.json"
        save_tree(tree, self._data_dir / tree_rel)

        # 2. Save graph (if available)
        graph_rel: str | None = None
        graph = self._ni._graphs.get(doc_name)
        if graph is not None:
            graph_rel = f"graphs/{doc_name}.json"
            graph_path = self._data_dir / graph_rel
            graph_path.parent.mkdir(parents=True, exist_ok=True)
            with open(graph_path, "w") as f:
                json.dump(graph.model_dump(), f, indent=2, ensure_ascii=False)

        # 3. Save embeddings (if available)
        emb_rel: str | None = None
        emb_data = self._ni._node_embeddings.get(doc_name)
        if emb_data is not None:
            emb_rel = f"embeddings/{doc_name}.npz"
            emb_path = self._data_dir / emb_rel
            emb_path.parent.mkdir(parents=True, exist_ok=True)
            import numpy as np

            np.savez(emb_path, **{k: np.array(v) for k, v in emb_data.items()})

        # 4. Add tree to DocumentStore
        self._store.add(tree, doc_id=doc_name)

        # 5. Create KBDocument record (with content hash)
        doc_id = slugify(doc_name)
        now = datetime.now(timezone.utc).isoformat()
        tree_json = json.dumps(tree.model_dump(exclude_none=True), sort_keys=True)
        content_hash = hashlib.md5(tree_json.encode()).hexdigest()
        kb_doc = KBDocument(
            doc_id=doc_id,
            doc_name=doc_name,
            source_path=str(pdf_path),
            added_at=now,
            tree_path=tree_rel,
            graph_path=graph_rel,
            embeddings_path=emb_rel,
            content_hash=content_hash,
        )
        self._config.documents.append(kb_doc)

        # 6. Try wiki compiler (lazy import — may not exist yet)
        try:
            from nanoindex.core import wiki_compiler

            wiki_compiler.incremental_update(
                wiki_path=self.path,
                new_doc=kb_doc,
                new_tree=tree,
                new_graph=graph,
                config=self._config,
                all_graphs={
                    d.doc_id: self._load_graph(d)
                    for d in self._config.documents
                    if self._graph_path(d).exists()
                },
            )
        except Exception:
            logger.debug("Wiki update failed; skipping", exc_info=True)

        # 7. Persist config
        self._save_config()

        # 8. Append to activity log
        from nanoindex.utils.tree_ops import iter_nodes

        all_nodes = list(iter_nodes(tree.structure))
        entity_count = len(graph.entities) if graph else 0
        concept_names = list(self._config.concept_index.keys())[-5:]
        concepts_str = ", ".join(concept_names) if concept_names else "none"
        self._append_log(
            "ingest",
            f"{pdf_path.name}\n"
            f"Added {len(all_nodes)} nodes, {entity_count} entities. "
            f"Concepts updated: {concepts_str}.\n",
        )

        return kb_doc

    # ------------------------------------------------------------------
    # Ask
    # ------------------------------------------------------------------

    def ask(self, question: str, mode: str = "fast") -> Answer:
        """Ask a question (sync wrapper)."""
        return _run(self.async_ask(question, mode))

    async def async_ask(self, question: str, mode: str = "fast") -> Answer:
        """Ask a question across the knowledge base."""
        docs = self._config.documents
        if not docs:
            return Answer(content="No documents in the knowledge base.", mode=mode)

        if len(docs) == 1:
            tree = load_tree(self._data_dir / docs[0].tree_path)
            answer = await self._ni.async_ask(question, tree, mode=mode)
        else:
            answer = await self._ni.async_multi_ask(question, self._store, mode=mode)

        # Persist query result
        query_slug = slugify(question)
        try:
            from nanoindex.knowledge import wiki_compiler  # type: ignore[import-untyped]

            query_dir = self._data_dir / "queries"
            query_dir.mkdir(parents=True, exist_ok=True)
            wiki_compiler.compile_query_page(
                self.path,
                question,
                answer,
                query_slug,
            )
        except (ImportError, ModuleNotFoundError):
            logger.debug("wiki_compiler not available; skipping query page")

        # Update concept pages that are mentioned in the answer
        try:
            answer_lower = answer.content.lower()
            concepts_dir = self.path / "concepts"
            for concept_name in self._config.concept_index:
                if concept_name.lower() in answer_lower:
                    concept_slug = slugify(concept_name)
                    concept_path = concepts_dir / f"{concept_slug}.md"
                    if concept_path.exists():
                        existing = concept_path.read_text(encoding="utf-8")
                        if "## Recent Queries" not in existing:
                            existing += "\n## Recent Queries\n"
                        query_link = f"\n- [[queries/{query_slug}|{question}]]"
                        if query_link.strip() not in existing:
                            concept_path.write_text(existing + query_link + "\n", encoding="utf-8")
        except Exception:
            logger.debug("Concept page update on ask failed; skipping", exc_info=True)

        # Append to activity log
        cited = ", ".join(c.title for c in answer.citations[:3]) if answer.citations else "none"
        self._append_log(
            "ask",
            f"{question}\nAnswer: {answer.content[:100]}. Cited: {cited}. Filed to queries/.\n",
        )

        return answer

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str) -> list:
        """Search across all documents (sync wrapper)."""
        return _run(self.async_search(query))

    async def async_search(self, query: str) -> list:
        """Search using multi_search across all documents."""
        if self._store.count == 0:
            return []
        return await self._ni.async_multi_search(query, self._store)

    # ------------------------------------------------------------------
    # Status / lint
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Return summary statistics for the knowledge base."""
        num_docs = len(self._config.documents)
        num_concepts = len(self._config.concept_index)
        num_queries = (
            len(list((self._data_dir / "queries").glob("*.json")))
            if (self._data_dir / "queries").exists()
            else 0
        )

        total_entities = 0
        total_relationships = 0
        for doc in self._config.documents:
            graph = self._ni._graphs.get(doc.doc_name)
            if graph is not None:
                total_entities += len(graph.entities)
                total_relationships += len(graph.relationships)

        return {
            "documents": num_docs,
            "concepts": num_concepts,
            "queries": num_queries,
            "entities": total_entities,
            "relationships": total_relationships,
        }

    def lint(self) -> list[str]:
        """Find inconsistencies in the knowledge base.

        Checks for:
        - Orphan concept files (concept file exists but not in index)
        - Missing concept files (in index but file missing)
        - Broken backlinks
        - Stale content hashes (tree changed since ingestion)
        """
        warnings: list[str] = []

        # Check for missing tree files and stale hashes
        for doc in self._config.documents:
            tree_path = self._data_dir / doc.tree_path
            if not tree_path.exists():
                warnings.append(f"Missing tree file: {doc.tree_path}")
            elif doc.content_hash:
                # Verify tree hash matches stored hash
                with open(tree_path) as f:
                    current_data = json.load(f)
                current_json = json.dumps(current_data, sort_keys=True)
                current_hash = hashlib.md5(current_json.encode()).hexdigest()
                if current_hash != doc.content_hash:
                    warnings.append(
                        f"Stale content hash for {doc.doc_name}: "
                        f"expected {doc.content_hash}, got {current_hash}"
                    )

            if doc.graph_path:
                graph_path = self._data_dir / doc.graph_path
                if not graph_path.exists():
                    warnings.append(f"Missing graph file: {doc.graph_path}")
            if doc.embeddings_path:
                emb_path = self._data_dir / doc.embeddings_path
                if not emb_path.exists():
                    warnings.append(f"Missing embeddings file: {doc.embeddings_path}")

        # Check for orphan tree files
        trees_dir = self._data_dir / "trees"
        if trees_dir.exists():
            known_trees = {doc.tree_path for doc in self._config.documents}
            for tree_file in trees_dir.glob("*.json"):
                rel = f"trees/{tree_file.name}"
                if rel not in known_trees:
                    warnings.append(f"Orphan tree file: {rel}")

        # Append to activity log
        num_concepts = len(self._config.concept_index)
        num_docs = len(self._config.documents)
        self._append_log(
            "lint",
            f"health check\n"
            f"{len(warnings)} warnings. {num_concepts} concepts, {num_docs} document(s).\n",
        )

        return warnings

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _graph_path(self, doc: KBDocument) -> Path:
        """Get the full path to a document's graph file."""
        if doc.graph_path:
            return self._data_dir / doc.graph_path
        return self._data_dir / "graphs" / f"{doc.doc_id}.json"

    def _load_graph(self, doc: KBDocument) -> DocumentGraph:
        """Load a document's graph from disk."""
        gpath = self._graph_path(doc)
        if gpath.exists():
            with open(gpath) as f:
                return DocumentGraph.model_validate(json.load(f))
        return DocumentGraph(doc_name=doc.doc_name)

    def _save_config(self) -> None:
        """Write config to .nanoindex/config.json."""
        config_path = self._data_dir / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(
                self._config.model_dump(),
                f,
                indent=2,
                ensure_ascii=False,
            )

    def _load_config(self) -> KBConfig:
        """Read config from .nanoindex/config.json."""
        config_path = self._data_dir / "config.json"
        with open(config_path) as f:
            data = json.load(f)
        return KBConfig.model_validate(data)

    def _load_all(self) -> None:
        """On resume: load all trees, graphs, and embeddings."""
        for doc in self._config.documents:
            # Load tree into DocumentStore
            tree_path = self._data_dir / doc.tree_path
            if tree_path.exists():
                tree = load_tree(tree_path)
                self._store.add(tree, doc_id=doc.doc_name)

            # Load graph into NanoIndex cache
            if doc.graph_path:
                graph_path = self._data_dir / doc.graph_path
                if graph_path.exists():
                    with open(graph_path) as f:
                        graph = DocumentGraph.model_validate(json.load(f))
                    self._ni._graphs[doc.doc_name] = graph

            # Load embeddings into NanoIndex cache
            if doc.embeddings_path:
                emb_path = self._data_dir / doc.embeddings_path
                if emb_path.exists():
                    import numpy as np

                    loaded = np.load(emb_path)
                    self._ni._node_embeddings[doc.doc_name] = {
                        k: loaded[k].tolist() for k in loaded.files
                    }
