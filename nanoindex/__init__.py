"""NanoIndex — Nanonets-powered document intelligence.

Public API:

    import nanoindex

    tree = nanoindex.index("report.pdf")
    nodes = nanoindex.search("What was the revenue?", tree)
    answer = nanoindex.ask("What was the revenue?", tree)

    # Or use the class-based API for full control:
    from nanoindex import NanoIndex, NanoIndexConfig
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from nanoindex.config import NanoIndexConfig, load_config
from nanoindex.exceptions import (
    ConfigError,
    ExtractionError,
    GenerationError,
    NanoIndexError,
    RateLimitError,
    RetrievalError,
    TreeBuildError,
)
from nanoindex.core.document_store import DocumentStore
from nanoindex.models import Answer, DocumentGraph, DocumentTree, RetrievedNode, TreeNode

__all__ = [
    "NanoIndex",
    "NanoIndexConfig",
    "DocumentStore",
    "index",
    "search",
    "ask",
    "visualize",
    "Answer",
    "DocumentGraph",
    "DocumentTree",
    "RetrievedNode",
    "TreeNode",
    "NanoIndexError",
    "ConfigError",
    "ExtractionError",
    "RateLimitError",
    "TreeBuildError",
    "RetrievalError",
    "GenerationError",
]

__version__ = "0.1.0"


def _run(coro):
    """Run an async coroutine from synchronous code."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


class NanoIndex:
    """Main orchestrator — wires extraction, tree building, enrichment,
    retrieval, and generation into a cohesive pipeline."""

    def __init__(self, config: NanoIndexConfig | None = None, **kwargs: Any) -> None:
        self.config = config or load_config(**kwargs)
        self._client = None
        self._llm = None
        self._reasoning_llm = None
        # Graph + embedding caches (loaded lazily)
        self._node_embeddings: dict[str, dict[str, list[float]]] = {}
        self._graphs: dict[str, DocumentGraph] = {}

    def _get_client(self):
        if self._client is None:
            from nanoindex.core.client import NanonetsClient
            key = self.config.require_nanonets_key()
            self._client = NanonetsClient(api_key=key)
        return self._client

    def _get_llm(self):
        """Default LLM — used for enrichment, refining, and summarization."""
        if self._llm is None:
            from nanoindex.core.llm import LLMClient
            self._llm = LLMClient(
                api_key=self.config.require_llm_key(),
                base_url=self.config.llm_base_url,
                model=self.config.llm_model,
            )
        return self._llm

    def _get_reasoning_llm(self):
        """Reasoning LLM — used for retrieval + answer generation.

        Falls back to the default LLM when no reasoning LLM is configured.
        """
        if self._reasoning_llm is None:
            cfg = self.config
            if cfg.reasoning_llm_model:
                from nanoindex.core.llm import LLMClient
                self._reasoning_llm = LLMClient(
                    api_key=cfg.reasoning_llm_api_key or cfg.require_llm_key(),
                    base_url=cfg.reasoning_llm_base_url or cfg.llm_base_url,
                    model=cfg.reasoning_llm_model,
                )
            else:
                self._reasoning_llm = self._get_llm()
        return self._reasoning_llm

    # ------------------------------------------------------------------
    # Async core
    # ------------------------------------------------------------------

    async def async_index(
        self,
        file_path: str | Path,
        *,
        add_summaries: bool | None = None,
        add_doc_description: bool | None = None,
    ) -> DocumentTree:
        """Extract, build tree, refine oversized nodes, and optionally enrich."""
        from nanoindex.core.tree_builder import build_document_tree
        from nanoindex.core.refiner import refine_tree
        from nanoindex.core.enricher import enrich_tree

        path = Path(file_path)
        parsed_document = None

        if self.config.parser == "nanonets":
            from nanoindex.core.extractor import extract_document
            client = self._get_client()
            extraction = await extract_document(path, client)
        else:
            from nanoindex.core.parsers import get_parser
            parser_kwargs: dict[str, Any] = {}
            if self.config.parser == "nanonets":
                parser_kwargs["api_key"] = self.config.require_nanonets_key()
            parser = get_parser(self.config.parser, **parser_kwargs)
            parsed_document = await parser.parse(path)
            extraction = parsed_document.to_extraction_result()

        tree = build_document_tree(extraction, path.stem, self.config)

        # Store the ParsedDocument on the tree for later multimodal processing
        if parsed_document is not None:
            tree._parsed_document = parsed_document  # type: ignore[attr-defined]

        llm = self._get_llm()

        # Split oversized leaf nodes (heuristic + optional LLM)
        tree = await refine_tree(tree, llm, self.config)

        should_summarise = add_summaries if add_summaries is not None else self.config.add_summaries
        should_describe = (
            add_doc_description if add_doc_description is not None else self.config.add_doc_description
        )

        if should_summarise or should_describe:
            cfg = self.config.model_copy()
            if add_summaries is not None:
                cfg.add_summaries = add_summaries
            if add_doc_description is not None:
                cfg.add_doc_description = add_doc_description
            tree = await enrich_tree(tree, llm, cfg)

        return tree

    async def async_search(
        self,
        query: str,
        tree: DocumentTree,
    ) -> list[RetrievedNode]:
        """Search the document tree for nodes relevant to *query*."""
        from nanoindex.core.retriever import search as _search
        llm = self._get_reasoning_llm()
        return await _search(query, tree, llm, self.config)

    async def async_ask(
        self,
        query: str,
        tree: DocumentTree,
        *,
        mode: str = "text",
        pdf_path: str | Path | None = None,
        include_metadata: bool = False,
    ) -> Answer:
        """Search + generate an answer in one step.

        Modes:
          - ``"text"``: text-only context
          - ``"vision"``: PDF page images + text
          - ``"agentic"``: multi-round agent that iteratively reads sections
          - ``"agentic_vision"``: agentic mode with PDF page images

        When *include_metadata* is ``True``, each citation carries
        bounding boxes and page dimensions for its source pages.
        """
        if mode.startswith("fast"):
            from nanoindex.core.fast_retriever import fast_search
            from nanoindex.core.generator import generate_answer
            llm = self._get_reasoning_llm()
            node_embeddings = self._node_embeddings.get(tree.doc_name)
            graph = self._graphs.get(tree.doc_name)
            nodes = await fast_search(
                query, tree, llm, self.config,
                node_embeddings=node_embeddings,
                graph=graph,
            )
            gen_mode = "vision" if "vision" in mode else "text"
            return await generate_answer(
                query, nodes, llm,
                mode=gen_mode, pdf_path=pdf_path,
                tree=tree, include_metadata=include_metadata,
            )
        if mode.startswith("agentic"):
            from nanoindex.core.agentic import agentic_ask
            llm = self._get_reasoning_llm()
            return await agentic_ask(
                query, tree, llm, self.config,
                pdf_path=pdf_path,
                use_vision="vision" in mode,
                include_metadata=include_metadata,
            )
        from nanoindex.core.generator import generate_answer
        nodes = await self.async_search(query, tree)
        llm = self._get_reasoning_llm()
        return await generate_answer(
            query, nodes, llm,
            mode=mode, pdf_path=pdf_path,
            tree=tree, include_metadata=include_metadata,
        )

    # ------------------------------------------------------------------
    # Multi-document search
    # ------------------------------------------------------------------

    async def async_multi_search(
        self,
        query: str,
        store: "DocumentStore",
        *,
        strategy: str = "description",
        filters: dict[str, Any] | None = None,
        doc_ids: list[str] | None = None,
        max_docs: int = 5,
    ) -> list[RetrievedNode]:
        """Search across multiple documents in a ``DocumentStore``.

        Strategy options:
          - ``"metadata"``: filter by metadata fields (pass *filters*)
          - ``"description"``: LLM selects docs by their descriptions
          - ``"direct"``: search specific *doc_ids* (or all if None)
        """
        import asyncio as _aio
        from nanoindex.core.retriever import search as _search

        entries = await self._select_docs(query, store, strategy, filters, doc_ids, max_docs)
        if not entries:
            return []

        llm = self._get_reasoning_llm()

        async def _search_one(entry):
            nodes = await _search(query, entry.tree, llm, self.config)
            for n in nodes:
                n.doc_name = entry.doc_name
            return nodes

        tasks = [_search_one(e) for e in entries]
        per_doc_results = await _aio.gather(*tasks, return_exceptions=True)

        merged: list[RetrievedNode] = []
        for result in per_doc_results:
            if isinstance(result, Exception):
                import logging
                logging.getLogger(__name__).warning("Search failed for a document: %s", result)
                continue
            merged.extend(result)

        return merged

    async def async_multi_ask(
        self,
        query: str,
        store: "DocumentStore",
        *,
        strategy: str = "description",
        filters: dict[str, Any] | None = None,
        doc_ids: list[str] | None = None,
        max_docs: int = 5,
        mode: str = "text",
        include_metadata: bool = False,
    ) -> Answer:
        """Search across multiple documents and generate a unified answer."""
        from nanoindex.core.generator import generate_answer
        nodes = await self.async_multi_search(
            query, store, strategy=strategy, filters=filters,
            doc_ids=doc_ids, max_docs=max_docs,
        )
        llm = self._get_reasoning_llm()
        return await generate_answer(
            query, nodes, llm,
            mode=mode, include_metadata=include_metadata,
        )

    async def _select_docs(self, query, store, strategy, filters, doc_ids, max_docs):
        if strategy == "metadata":
            if not filters:
                raise ValueError("filters are required for 'metadata' strategy")
            return store.select_by_metadata(filters)
        elif strategy == "description":
            llm = self._get_reasoning_llm()
            return await store.select_by_description(query, llm, max_docs=max_docs)
        elif strategy == "direct":
            return store.select_direct(doc_ids)
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}. Use 'metadata', 'description', or 'direct'.")

    def load_graph(self, doc_name: str, graph_path: str | Path) -> None:
        """Load a pre-built graph for a document."""
        from nanoindex.core.entity_extractor import load_graph
        self._graphs[doc_name] = load_graph(Path(graph_path))

    def load_embeddings(self, doc_name: str, embeddings_path: str | Path) -> None:
        """Load pre-built embeddings for a document."""
        from nanoindex.core.embedder import load_embeddings
        self._node_embeddings[doc_name] = load_embeddings(Path(embeddings_path))

    async def async_build_graph(
        self,
        tree: DocumentTree,
        parsed: "ParsedDocument | None" = None,
    ) -> DocumentGraph:
        """Extract entities and relationships from a tree.

        When *parsed* is supplied and contains ``modal_contents``,
        multimodal entities (images, tables, etc.) are extracted and
        merged into the graph.
        """
        from nanoindex.core.entity_extractor import extract_entities, extract_multimodal_entities
        from nanoindex.models import ParsedDocument as _ParsedDocument  # noqa: F811

        llm = self._get_llm()
        graph = await extract_entities(tree, llm)

        # If a ParsedDocument was stashed on the tree during indexing, use it
        if parsed is None:
            parsed = getattr(tree, "_parsed_document", None)

        if parsed is not None and parsed.modal_contents:
            mm_entities, mm_relationships = await extract_multimodal_entities(
                parsed, tree, llm,
            )
            graph.entities.extend(mm_entities)
            graph.relationships.extend(mm_relationships)

        self._graphs[tree.doc_name] = graph
        return graph

    async def async_build_embeddings(self, tree: DocumentTree) -> dict[str, list[float]]:
        """Embed all node summaries for a tree."""
        from nanoindex.core.embedder import embed_tree
        embed_key = self.config.embedding_api_key or self.config.require_llm_key()
        embeddings = await embed_tree(
            tree,
            api_key=embed_key,
            model=self.config.embedding_model,
            base_url=self.config.embedding_base_url,
        )
        self._node_embeddings[tree.doc_name] = embeddings
        return embeddings

    async def async_close(self) -> None:
        if self._client:
            await self._client.close()
        if self._llm:
            await self._llm.close()
        if self._reasoning_llm and self._reasoning_llm is not self._llm:
            await self._reasoning_llm.close()

    # ------------------------------------------------------------------
    # Sync wrappers
    # ------------------------------------------------------------------

    def index(self, file_path: str | Path, **kwargs: Any) -> DocumentTree:
        return _run(self.async_index(file_path, **kwargs))

    def search(self, query: str, tree: DocumentTree) -> list[RetrievedNode]:
        return _run(self.async_search(query, tree))

    def ask(
        self,
        query: str,
        tree: DocumentTree,
        *,
        mode: str = "text",
        pdf_path: str | Path | None = None,
        include_metadata: bool = False,
    ) -> Answer:
        return _run(self.async_ask(
            query, tree, mode=mode, pdf_path=pdf_path,
            include_metadata=include_metadata,
        ))

    def multi_search(
        self, query: str, store: "DocumentStore", **kwargs: Any,
    ) -> list[RetrievedNode]:
        return _run(self.async_multi_search(query, store, **kwargs))

    def multi_ask(
        self, query: str, store: "DocumentStore", **kwargs: Any,
    ) -> Answer:
        return _run(self.async_multi_ask(query, store, **kwargs))

    def close(self) -> None:
        _run(self.async_close())


# ------------------------------------------------------------------
# Module-level convenience functions
# ------------------------------------------------------------------

def index(file_path: str | Path, **kwargs: Any) -> DocumentTree:
    """Index a document using default config (reads NANONETS_API_KEY from env)."""
    ni = NanoIndex(**kwargs)
    return ni.index(file_path)


def search(query: str, tree: DocumentTree, **kwargs: Any) -> list[RetrievedNode]:
    """Search a document tree for relevant nodes."""
    ni = NanoIndex(**kwargs)
    return ni.search(query, tree)


def ask(
    query: str,
    tree: DocumentTree,
    *,
    mode: str = "text",
    pdf_path: str | Path | None = None,
    include_metadata: bool = False,
    **kwargs: Any,
) -> Answer:
    """Index (if needed), search, and answer a question about a document."""
    ni = NanoIndex(**kwargs)
    return ni.ask(
        query, tree, mode=mode, pdf_path=pdf_path,
        include_metadata=include_metadata,
    )


def visualize(
    tree: DocumentTree,
    graph: DocumentGraph | None = None,
    *,
    port: int = 3000,
) -> None:
    """Open the interactive visualization dashboard for a tree.

    Usage::

        tree = nanoindex.index("report.pdf")
        nanoindex.visualize(tree)
    """
    import shutil
    import subprocess
    import threading
    import webbrowser

    viz_dir = Path(__file__).resolve().parent.parent / "viz"
    if not viz_dir.exists():
        raise RuntimeError("Viz directory not found. Install NanoIndex from source.")

    if not shutil.which("node"):
        raise RuntimeError("Node.js not found. Install from https://nodejs.org")

    # Save tree to cache for the dashboard to pick up
    from nanoindex.utils.tree_ops import save_tree

    cache_dir = viz_dir.parent / "benchmarks" / "cache_v3"
    cache_dir.mkdir(parents=True, exist_ok=True)
    doc_name = tree.doc_name or "document"
    save_tree(tree, cache_dir / f"{doc_name}.json")

    # Save graph if provided
    if graph:
        from nanoindex.core.entity_extractor import save_graph

        graph_dir = viz_dir.parent / "benchmarks" / "graph_cache"
        graph_dir.mkdir(parents=True, exist_ok=True)
        save_graph(graph, graph_dir / f"{doc_name}.json")

    # Install deps if needed
    if not (viz_dir / "node_modules").exists():
        subprocess.run(["npm", "install"], cwd=viz_dir, check=True, capture_output=True)

    url = f"http://localhost:{port}/tree?name={doc_name}"
    threading.Timer(2.0, lambda: webbrowser.open(url)).start()

    import os
    try:
        subprocess.run(
            ["npx", "next", "dev", "--port", str(port)],
            cwd=viz_dir,
            env={**os.environ, "PORT": str(port)},
        )
    except KeyboardInterrupt:
        pass
