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
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

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
from nanoindex.models import (
    Answer, DocumentGraph, DocumentTree, ExtractionResult2, RetrievedNode, TreeNode, ValidationResult,
)

__all__ = [
    "NanoIndex",
    "NanoIndexConfig",
    "KnowledgeBase",
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
    "ExtractionResult2",
    "ValidationResult",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "KnowledgeBase":
        from nanoindex.kb import KnowledgeBase
        return KnowledgeBase
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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


def _parse_llm_string(llm: str) -> tuple[str, str | None, str | None]:
    """Parse 'provider:model' or just 'model' into (model, base_url, api_key).

    Examples:
        'openai:gpt-5.4'           → ('gpt-5.4', openai_url, OPENAI_API_KEY)
        'anthropic:claude-sonnet-4-6' → ('claude-sonnet-4-6', None, ANTHROPIC_API_KEY)
        'gemini:gemini-2.5-flash' → ('gemini-2.5-flash', gemini_url, GOOGLE_API_KEY)
        'ollama:llama3'           → ('llama3', 'http://localhost:11434/v1', None)
        'gpt-5.4'                  → ('gpt-5.4', auto-detected, auto-detected)
    """
    import os

    _PROVIDERS = {
        "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
        "anthropic": (None, "ANTHROPIC_API_KEY"),  # uses native SDK
        "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GOOGLE_API_KEY"),
        "google": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GOOGLE_API_KEY"),
        "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
        "ollama": ("http://localhost:11434/v1", None),
        "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
        "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    }

    if ":" in llm:
        provider, model = llm.split(":", 1)
        provider = provider.lower().strip()
        model = model.strip()
        if provider in _PROVIDERS:
            base_url, env_key = _PROVIDERS[provider]
            api_key = os.environ.get(env_key) if env_key else None
            return model, base_url, api_key
        # Unknown provider — treat as base_url
        return model, provider, None

    # No provider prefix — just model name, auto-detect from llm.py
    return llm, None, None


def _auto_detect_nanonets_key() -> str | None:
    """Try to find Nanonets API key from environment."""
    import os
    return os.environ.get("NANONETS_API_KEY")


def _auto_detect_llm() -> str | None:
    """Try to find any LLM API key from env and return a default model."""
    import os
    for env_key, model in [
        ("ANTHROPIC_API_KEY", "claude-sonnet-4-6"),
        ("OPENAI_API_KEY", "gpt-5.4"),
        ("GOOGLE_API_KEY", "gemini-2.5-flash"),
        ("GROQ_API_KEY", "llama-3.3-70b-versatile"),
        ("GOOGLE_API_KEY", "gemini-2.5-flash"),
        ("GROQ_API_KEY", "llama-3.3-70b-versatile"),
    ]:
        if os.environ.get(env_key):
            return model
    return None


class NanoIndex:
    """Main orchestrator — wires extraction, tree building, enrichment,
    retrieval, and generation into a cohesive pipeline.

    Quick start::

        # Everything from env vars
        ni = NanoIndex()

        # Explicit keys
        ni = NanoIndex(nanonets_api_key="...", llm="openai:gpt-5.4")

        # Provider shorthand
        ni = NanoIndex(llm="anthropic:claude-sonnet-4-6")
    """

    def __init__(
        self,
        config: NanoIndexConfig | None = None,
        *,
        llm: str | None = None,
        **kwargs: Any,
    ) -> None:
        # Auto-detect nanonets key from env if not provided
        if "nanonets_api_key" not in kwargs and config is None:
            auto_key = _auto_detect_nanonets_key()
            if auto_key:
                kwargs["nanonets_api_key"] = auto_key

        # Parse llm shorthand: "openai:gpt-5.4" or "claude-sonnet-4-6"
        if llm:
            model, base_url, api_key = _parse_llm_string(llm)
            kwargs.setdefault("reasoning_llm_model", model)
            if base_url:
                kwargs.setdefault("reasoning_llm_base_url", base_url)
            if api_key:
                kwargs.setdefault("reasoning_llm_api_key", api_key)
        elif "reasoning_llm_model" not in kwargs and config is None:
            # Auto-detect LLM from env vars
            auto_llm = _auto_detect_llm()
            if auto_llm:
                kwargs["reasoning_llm_model"] = auto_llm

        self.config = config or load_config(**kwargs)

        # Validate keys early with clear messages
        if not self.config.nanonets_api_key:
            logger.warning(
                "No NANONETS_API_KEY set. You need this for document parsing.\n"
                "  Get a free key (10K pages) at https://docstrange.nanonets.com/app\n"
                "  Then: export NANONETS_API_KEY=your_key"
            )
        if not self.config.reasoning_llm_model:
            logger.warning(
                "No LLM configured. You need this for answering questions.\n"
                "  Set one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY\n"
                "  Or pass: NanoIndex(llm='anthropic:claude-sonnet-4-6')\n"
                "  Indexing will work but ask() requires an LLM."
            )

        self._client = None
        self._llm = None
        self._reasoning_llm = None
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

        Raises ConfigError if no reasoning LLM is configured.
        Will NOT fall back to the Nanonets OCR model.
        """
        if self._reasoning_llm is None:
            cfg = self.config
            # Ensure a reasoning LLM is configured — don't silently use OCR model
            cfg.require_reasoning_llm()
            from nanoindex.core.llm import LLMClient, _auto_detect_key, _auto_detect_url
            key = cfg.reasoning_llm_api_key or _auto_detect_key(cfg.reasoning_llm_model, None) or cfg.require_llm_key()
            url = cfg.reasoning_llm_base_url or _auto_detect_url(cfg.reasoning_llm_model, None)
            self._reasoning_llm = LLMClient(
                api_key=key,
                base_url=url,
                model=cfg.reasoning_llm_model,
            )
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

        # Use the pluggable parser system for all parsers (including nanonets)
        from nanoindex.core.parsers import get_parser
        parser_kwargs: dict[str, Any] = {}
        if self.config.parser == "nanonets":
            parser_kwargs["api_key"] = self.config.require_nanonets_key()
            parser_kwargs["use_v2"] = self.config.use_v2_api
        parser = get_parser(self.config.parser, **parser_kwargs)
        parsed_document = await parser.parse(path)
        extraction = parsed_document.to_extraction_result()

        # Auto-detect document mode if set to "auto"
        if self.config.doc_mode == "auto":
            from nanoindex.core.document_classifier import classify_parsed
            detected_mode = classify_parsed(parsed_document)
            logger.info("Auto-detected document type: %s", detected_mode)
        else:
            detected_mode = self.config.doc_mode

        # Extract tables and forms for non-hierarchical documents
        _extracted_tables = []
        _extracted_form = None
        if detected_mode == "tabular":
            from nanoindex.core.table_extractor import tables_from_markdown
            _extracted_tables = tables_from_markdown(parsed_document.markdown)
            if _extracted_tables:
                logger.info("Extracted %d tables from document", len(_extracted_tables))
        elif detected_mode == "form":
            from nanoindex.core.form_extractor import extract_form_from_markdown
            _extracted_form = extract_form_from_markdown(parsed_document.markdown)
            if _extracted_form and _extracted_form.fields:
                logger.info("Extracted %d form fields", len(_extracted_form.fields))

        tree = build_document_tree(extraction, path.stem, self.config)

        # Tag domain for downstream KB/prompt selection
        if not tree.domain:
            from nanoindex.core.gliner_extractor import _detect_domain
            from nanoindex.utils.tree_ops import iter_nodes
            sample_text = " ".join(n.text or "" for n in list(iter_nodes(tree.structure))[:5])
            tree.domain = _detect_domain(sample_text, doc_name=tree.doc_name)
            logger.info("Document domain: %s", tree.domain)

        # Store metadata on the tree for later use
        if parsed_document is not None:
            tree._parsed_document = parsed_document  # type: ignore[attr-defined]
        tree._doc_mode = detected_mode  # type: ignore[attr-defined]
        if _extracted_tables:
            tree._tables = _extracted_tables  # type: ignore[attr-defined]
        if _extracted_form:
            tree._form = _extracted_form  # type: ignore[attr-defined]

        llm = self._get_llm()

        # Split oversized leaf nodes (heuristic + optional LLM)
        tree = await refine_tree(tree, llm, self.config)

        # Validate tree quality
        from nanoindex.core.tree_validator import validate_tree
        validation = validate_tree(tree)
        tree._validation = validation  # type: ignore[attr-defined]

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

        # Disambiguate repetitive titles (e.g. "Reconciliation" x8)
        from nanoindex.core.title_disambiguator import disambiguate_titles
        tree = disambiguate_titles(tree)

        # Build entity graph (default: on)
        if self.config.build_graph:
            await self.async_build_graph(tree, parsed_document)

        # Build node embeddings (default: on)
        if self.config.build_embeddings:
            await self.async_build_embeddings(tree)

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
        mode: str = "agentic_vision",
        pdf_path: str | Path | None = None,
        include_metadata: bool = False,
        pure_vision: bool = True,
    ) -> Answer:
        """Search + generate an answer in one step.

        Retrieval modes (how candidates are found):
          - ``"fast"``: entity graph pre-filter, then LLM picks from ~20
            candidates. Cheapest, works when graph is built.
          - ``"fast_vision"``: same retrieval, answer uses page images too.
          - ``"agentic"``: LLM navigates the full tree outline in multiple rounds.
            Most thorough, highest accuracy, most expensive.
          - ``"agentic_vision"``: agentic retrieval + page images for answering.
          - ``"agentic_graph"``: graph-seeded agentic — uses entity graph to find
            seed nodes, then agent reasons and expands. Best of both worlds:
            graph precision + agentic depth, fewer LLM calls than pure agentic.
          - ``"agentic_graph_vision"``: same + page images.
          - ``"global"``: community-based map-reduce over entity graph.
            Best for broad questions like "What are the main themes?"
          - ``"text"``: alias for ``"fast"`` (backward compat).
          - ``"vision"``: alias for ``"fast_vision"`` (backward compat).

        When *include_metadata* is ``True``, each citation carries
        bounding boxes and page dimensions for its source pages.
        """
        if tree is None:
            raise ValueError(
                "No document tree provided. Index a document first:\n\n"
                "  tree = ni.index('document.pdf')\n"
                "  answer = ni.ask('your question', tree)\n"
            )

        # Backward-compat aliases
        if mode == "text":
            mode = "fast"
        elif mode == "vision":
            mode = "fast_vision"

        answer: Answer
        if mode == "global":
            from nanoindex.core.community_detector import detect_communities, auto_summarize_community
            graph = self._graphs.get(tree.doc_name)
            if not graph:
                # Fall back to agentic
                mode = "agentic_vision"
            else:
                communities = detect_communities(graph)
                if not communities:
                    mode = "agentic_vision"
                else:
                    # Map: get partial answers from each community
                    summaries = [auto_summarize_community(c, graph) for c in communities]
                    llm = self._get_reasoning_llm()

                    partial_answers = []
                    for i, summary in enumerate(summaries):
                        map_prompt = f"""Based on this group of related entities, answer the question if relevant.
If this group is not relevant, respond with "NOT RELEVANT".

Question: {query}

Entity Group:
{summary}

Answer (or "NOT RELEVANT"):"""
                        try:
                            resp = await llm.chat([{"role": "user", "content": map_prompt}], max_tokens=300)
                            if "not relevant" not in resp.lower():
                                partial_answers.append(resp)
                        except Exception:
                            continue

                    if not partial_answers:
                        mode = "agentic_vision"  # fallback
                    else:
                        # Reduce: combine partial answers
                        reduce_prompt = f"""Combine these partial answers into a comprehensive final answer.

Question: {query}

Partial Answers:
{chr(10).join(f"--- Answer {i+1} ---{chr(10)}{a}" for i, a in enumerate(partial_answers))}

Final comprehensive answer:"""

                        final = await llm.chat([{"role": "user", "content": reduce_prompt}], max_tokens=1024)
                        answer = Answer(content=final, citations=[], mode="global")
                        from nanoindex.core.citation_resolver import resolve_citations
                        return resolve_citations(answer, tree)

        if mode.startswith("fast"):
            from nanoindex.core.fast_retriever import fast_search
            from nanoindex.core.generator import generate_answer
            llm = self._get_reasoning_llm()
            graph = self._graphs.get(tree.doc_name)
            nodes = await fast_search(
                query, tree, llm, self.config,
                graph=graph,
            )
            gen_mode = "vision" if "vision" in mode else "text"
            answer = await generate_answer(
                query, nodes, llm,
                mode=gen_mode, pdf_path=pdf_path,
                tree=tree, include_metadata=include_metadata,
            )
        elif mode.startswith("agentic"):
            from nanoindex.core.agentic import agentic_ask
            llm = self._get_reasoning_llm()
            # For agentic_graph modes, pass the graph for graph-seeded retrieval
            graph = self._graphs.get(tree.doc_name) if "graph" in mode else None
            answer = await agentic_ask(
                query, tree, llm, self.config,
                pdf_path=pdf_path,
                use_vision="vision" in mode,
                pure_vision=pure_vision,
                include_metadata=include_metadata,
                graph=graph,
            )
        else:
            from nanoindex.core.generator import generate_answer
            nodes = await self.async_search(query, tree)
            llm = self._get_reasoning_llm()
            answer = await generate_answer(
                query, nodes, llm,
                mode=mode, pdf_path=pdf_path,
                tree=tree, include_metadata=include_metadata,
            )

        # Resolve citations to exact bounding boxes
        from nanoindex.core.citation_resolver import resolve_citations
        answer = resolve_citations(answer, tree)
        return answer

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

    def get_graph(self, tree: "DocumentTree") -> "DocumentGraph | None":
        """Get the entity graph for a document. Returns None if not built."""
        return self._graphs.get(tree.doc_name)

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

        Uses a hybrid approach:
          1. GLiNER2 zero-shot NER (if installed), else spaCy NLP for base extraction
          2. Entity resolution to merge duplicates
          3. If a reasoning LLM is available, enhances with LLM extraction

        When *parsed* is supplied and contains ``modal_contents``,
        multimodal entities (images, tables, etc.) are also extracted.
        """
        # Step 1: Try GLiNER2 first, fall back to spaCy
        try:
            from nanoindex.core.gliner_extractor import extract_entities_gliner
            graph = extract_entities_gliner(tree)
            logger.info("GLiNER2 graph: %d entities, %d relationships", len(graph.entities), len(graph.relationships))
        except ImportError:
            from nanoindex.core.spacy_extractor import extract_entities_spacy
            graph = extract_entities_spacy(tree)
            logger.info("spaCy graph: %d entities, %d relationships", len(graph.entities), len(graph.relationships))

        # Step 2: Entity resolution (always)
        from nanoindex.core.entity_resolver import resolve_entities
        graph = resolve_entities(graph)

        # Step 2: LLM enhancement (if reasoning LLM configured)
        if self.config.reasoning_llm_model:
            try:
                from nanoindex.core.entity_extractor import extract_entities
                llm = self._get_reasoning_llm()
                llm_graph = await extract_entities(tree, llm)
                if llm_graph.entities:
                    # Merge: add LLM entities not already found by spaCy
                    spacy_names = {e.name.lower() for e in graph.entities}
                    new_entities = [e for e in llm_graph.entities if e.name.lower() not in spacy_names]
                    new_rels = [r for r in llm_graph.relationships
                                if (r.source.lower(), r.target.lower()) not in
                                {(r2.source.lower(), r2.target.lower()) for r2 in graph.relationships}]
                    graph.entities.extend(new_entities)
                    graph.relationships.extend(new_rels)
                    logger.info("LLM enhanced: +%d entities, +%d relationships", len(new_entities), len(new_rels))
            except Exception:
                logger.debug("LLM enhancement failed, using spaCy-only graph", exc_info=True)

        # Step 3: Multimodal entities (if available)
        if parsed is None:
            parsed = getattr(tree, "_parsed_document", None)
        if parsed is not None and parsed.modal_contents and self.config.reasoning_llm_model:
            try:
                from nanoindex.core.entity_extractor import extract_multimodal_entities
                llm = self._get_reasoning_llm()
                mm_entities, mm_relationships = await extract_multimodal_entities(parsed, tree, llm)
                graph.entities.extend(mm_entities)
                graph.relationships.extend(mm_relationships)
            except Exception:
                logger.debug("Multimodal extraction failed", exc_info=True)

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
        mode: str = "agentic_vision",
        pdf_path: str | Path | None = None,
        include_metadata: bool = False,
        pure_vision: bool = True,
    ) -> Answer:
        return _run(self.async_ask(
            query, tree, mode=mode, pdf_path=pdf_path,
            include_metadata=include_metadata, pure_vision=pure_vision,
        ))

    def multi_search(
        self, query: str, store: "DocumentStore", **kwargs: Any,
    ) -> list[RetrievedNode]:
        return _run(self.async_multi_search(query, store, **kwargs))

    def multi_ask(
        self, query: str, store: "DocumentStore", **kwargs: Any,
    ) -> Answer:
        return _run(self.async_multi_ask(query, store, **kwargs))

    def extract(
        self,
        file_path: str | Path,
        *,
        mode: str = "auto",
        schema: dict[str, Any] | None = None,
    ) -> "ExtractionResult2":
        """Extract structured data from a PDF. Returns ExtractionResult2."""
        return _run(self.async_extract(file_path, mode=mode, schema=schema))

    async def async_extract(
        self,
        file_path: str | Path,
        *,
        mode: str = "auto",
        schema: dict[str, Any] | None = None,
    ) -> "ExtractionResult2":
        """Extract structured data from a PDF (async). Returns ExtractionResult2."""
        from nanoindex.extract import extract_document
        return await extract_document(file_path, self, mode=mode, schema=schema)

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
    mode: str = "agentic_vision",
    pdf_path: str | Path | None = None,
    include_metadata: bool = False,
    pure_vision: bool = True,
    **kwargs: Any,
) -> Answer:
    """Index (if needed), search, and answer a question about a document."""
    ni = NanoIndex(**kwargs)
    return ni.ask(
        query, tree, mode=mode, pdf_path=pdf_path,
        include_metadata=include_metadata, pure_vision=pure_vision,
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
