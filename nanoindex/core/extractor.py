"""Orchestrates Nanonets extraction calls and normalises the output.

Extraction strategies:

* **Small documents (<=5 pages)** — one combined sync call that returns
  markdown + bounding boxes + TOC in a single request.
* **Large documents (>5 pages)** — two concurrent tracks that run in
  parallel:
    1. Page-parallel sync: split PDF into single pages, extract each via
       the sync endpoint in batches of ``concurrency`` (default 10).
       Produces markdown + bounding boxes.
    2. Async TOC: send the full document to the async endpoint for
       ``table-of-contents`` extraction.  Produces a rich, multi-level
       hierarchy with page numbers.
  Total time is ``max(page_parallel, async_toc)`` — typically ~90 s.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Any

import json as _json

from nanoindex.core.client import NanonetsClient
from nanoindex.exceptions import ExtractionError
from nanoindex.models import (
    BoundingBox,
    ExtractionResult,
    HierarchyEntity,
    HierarchyKVPair,
    HierarchyRelationship,
    HierarchySection,
    HierarchyTable,
    PageDimensions,
    TOCEntry,
)

logger = logging.getLogger(__name__)

_DEFAULT_CONCURRENCY = 25


# ------------------------------------------------------------------
# Response parsers
# ------------------------------------------------------------------

def _parse_markdown_response(resp: dict[str, Any]) -> tuple[str, list[BoundingBox], list[PageDimensions]]:
    """Extract markdown text, bounding boxes, and page dims from a response."""
    markdown = ""
    bboxes: list[BoundingBox] = []
    page_dims: list[PageDimensions] = []

    result = resp.get("result") or resp
    if isinstance(result, list):
        result = result[0] if result else {}

    md_obj = result.get("markdown") or {}
    if isinstance(md_obj, str):
        markdown = md_obj
    elif isinstance(md_obj, dict):
        markdown = md_obj.get("content", "") or ""
        metadata = md_obj.get("metadata") or {}
        bb_meta = metadata.get("bounding_boxes") or {}

        for el in bb_meta.get("elements", []):
            bb = el.get("bounding_box") or el
            bboxes.append(BoundingBox(
                page=bb.get("page", 1),
                x=bb.get("x", 0),
                y=bb.get("y", 0),
                width=bb.get("width", 0),
                height=bb.get("height", 0),
                confidence=bb.get("confidence", 1.0),
                region_type=bb.get("type", "unknown"),
                text=el.get("content") or bb.get("text"),
            ))

        pd_meta = bb_meta.get("page_dimensions") or {}
        for pg in pd_meta.get("pages", []):
            page_dims.append(PageDimensions(
                page=pg.get("page", 1),
                width=pg.get("width", 0),
                height=pg.get("height", 0),
            ))
    else:
        markdown = str(md_obj)

    if not markdown:
        markdown = result.get("text", "") or ""

    return markdown, bboxes, page_dims


def _parse_toc_response(resp: dict[str, Any]) -> list[TOCEntry]:
    """Parse the ``table-of-contents`` JSON into a list of ``TOCEntry``."""
    result = resp.get("result") or resp
    if isinstance(result, list):
        result = result[0] if result else {}

    json_obj = result.get("json") or {}
    content = json_obj.get("content") or json_obj if isinstance(json_obj, dict) else {}

    if isinstance(content, str):
        try:
            content = _json.loads(content)
        except (ValueError, TypeError):
            return []

    hierarchy = content.get("hierarchy", [])
    entries: list[TOCEntry] = []
    for h in hierarchy:
        entries.append(TOCEntry(
            id=h.get("id", ""),
            title=h.get("title", ""),
            level=h.get("level", 1),
            page=h.get("page", 0),
            parent_ids=h.get("parent_ids", []),
        ))
    return entries


def _parse_hierarchy_response(
    resp: dict[str, Any],
) -> tuple[list[HierarchySection], list[HierarchyTable], list[HierarchyKVPair]]:
    """Parse the ``hierarchy_output`` JSON into typed models."""
    result = resp.get("result") or resp
    if isinstance(result, list):
        result = result[0] if result else {}

    json_obj = result.get("json") or {}
    if isinstance(json_obj, dict):
        content = json_obj.get("content") or json_obj
    else:
        content = result

    if isinstance(content, str):
        try:
            content = _json.loads(content)
        except (ValueError, TypeError):
            return [], [], []

    doc = content.get("document", content)

    sections = _parse_sections(doc.get("sections", []))
    tables = [
        HierarchyTable(
            id=t.get("id", ""),
            title=t.get("title", ""),
            headers=t.get("headers", []),
            rows=t.get("rows", []),
        )
        for t in doc.get("tables", [])
    ]
    kv_pairs = [
        HierarchyKVPair(key=kv.get("key", ""), value=kv.get("value", ""))
        for kv in doc.get("key_value_pairs", [])
    ]
    return sections, tables, kv_pairs


def _parse_sections(raw: list[dict[str, Any]]) -> list[HierarchySection]:
    sections: list[HierarchySection] = []
    for s in raw:
        section = HierarchySection(
            id=s.get("id", ""),
            title=s.get("title") or s.get("heading", ""),
            level=s.get("level", 1),
            content=s.get("content", ""),
            subsections=_parse_sections(s.get("subsections", [])),
        )
        sections.append(section)
    return sections


# ------------------------------------------------------------------
# Hierarchy v2 (beta pipeline) response parser
# ------------------------------------------------------------------

def _parse_hierarchy_v2_sections(raw: list[dict[str, Any]]) -> list[HierarchySection]:
    """Recursively parse sections from the beta hierarchy_output API."""
    sections: list[HierarchySection] = []
    for s in raw:
        # Parse inline tables
        tables = [
            HierarchyTable(
                id=t.get("id", ""),
                title=t.get("title", ""),
                headers=t.get("headers", []),
                rows=t.get("rows", []),
                page=t.get("page", 0),
                section_id=t.get("section_id", ""),
                bounding_box=t.get("bounding_box"),
            )
            for t in s.get("tables", [])
        ]

        # Parse inline KV pairs
        kv_pairs = [
            HierarchyKVPair(
                key=kv.get("key", ""),
                value=str(kv.get("value", "")),
                page=kv.get("page", 0),
                section_id=kv.get("section_id", ""),
                bounding_box=kv.get("bounding_box"),
            )
            for kv in s.get("key_value_pairs", [])
        ]

        # Parse inline entities
        entities = [
            HierarchyEntity(
                name=e.get("name", ""),
                entity_type=e.get("type", "Other"),
                value=e.get("value", ""),
            )
            for e in s.get("entities", [])
        ]

        # Parse inline relationships — normalise both API formats
        relationships = []
        for r in s.get("relationships", []):
            src = r.get("source") or r.get("subject") or ""
            tgt = r.get("target") or r.get("object") or ""
            rel_type = r.get("type") or r.get("relationship") or r.get("predicate") or "related_to"
            if rel_type == "?":
                rel_type = "related_to"
            if src and tgt:
                relationships.append(HierarchyRelationship(
                    source=src, target=tgt, rel_type=rel_type,
                ))

        section = HierarchySection(
            id=s.get("id", ""),
            title=s.get("title") or s.get("heading", ""),
            level=s.get("level", 1),
            content=s.get("content", ""),
            aggregated_content=s.get("aggregated_content", ""),
            summary=s.get("summary", ""),
            page=s.get("page", 0),
            end_page=s.get("end_page", 0),
            subsections=_parse_hierarchy_v2_sections(s.get("subsections", [])),
            tables=tables,
            key_value_pairs=kv_pairs,
            entities=entities,
            relationships=relationships,
            title_bounding_box=s.get("title_bounding_box"),
            content_bounding_box=s.get("content_bounding_box"),
        )
        sections.append(section)
    return sections


def _parse_hierarchy_v2_response(resp: dict[str, Any]) -> ExtractionResult:
    """Parse the full response from the beta hierarchy_output API.

    Extracts sections (with inline tables, KVs, entities, relationships,
    summaries, bboxes), per-page markdown, and page dimensions.
    """
    result = resp.get("result") or resp
    if isinstance(result, list):
        result = result[0] if result else {}

    json_obj = result.get("json") or {}
    content = json_obj.get("content") or {} if isinstance(json_obj, dict) else {}
    doc = content.get("document", {}) if isinstance(content, dict) else {}

    # -- Sections (recursive with all inline data) -----------------------
    sections = _parse_hierarchy_v2_sections(doc.get("sections", []))

    # -- Per-page markdown -----------------------------------------------
    page_markdowns: list[str] = []
    raw_pages = doc.get("pages", [])
    if isinstance(raw_pages, list):
        # Sort by page number, fill gaps with empty strings
        page_map: dict[int, str] = {}
        for p in raw_pages:
            if isinstance(p, dict):
                pnum = p.get("page", 0)
                md = p.get("raw_markdown", "") or p.get("markdown", "") or ""
                if pnum:
                    page_map[pnum] = md
        if page_map:
            max_page = max(page_map.keys())
            page_markdowns = [page_map.get(i, "") for i in range(1, max_page + 1)]

    # Assemble full markdown from per-page
    markdown = "\n\n".join(page_markdowns) if page_markdowns else ""

    # -- Bounding boxes from section-level data --------------------------
    bboxes: list[BoundingBox] = []
    page_dims_set: dict[int, PageDimensions] = {}

    def _collect_bboxes(secs: list[HierarchySection]) -> None:
        for sec in secs:
            for bb_data in [sec.title_bounding_box, sec.content_bounding_box]:
                if not bb_data or not isinstance(bb_data, dict):
                    continue
                pg = bb_data.get("page", sec.page or 0)
                bboxes.append(BoundingBox(
                    page=pg,
                    x=bb_data.get("x", 0),
                    y=bb_data.get("y", 0),
                    width=bb_data.get("width", 0),
                    height=bb_data.get("height", 0),
                    confidence=bb_data.get("confidence", 1.0),
                    region_type="heading" if bb_data is sec.title_bounding_box else "content",
                    text=sec.title if bb_data is sec.title_bounding_box else None,
                ))
                img_dims = bb_data.get("image_dimensions") or {}
                if img_dims and pg and pg not in page_dims_set:
                    page_dims_set[pg] = PageDimensions(
                        page=pg,
                        width=img_dims.get("width", 0),
                        height=img_dims.get("height", 0),
                    )
            _collect_bboxes(sec.subsections)

    _collect_bboxes(sections)
    page_dims = sorted(page_dims_set.values(), key=lambda d: d.page)

    # -- Page count ------------------------------------------------------
    page_count = resp.get("pages_processed") or 0
    if not page_count and page_markdowns:
        page_count = len(page_markdowns)
    if not page_count and page_dims:
        page_count = max(d.page for d in page_dims)

    processing_time = resp.get("processing_time") or 0
    if isinstance(processing_time, str):
        processing_time = float(processing_time.rstrip("s"))

    return ExtractionResult(
        markdown=markdown,
        page_markdowns=page_markdowns,
        hierarchy_sections=sections,
        bounding_boxes=bboxes,
        page_dimensions=page_dims,
        page_count=page_count,
        processing_time=round(float(processing_time), 2),
        record_id=str(resp.get("record_id", "")),
    )


# ------------------------------------------------------------------
# Single-page extraction helper (for page-parallel)
# ------------------------------------------------------------------

async def _extract_single_page(
    page_num: int,
    page_bytes: bytes,
    client: NanonetsClient,
    sem: asyncio.Semaphore,
) -> tuple[int, dict[str, Any]]:
    """Extract one page, respecting the concurrency semaphore."""
    async with sem:
        logger.debug("Extracting page %d …", page_num)
        resp = await client.extract_sync_bytes(
            page_bytes,
            filename=f"page_{page_num}.pdf",
            output_format="markdown",
            include_metadata="confidence_score,bounding_boxes",
        )
        return page_num, resp


_PAGE_HEADER_RE = re.compile(r"^## Page \d+\s*$", re.MULTILINE)


def _remap_page_result(
    page_num: int,
    resp: dict[str, Any],
) -> tuple[str, list[BoundingBox], list[PageDimensions]]:
    """Parse a single-page response and remap page references to *page_num*."""
    md, bboxes, dims = _parse_markdown_response(resp)
    md = _PAGE_HEADER_RE.sub(f"<!-- nanoindex:page:{page_num} -->", md, count=1)
    for bb in bboxes:
        bb.page = page_num
    for pd in dims:
        pd.page = page_num
    return md, bboxes, dims


# ------------------------------------------------------------------
# Strategy A: Small documents (<=5 pages) — one combined sync call
# ------------------------------------------------------------------

async def _extract_small(
    file_path: Path,
    client: NanonetsClient,
) -> ExtractionResult:
    """Single combined sync call: markdown + bboxes + TOC."""
    t0 = time.monotonic()

    resp = await client.extract_sync(
        file_path,
        output_format="markdown,json",
        json_options="table-of-contents",
        include_metadata="confidence_score,bounding_boxes",
    )

    markdown, bboxes, page_dims = _parse_markdown_response(resp)
    toc = _parse_toc_response(resp)

    page_count = 0
    if page_dims:
        page_count = max(pd.page for pd in page_dims)
    elif bboxes:
        page_count = max(bb.page for bb in bboxes)
    pp = resp.get("pages_processed")
    if pp and pp > page_count:
        page_count = pp

    elapsed = time.monotonic() - t0
    logger.info("Small-doc extraction done in %.1fs — %d chars, %d TOC entries",
                elapsed, len(markdown), len(toc))

    return ExtractionResult(
        markdown=markdown,
        toc=toc,
        bounding_boxes=bboxes,
        page_dimensions=page_dims,
        page_count=page_count,
        processing_time=round(elapsed, 2),
    )


# ------------------------------------------------------------------
# Strategy B: Large documents (>5 pages) — hybrid parallel + async TOC
# ------------------------------------------------------------------

async def _page_parallel_markdown(
    file_path: Path,
    client: NanonetsClient,
    concurrency: int,
) -> tuple[str, list[BoundingBox], list[PageDimensions], int, int]:
    """Page-parallel markdown+bboxes extraction. Returns
    (merged_markdown, bboxes, page_dims, total_pages, failed_count)."""
    from nanoindex.utils.pdf import split_pdf_pages

    pages = split_pdf_pages(file_path)
    total_pages = len(pages)
    logger.info("Page-parallel: %d pages, concurrency=%d", total_pages, concurrency)

    sem = asyncio.Semaphore(concurrency)
    tasks = [
        _extract_single_page(pn, pb, client, sem)
        for pn, pb in pages
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_md: list[tuple[int, str]] = []
    all_bboxes: list[BoundingBox] = []
    all_dims: list[PageDimensions] = []
    failed = 0

    for res in results:
        if isinstance(res, Exception):
            logger.warning("Page extraction failed: %s", res)
            failed += 1
            continue
        page_num, resp = res
        md, bboxes, dims = _remap_page_result(page_num, resp)
        all_md.append((page_num, md))
        all_bboxes.extend(bboxes)
        all_dims.extend(dims)

    if failed == total_pages:
        raise ExtractionError("All page extractions failed")

    all_md.sort(key=lambda x: x[0])
    merged = "\n\n".join(md for _, md in all_md)

    return merged, all_bboxes, all_dims, total_pages, failed


async def _async_toc(
    file_path: Path,
    client: NanonetsClient,
) -> list[TOCEntry]:
    """Submit full document for async TOC extraction and poll until done."""
    try:
        record_id = await client.extract_async(
            file_path,
            output_format="json",
            json_options="table-of-contents",
        )
        resp = await client.poll_result(record_id, max_wait=6000.0)
        return _parse_toc_response(resp)
    except Exception as exc:
        logger.warning("Async TOC extraction failed: %s", exc)
        return []


async def _extract_large(
    file_path: Path,
    client: NanonetsClient,
    concurrency: int = _DEFAULT_CONCURRENCY,
) -> ExtractionResult:
    """Hybrid: page-parallel markdown + concurrent async TOC."""
    t0 = time.monotonic()

    md_task = _page_parallel_markdown(file_path, client, concurrency)
    toc_task = _async_toc(file_path, client)

    (markdown, bboxes, page_dims, total_pages, failed), toc = await asyncio.gather(
        md_task, toc_task,
    )

    elapsed = time.monotonic() - t0
    logger.info(
        "Hybrid extraction done in %.1fs — %d/%d pages, %d TOC entries",
        elapsed, total_pages - failed, total_pages, len(toc),
    )

    return ExtractionResult(
        markdown=markdown,
        toc=toc,
        bounding_boxes=bboxes,
        page_dimensions=page_dims,
        page_count=total_pages,
        processing_time=round(elapsed, 2),
    )


# ------------------------------------------------------------------
# Strategy C: Hierarchy API (beta pipeline) — single async call
# ------------------------------------------------------------------

async def _extract_hierarchy(
    file_path: Path,
    client: NanonetsClient,
    *,
    financial_doc: bool = False,
    include_summaries: bool = True,
    include_entities: bool = True,
) -> ExtractionResult:
    """Single-call extraction using the beta hierarchy_output API.

    Returns sections with content, tables, KV pairs, bounding boxes,
    summaries, entities, and relationships — all in one request.
    """
    t0 = time.monotonic()

    extraction_options = {
        "pipeline": "beta",
        "financial_doc": financial_doc,
        "include_summaries": include_summaries,
        "include_entities": include_entities,
    }

    record_id = await client.extract_async(
        file_path,
        output_format="json",
        json_options="hierarchy_output",
        include_metadata="bounding_boxes",
        extraction_options=extraction_options,
    )
    resp = await client.poll_result(record_id, max_wait=6000.0)
    result = _parse_hierarchy_v2_response(resp)

    elapsed = time.monotonic() - t0
    logger.info(
        "Hierarchy extraction done in %.1fs — %d pages, %d sections",
        elapsed, result.page_count, len(result.hierarchy_sections),
    )
    result.processing_time = round(elapsed, 2)
    return result


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

async def extract_document(
    file_path: str | Path,
    client: NanonetsClient,
    *,
    concurrency: int = _DEFAULT_CONCURRENCY,
    use_hierarchy: bool = False,
    financial_doc: bool = False,
    include_summaries: bool = True,
    include_entities: bool = True,
) -> ExtractionResult:
    """Extract a document, automatically choosing the best strategy.

    * **use_hierarchy=True** — single hierarchy API call (beta, preferred)
    * **<=5 pages** — one combined sync call (markdown + bboxes + TOC)
    * **>5 pages** — page-parallel markdown + concurrent async TOC
    """
    path = Path(file_path)
    if not path.exists():
        raise ExtractionError(f"File not found: {path}")

    if use_hierarchy:
        return await _extract_hierarchy(
            path, client,
            financial_doc=financial_doc,
            include_summaries=include_summaries,
            include_entities=include_entities,
        )

    from nanoindex.utils.pdf import get_page_count
    pages = get_page_count(path)
    logger.info("Document has %d pages", pages)

    if pages <= 5:
        return await _extract_small(path, client)
    return await _extract_large(path, client, concurrency)
