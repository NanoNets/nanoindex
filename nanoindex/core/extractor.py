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

from nanoindex.core.client import NanonetsClient
from nanoindex.exceptions import ExtractionError
from nanoindex.models import (
    BoundingBox,
    ExtractionResult,
    HierarchyKVPair,
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
        import json as _json
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
        import json as _json
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
        resp = await client.poll_result(record_id, max_wait=120.0)
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
# Public API
# ------------------------------------------------------------------

async def extract_document(
    file_path: str | Path,
    client: NanonetsClient,
    *,
    concurrency: int = _DEFAULT_CONCURRENCY,
) -> ExtractionResult:
    """Extract a document, automatically choosing the best strategy.

    * **<=5 pages** — one combined sync call (markdown + bboxes + TOC)
    * **>5 pages** — page-parallel markdown + concurrent async TOC
    """
    path = Path(file_path)
    if not path.exists():
        raise ExtractionError(f"File not found: {path}")

    from nanoindex.utils.pdf import get_page_count
    pages = get_page_count(path)
    logger.info("Document has %d pages", pages)

    if pages <= 5:
        return await _extract_small(path, client)
    return await _extract_large(path, client, concurrency)
