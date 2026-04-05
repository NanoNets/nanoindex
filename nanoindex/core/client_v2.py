"""Async client for the Nanonets V2 REST API.

Provides typed methods for file upload, parsing, extraction, and classification
using the V2 endpoint structure (upload file first, then operate on file_id).
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

import httpx

from nanoindex.exceptions import ExtractionError, RateLimitError
from nanoindex.models import BoundingBox, PageDimensions, ParsedDocument

logger = logging.getLogger(__name__)

_BASE_URL = "https://extraction-api.nanonets.com"
_UPLOAD_TIMEOUT = 30.0
_OPERATION_TIMEOUT = 300.0
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0


class NanonetsV2Client:
    """Async client for the Nanonets V2 document APIs."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = _BASE_URL,
        upload_timeout: float = _UPLOAD_TIMEOUT,
        operation_timeout: float = _OPERATION_TIMEOUT,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._upload_timeout = upload_timeout
        self._operation_timeout = operation_timeout
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Client lifecycle
    # ------------------------------------------------------------------

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=httpx.Timeout(self._operation_timeout, connect=30.0),
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying httpx client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Retry logic
    # ------------------------------------------------------------------

    async def _request_with_retry(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        client = await self._ensure_client()
        effective_timeout = timeout or self._operation_timeout
        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                logger.info("V2 %s %s (attempt %d)", method, path, attempt + 1)

                req_timeout = httpx.Timeout(effective_timeout, connect=30.0)
                if json_body is not None:
                    resp = await client.request(method, path, json=json_body, timeout=req_timeout)
                else:
                    resp = await client.request(method, path, data=data, files=files, timeout=req_timeout)

                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", _BACKOFF_BASE ** (attempt + 1)))
                    if attempt < _MAX_RETRIES - 1:
                        logger.warning("Rate limited (429), retrying in %.1fs …", retry_after)
                        await asyncio.sleep(retry_after)
                        continue
                    raise RateLimitError(retry_after=retry_after)

                if resp.status_code >= 500:
                    if attempt < _MAX_RETRIES - 1:
                        wait = _BACKOFF_BASE ** (attempt + 1)
                        logger.warning("Server error %d, retrying in %.1fs …", resp.status_code, wait)
                        await asyncio.sleep(wait)
                        continue

                if resp.status_code in (401, 403):
                    raise ExtractionError(
                        "Invalid or expired NANONETS_API_KEY. "
                        "Get a free key at https://docstrange.nanonets.com/app"
                    )

                if resp.status_code == 400:
                    body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                    detail = body.get("detail", resp.text[:200])
                    raise ExtractionError(f"Bad request: {detail}")

                resp.raise_for_status()
                return resp

            except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadTimeout) as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    wait = _BACKOFF_BASE ** (attempt + 1)
                    logger.warning("Connection error: %s. Retrying in %.1fs …", type(exc).__name__, wait)
                    await asyncio.sleep(wait)
                    continue
                raise ExtractionError(
                    f"Connection to Nanonets API failed after {_MAX_RETRIES} attempts. "
                    f"Check your internet connection. Error: {type(exc).__name__}"
                ) from exc

        raise ExtractionError(f"Request failed after {_MAX_RETRIES} attempts") from last_exc

    # ------------------------------------------------------------------
    # File upload
    # ------------------------------------------------------------------

    async def upload(self, file_path: str | Path) -> str:
        """Upload a file and return its file_id (e.g. ``file://uuid``).

        POST /api/v2/files (multipart/form-data)
        """
        path = Path(file_path)
        logger.info("Uploading %s to V2 API", path.name)

        with open(path, "rb") as f:
            files = {"file": (path.name, f, "application/octet-stream")}
            resp = await self._request_with_retry(
                "POST", "/api/v2/files", files=files, timeout=self._upload_timeout,
            )

        data = resp.json()
        file_id = data.get("file_id") or data.get("id", "")
        if not file_id:
            raise ExtractionError(f"No file_id in upload response: {data}")
        logger.info("Uploaded %s -> %s", path.name, file_id)
        return file_id

    # ------------------------------------------------------------------
    # Parse
    # ------------------------------------------------------------------

    async def parse(
        self,
        file_id: str,
        output_format: str = "markdown",
        include_metadata: str = "confidence_score,bounding_boxes",
    ) -> dict[str, Any]:
        """Parse an uploaded file. Auto-selects sync or async based on page count.

        Sync: POST /api/v2/parse/sync (<=5 pages)
        Async: POST /api/v2/parse/async + poll (>5 pages)
        """
        body = {
            "input": file_id,
            "parse_config": {
                "output_format": output_format,
                "include_metadata": include_metadata,
            },
        }
        logger.info("Parsing %s (format=%s)", file_id, output_format)

        # Try sync first, fall back to async if too many pages
        try:
            resp = await self._request_with_retry("POST", "/api/v2/parse/sync", json_body=body)
            return resp.json()
        except ExtractionError as exc:
            if "exceeding the maximum limit" not in str(exc) and "maximum limit" not in str(exc):
                raise
            logger.info("Document too large for sync parse, switching to async")

        # Async parse: queue then poll
        resp = await self._request_with_retry("POST", "/api/v2/parse/async", json_body=body)
        data = resp.json()
        record_id = data.get("record_id")
        if not record_id:
            raise ExtractionError(f"No record_id in async parse response: {data}")

        return await self._poll_result(record_id)

    async def _poll_result(self, record_id: str, timeout: float = 1800.0) -> dict[str, Any]:
        """Poll for async job completion."""
        # V2 async jobs use the V1 results endpoint for polling
        url = f"/api/v1/extract/results/{record_id}"
        start = time.monotonic()
        poll_interval = 2.0

        while time.monotonic() - start < timeout:
            resp = await self._request_with_retry("GET", url)
            data = resp.json()
            status = data.get("status", "")

            if status == "completed":
                logger.info("Async job %s completed (%.1fs)", record_id, time.monotonic() - start)
                return data
            if status == "failed":
                raise ExtractionError(f"Async job {record_id} failed: {data.get('message', '')}")

            logger.debug("Job %s status: %s, polling in %.1fs", record_id, status, poll_interval)
            await asyncio.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.3, 10.0)

        raise ExtractionError(f"Polling timed out after {timeout}s for job {record_id}")

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    async def extract_json(self, file_id: str, fields: list[str]) -> dict[str, Any]:
        """Extract structured JSON fields from an uploaded file.

        POST /api/v2/extract/sync
        """
        body = {
            "input": file_id,
            "extraction_config": {
                "output_format": "json",
                "json_options": fields,
            },
        }
        logger.info("Extracting JSON from %s (fields=%s)", file_id, fields)
        resp = await self._request_with_retry("POST", "/api/v2/extract/sync", json_body=body)
        return resp.json()

    async def extract_csv(self, file_id: str) -> dict[str, Any]:
        """Extract tabular CSV data from an uploaded file.

        POST /api/v2/extract/sync
        """
        body = {
            "input": file_id,
            "extraction_config": {
                "output_format": "csv",
                "csv_options": "table",
            },
        }
        logger.info("Extracting CSV from %s", file_id)
        resp = await self._request_with_retry("POST", "/api/v2/extract/sync", json_body=body)
        return resp.json()

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    async def classify(
        self,
        file_id: str,
        categories: list[dict[str, Any]],
        mode: str = "split",
    ) -> dict[str, Any]:
        """Classify an uploaded file into categories.

        POST /api/v2/classify/sync
        """
        body = {
            "input": file_id,
            "classification_config": {
                "categories": categories,
                "mode": mode,
            },
        }
        logger.info("Classifying %s (mode=%s, %d categories)", file_id, mode, len(categories))
        resp = await self._request_with_retry("POST", "/api/v2/classify/sync", json_body=body)
        return resp.json()

    # ------------------------------------------------------------------
    # Convenience: upload + parse -> ParsedDocument
    # ------------------------------------------------------------------

    async def parse_to_document(self, file_path: str | Path) -> ParsedDocument:
        """Upload a file, parse it, and return a fully populated :class:`ParsedDocument`.

        This is the primary convenience method for obtaining bounding boxes and
        structured markdown from the V2 API in a single call.
        """
        start = time.monotonic()

        file_id = await self.upload(file_path)
        result = await self.parse(file_id, include_metadata="confidence_score,bounding_boxes")

        elapsed = time.monotonic() - start

        # Navigate the response structure — handle V2 sync and V1 poll formats
        #
        # V2 sync: result.content = "markdown string"
        #          result.elements = [{bounding_box: {...}, content: "text"}, ...]
        #
        # V1 poll: result.markdown = {"content": "markdown string",
        #                             "metadata": {"bounding_boxes": {...}}}
        #          result.html / result.json / result.csv also present
        #
        result_data = result.get("result", {}) or {}

        content = ""

        # Try V2 sync format first: result.content is a string
        if isinstance(result_data.get("content"), str):
            content = result_data["content"]
        # V1 poll format: result.markdown is a dict with "content" key
        elif isinstance(result_data.get("markdown"), dict):
            md_obj = result_data["markdown"]
            content = md_obj.get("content", "")
            # Extract elements + page_dimensions from metadata.bounding_boxes
            md_meta = md_obj.get("metadata", {})
            if isinstance(md_meta, dict) and "bounding_boxes" in md_meta:
                bb_data = md_meta["bounding_boxes"]
                if isinstance(bb_data, dict):
                    # Elements are at metadata.bounding_boxes.elements
                    v1_elements = bb_data.get("elements", [])
                    if v1_elements:
                        result_data["_v1_elements"] = v1_elements
                    # Page dimensions at metadata.bounding_boxes.page_dimensions.pages
                    v1_pages = bb_data.get("page_dimensions", {})
                    if isinstance(v1_pages, dict) and "pages" in v1_pages:
                        result_data["_v1_page_dims"] = v1_pages["pages"]
        # V1 poll: result.markdown is a plain string
        elif isinstance(result_data.get("markdown"), str):
            content = result_data["markdown"]
        # Fallback: top-level content
        elif isinstance(result.get("content"), str):
            content = result["content"]

        # Get elements from V2 sync or V1 poll metadata
        elements = result_data.get("elements", []) or result_data.get("_v1_elements", [])
        page_count = result_data.get("page_count", 0) or result.get("pages_processed", 0)

        # Build bounding boxes from elements
        bounding_boxes: list[BoundingBox] = []
        seen_pages: dict[int, tuple[int, int]] = {}

        for elem in elements:
            # V2 elements can have bbox nested under "bounding_box" or flat
            bb_data = elem.get("bounding_box", elem)
            raw_page = bb_data.get("page", elem.get("page", 0))
            # V2 pages are 1-indexed already if from bounding_box object
            page = raw_page if raw_page >= 1 else raw_page + 1

            # Text content is in elem.content, not in bbox.text
            text_content = elem.get("content", "") or bb_data.get("text", "")
            if text_content == "text":  # V2 puts literal "text" as type, not content
                text_content = elem.get("content", "")

            bbox = BoundingBox(
                page=page,
                x=float(bb_data.get("x", 0)),
                y=float(bb_data.get("y", 0)),
                width=float(bb_data.get("width", 0)),
                height=float(bb_data.get("height", 0)),
                confidence=float(bb_data.get("confidence", 1.0)),
                region_type=bb_data.get("type", elem.get("type", "unknown")),
                text=text_content,
            )
            bounding_boxes.append(bbox)

            # Track page dimensions from image_dimensions or element data
            img_dims = bb_data.get("image_dimensions", {})
            if page not in seen_pages:
                page_w = int(img_dims.get("width", 0)) or int(elem.get("page_width", 0))
                page_h = int(img_dims.get("height", 0)) or int(elem.get("page_height", 0))
                if page_w and page_h:
                    seen_pages[page] = (page_w, page_h)

        # Build page dimensions from V1 metadata, element data, or page_confidence
        page_dimensions: list[PageDimensions] = []

        # V1 poll: explicit page dimensions from metadata
        v1_page_dims = result_data.get("_v1_page_dims", [])
        for pd in v1_page_dims:
            if isinstance(pd, dict):
                pg = pd.get("page", 0)
                if pg not in seen_pages:
                    seen_pages[pg] = (int(pd.get("width", 0)), int(pd.get("height", 0)))

        page_confidence = result_data.get("page_confidence", {})

        if page_confidence and not seen_pages:
            # Infer page count from page_confidence keys
            for page_key in page_confidence:
                try:
                    pg = int(page_key)
                    if pg not in seen_pages:
                        seen_pages[pg] = (0, 0)
                except (ValueError, TypeError):
                    pass

        for pg, (w, h) in sorted(seen_pages.items()):
            page_dimensions.append(PageDimensions(page=pg, width=w, height=h))

        if not page_count and seen_pages:
            page_count = max(seen_pages.keys()) + 1

        logger.info(
            "Parsed %s: %d pages, %d bounding boxes in %.1fs",
            Path(file_path).name, page_count, len(bounding_boxes), elapsed,
        )

        # Final safety: markdown MUST be a string, period.
        while isinstance(content, dict):
            content = content.get("content", "") or content.get("markdown", "") or content.get("text", "") or ""
        if not isinstance(content, str):
            content = str(content) if content else ""

        return ParsedDocument(
            markdown=content,
            page_count=page_count,
            bounding_boxes=bounding_boxes,
            page_dimensions=page_dimensions,
            processing_time=elapsed,
            parser_name="nanonets_v2",
        )
