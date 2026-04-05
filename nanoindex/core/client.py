"""Thin async wrapper around the Nanonets REST API.

Handles extraction (sync/async/streaming), classification, and polling.
No business logic — just HTTP call translation with retries and error mapping.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, AsyncIterator

import httpx

from nanoindex.exceptions import ExtractionError, RateLimitError

logger = logging.getLogger(__name__)

_BASE_URL = "https://extraction-api.nanonets.com"
_DEFAULT_TIMEOUT = 120.0
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0
_POLL_INTERVAL = 2.0


class NanonetsClient:
    """Async client for the Nanonets document APIs."""

    def __init__(self, api_key: str, *, base_url: str = _BASE_URL, timeout: float = _DEFAULT_TIMEOUT) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=httpx.Timeout(self._timeout, connect=30.0),
            )
        return self._client

    async def close(self) -> None:
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
        data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> httpx.Response:
        client = await self._ensure_client()
        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                if json_body is not None:
                    resp = await client.request(method, path, json=json_body)
                else:
                    resp = await client.request(method, path, data=data, files=files)

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
    # Extraction endpoints
    # ------------------------------------------------------------------

    def _build_extract_fields(
        self,
        output_format: str = "markdown",
        json_options: str | None = None,
        include_metadata: str | None = None,
        custom_instructions: str | None = None,
    ) -> dict[str, str]:
        fields: dict[str, str] = {"output_format": output_format}
        if json_options:
            fields["json_options"] = json_options
        if include_metadata:
            fields["include_metadata"] = include_metadata
        if custom_instructions:
            fields["custom_instructions"] = custom_instructions
        return fields

    async def extract_sync(
        self,
        file_path: str | Path,
        *,
        output_format: str = "markdown",
        json_options: str | None = None,
        include_metadata: str | None = None,
        custom_instructions: str | None = None,
    ) -> dict[str, Any]:
        """Synchronous extraction — blocks until result is ready."""
        path = Path(file_path)
        fields = self._build_extract_fields(output_format, json_options, include_metadata, custom_instructions)

        with open(path, "rb") as f:
            files = {"file": (path.name, f, "application/octet-stream")}
            resp = await self._request_with_retry("POST", "/api/v1/extract/sync", data=fields, files=files)

        return resp.json()

    async def extract_sync_bytes(
        self,
        file_bytes: bytes,
        filename: str = "page.pdf",
        *,
        output_format: str = "markdown",
        json_options: str | None = None,
        include_metadata: str | None = None,
    ) -> dict[str, Any]:
        """Synchronous extraction from in-memory bytes (single-page PDFs)."""
        fields = self._build_extract_fields(output_format, json_options, include_metadata)
        files = {"file": (filename, file_bytes, "application/pdf")}
        resp = await self._request_with_retry("POST", "/api/v1/extract/sync", data=fields, files=files)
        return resp.json()

    async def extract_async(
        self,
        file_path: str | Path,
        *,
        output_format: str = "markdown",
        json_options: str | None = None,
        include_metadata: str | None = None,
        custom_instructions: str | None = None,
    ) -> str:
        """Asynchronous extraction — returns a ``record_id`` for polling."""
        path = Path(file_path)
        fields = self._build_extract_fields(output_format, json_options, include_metadata, custom_instructions)

        with open(path, "rb") as f:
            files = {"file": (path.name, f, "application/octet-stream")}
            resp = await self._request_with_retry("POST", "/api/v1/extract/async", data=fields, files=files)

        data = resp.json()
        record_id = data.get("record_id") or data.get("id", "")
        if not record_id:
            raise ExtractionError(f"No record_id in async response: {data}")
        return record_id

    async def poll_result(
        self,
        record_id: str,
        *,
        interval: float = _POLL_INTERVAL,
        max_wait: float = 300.0,
    ) -> dict[str, Any]:
        """Poll an async extraction job until complete."""
        start = time.monotonic()
        consecutive_errors = 0
        while True:
            elapsed = time.monotonic() - start
            if elapsed > max_wait:
                raise ExtractionError(f"Polling timed out after {max_wait}s for record {record_id}")

            try:
                resp = await self._request_with_retry("GET", f"/api/v1/extract/results/{record_id}")
                data = resp.json()
                consecutive_errors = 0
            except Exception as exc:
                consecutive_errors += 1
                if consecutive_errors >= 5:
                    raise ExtractionError(
                        f"Polling failed after {consecutive_errors} consecutive errors: {exc}"
                    ) from exc
                logger.warning("Poll error (%s), will retry…", exc)
                await asyncio.sleep(interval * 2)
                continue

            status = data.get("status", "").lower()

            if status in ("completed", "done", "success"):
                return data
            if status in ("failed", "error"):
                raise ExtractionError(f"Extraction failed: {data.get('error', data)}")

            logger.debug("Polling %s — status=%s, elapsed=%.1fs", record_id, status, elapsed)
            await asyncio.sleep(interval)

    async def extract_stream(
        self,
        file_path: str | Path,
        *,
        output_format: str = "markdown",
        json_options: str | None = None,
        include_metadata: str | None = None,
    ) -> AsyncIterator[str]:
        """Streaming extraction via SSE — yields partial results."""
        path = Path(file_path)
        fields = self._build_extract_fields(output_format, json_options, include_metadata)

        client = await self._ensure_client()
        with open(path, "rb") as f:
            files = {"file": (path.name, f, "application/octet-stream")}
            async with client.stream("POST", "/api/v1/extract/stream", data=fields, files=files) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        yield line[5:].strip()

    # ------------------------------------------------------------------
    # Convenience: auto-select sync vs async
    # ------------------------------------------------------------------

    async def extract(
        self,
        file_path: str | Path,
        *,
        output_format: str = "markdown",
        json_options: str | None = None,
        include_metadata: str | None = None,
        use_async: bool | None = None,
    ) -> dict[str, Any]:
        """Extract a document, picking sync (≤ 5 pages) or async+poll.

        ``use_async=True`` forces the async path.  When ``None`` (default),
        we use a file-size heuristic: files > 2 MB go async because they
        are likely to exceed the 5-page sync limit.
        """
        path = Path(file_path)

        if use_async is None:
            size_mb = path.stat().st_size / (1024 * 1024)
            use_async = size_mb > 2.0

        if use_async:
            record_id = await self.extract_async(
                path, output_format=output_format,
                json_options=json_options, include_metadata=include_metadata,
            )
            return await self.poll_result(record_id)

        try:
            return await self.extract_sync(
                path, output_format=output_format,
                json_options=json_options, include_metadata=include_metadata,
            )
        except ExtractionError:
            logger.info("Sync extraction failed — falling back to async")
            record_id = await self.extract_async(
                path, output_format=output_format,
                json_options=json_options, include_metadata=include_metadata,
            )
            return await self.poll_result(record_id)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    async def classify(
        self,
        file_path: str | Path,
        categories: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        """Classify pages of a document into the given categories."""
        import json as _json

        path = Path(file_path)
        with open(path, "rb") as f:
            files = {"file": (path.name, f, "application/octet-stream")}
            data = {"categories": _json.dumps(categories)}
            resp = await self._request_with_retry("POST", "/api/v1/classify/sync", data=data, files=files)

        result = resp.json()
        pages = result.get("result", {}).get("pages", [])
        return pages
