"""Nanonets parser — uses V2 API by default for bounding boxes, falls back to V1."""

from __future__ import annotations

import logging
from pathlib import Path

from nanoindex.core.parsers.base import BaseParser
from nanoindex.models import ParsedDocument

logger = logging.getLogger(__name__)


class NanonetsParser(BaseParser):
    """Parser that delegates to the Nanonets extraction API.

    Uses the V2 API by default (provides bounding boxes and element metadata).
    Falls back to V1 if V2 fails.
    """

    name: str = "nanonets"

    def __init__(self, api_key: str, use_v2: bool = True, **kwargs) -> None:
        self._api_key = api_key
        self._use_v2 = use_v2
        self._kwargs = kwargs

    async def parse(self, file_path: Path) -> ParsedDocument:
        """Parse a document using the Nanonets API.

        Uses V2 by default (bounding boxes + auto sync/async routing).
        Set use_v2=False to use V1 instead.
        """
        if self._use_v2:
            return await self._parse_v2(file_path)

        return await self._parse_v1(file_path)

    async def _parse_v2(self, file_path: Path) -> ParsedDocument:
        """Parse using V2 API — returns bounding boxes."""
        from nanoindex.core.client_v2 import NanonetsV2Client

        client = NanonetsV2Client(api_key=self._api_key)
        try:
            parsed = await client.parse_to_document(file_path)
            parsed.parser_name = "nanonets-v2"
            return parsed
        finally:
            await client.close()

    async def _parse_v1(self, file_path: Path) -> ParsedDocument:
        """Parse using V1 API — no bounding boxes but handles all doc sizes."""
        from nanoindex.core.client import NanonetsClient
        from nanoindex.core.extractor import extract_document

        client = NanonetsClient(self._api_key, **self._kwargs)
        try:
            result = await extract_document(file_path, client)
            return ParsedDocument(
                markdown=result.markdown,
                page_count=result.page_count,
                toc=result.toc,
                hierarchy_sections=result.hierarchy_sections,
                bounding_boxes=result.bounding_boxes,
                page_dimensions=result.page_dimensions,
                processing_time=result.processing_time,
                parser_name="nanonets-v1",
            )
        finally:
            await client.close()
