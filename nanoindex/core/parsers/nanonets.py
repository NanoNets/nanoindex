"""Nanonets parser — wraps the existing extractor as a BaseParser."""

from __future__ import annotations

import asyncio
from pathlib import Path

from nanoindex.core.parsers.base import BaseParser
from nanoindex.models import ParsedDocument


class NanonetsParser(BaseParser):
    """Parser that delegates to the Nanonets extraction API."""

    name: str = "nanonets"

    def __init__(self, api_key: str, **kwargs) -> None:
        self._api_key = api_key
        self._kwargs = kwargs

    async def parse(self, file_path: Path) -> ParsedDocument:
        """Parse a document using the Nanonets extractor."""
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
                parser_name=self.name,
            )
        finally:
            await client.close()
