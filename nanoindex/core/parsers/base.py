"""Abstract base class for document parsers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from nanoindex.models import ParsedDocument


class BaseParser(ABC):
    """Abstract parser interface. All parsers produce a ParsedDocument."""

    name: str = "base"

    @abstractmethod
    async def parse(self, file_path: Path) -> ParsedDocument:
        """Parse a document file and return structured output."""
        ...

    def supports(self, file_path: Path) -> bool:
        """Check if this parser can handle the given file type."""
        return file_path.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".tiff"}
