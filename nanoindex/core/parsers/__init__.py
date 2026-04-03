"""Parser registry and factory."""

from __future__ import annotations

from nanoindex.core.parsers.base import BaseParser

_REGISTRY: dict[str, type[BaseParser]] = {}


def register_parser(name: str, cls: type[BaseParser]) -> None:
    _REGISTRY[name] = cls


def get_parser(name: str, **kwargs) -> BaseParser:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown parser: {name!r}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name](**kwargs)


def available_parsers() -> list[str]:
    return list(_REGISTRY.keys())


# --- Auto-register built-in parsers ---
from nanoindex.core.parsers.nanonets import NanonetsParser  # noqa: E402
from nanoindex.core.parsers.pymupdf import PyMuPDFParser  # noqa: E402

register_parser("nanonets", NanonetsParser)
register_parser("pymupdf", PyMuPDFParser)
