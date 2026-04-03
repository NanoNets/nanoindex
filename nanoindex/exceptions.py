"""Typed exception hierarchy for NanoIndex.

Every failure raises a specific subclass so callers can catch precisely
what they care about (e.g. ``except RateLimitError`` vs ``except NanoIndexError``).
"""

from __future__ import annotations


class NanoIndexError(Exception):
    """Base exception for all NanoIndex errors."""


class ConfigError(NanoIndexError):
    """Missing or invalid configuration (API keys, bad YAML, etc.)."""


class ExtractionError(NanoIndexError):
    """Nanonets extraction API returned an error or unexpected response."""


class RateLimitError(ExtractionError):
    """HTTP 429 — Nanonets rate limit exceeded."""

    def __init__(self, retry_after: float | None = None) -> None:
        self.retry_after = retry_after
        msg = "Nanonets rate limit exceeded"
        if retry_after is not None:
            msg += f" (retry after {retry_after:.1f}s)"
        super().__init__(msg)


class TreeBuildError(NanoIndexError):
    """Failed to construct a document tree from extraction output."""


class RetrievalError(NanoIndexError):
    """Tree search or node retrieval failed."""


class GenerationError(NanoIndexError):
    """Answer generation (text or vision) failed."""
