"""Financial Knowledge Base — lookup and formatting utilities.

The KB is loaded once from ``financial_kb.json`` and cached for the process
lifetime. At query time, ``lookup_relevant_terms`` matches the query (and
optional decomposition) against term aliases/keywords and returns a concise
prompt-ready block with the canonical definitions, formulas, and conventions.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_KB_PATH = Path(__file__).parent / "financial_kb.json"


def _normalize(text: str) -> str:
    """Lowercase and collapse hyphens/underscores to spaces for fuzzy matching."""
    return text.lower().replace("-", " ").replace("_", " ")


class FinancialKB:
    """In-memory financial knowledge base."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.meta = data.get("_meta", {})
        self.terms: dict[str, dict[str, Any]] = data.get("terms", {})
        self._alias_index: dict[str, str] = {}
        for key, term in self.terms.items():
            for alias in term.get("aliases", []):
                self._alias_index[alias.lower()] = key

    def search(self, text: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Return KB entries whose aliases appear in *text*."""
        text_norm = _normalize(text)
        scored: dict[str, int] = {}
        for alias, key in self._alias_index.items():
            if _normalize(alias) in text_norm:
                scored[key] = scored.get(key, 0) + len(alias)

        ranked = sorted(scored, key=scored.__getitem__, reverse=True)
        return [self.terms[k] for k in ranked[:max_results]]

    def search_with_decomposition(
        self,
        query: str,
        decomposition: dict | None = None,
        max_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Search using query text + decomposition data points."""
        combined = query
        if decomposition:
            dps = decomposition.get("data_points", [])
            stmts = decomposition.get("statements_needed", [])
            combined += " " + " ".join(dps) + " " + " ".join(stmts)
        return self.search(combined, max_results=max_results)


@lru_cache(maxsize=1)
def _load_kb() -> FinancialKB:
    """Load the JSON knowledge base (cached for process lifetime)."""
    with open(_KB_PATH) as f:
        data = json.load(f)
    kb = FinancialKB(data)
    logger.info(
        "Financial KB loaded: %d terms, %d aliases",
        len(kb.terms),
        len(kb._alias_index),
    )
    return kb


def format_kb_entries(entries: list[dict[str, Any]]) -> str:
    """Format KB entries into a concise prompt-ready block."""
    if not entries:
        return ""

    parts = [
        "FINANCIAL REFERENCE — You MUST use the exact formulas below. "
        "Do NOT invent alternative formulas or skip components. "
        "If a convention says CRITICAL, follow it exactly."
    ]
    for entry in entries:
        name = entry["name"]
        formula = entry.get("formula", "N/A")
        parts.append(f"\n  {name}")
        parts.append(f"    Formula: {formula}")
        alt = entry.get("alternate_formula")
        if alt:
            parts.append(f"    Alt formula: {alt}")
        conventions = entry.get("conventions", [])
        if conventions:
            for conv in conventions:
                parts.append(f"    • {conv}")
        where = entry.get("where_to_find")
        if where:
            parts.append(f"    Source: {where}")

    return "\n".join(parts)


def lookup_relevant_terms(
    query: str,
    decomposition: dict | None = None,
    max_results: int = 5,
) -> str:
    """Look up and format relevant financial terms for a query.

    Returns a prompt-ready string, or empty string if no terms match.
    """
    kb = _load_kb()
    entries = kb.search_with_decomposition(query, decomposition, max_results)
    return format_kb_entries(entries)
