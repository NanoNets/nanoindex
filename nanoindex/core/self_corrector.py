"""Self-correcting extraction loop.

When validation fails (row count mismatch, total mismatch), this module
asks an LLM to diagnose and fix the extraction, retrying up to N times.
"""

from __future__ import annotations

import json as json_mod
import logging
import re
from typing import Any

from nanoindex.core.table_validator import validate_extraction

logger = logging.getLogger(__name__)


async def self_correcting_extract(
    pdf_path: Any,
    parsed_markdown: str,
    initial_rows: list[dict],
    anchors: dict,
    llm: Any,
    *,
    max_iterations: int = 3,
) -> tuple[list[dict], list[str]]:
    """Self-correcting extraction loop.

    Iteratively validates extracted rows against document anchors and uses
    an LLM to diagnose and fix mismatches.

    Returns
    -------
    tuple[list[dict], list[str]]
        ``(corrected_rows, corrections_log)``
    """
    rows = list(initial_rows)
    corrections: list[str] = []

    for iteration in range(max_iterations):
        validation = validate_extraction(rows, anchors)
        if validation.passed:
            break

        # Build prompt for the agent
        prompt = f"""You are validating extracted data from a document.

Validation failed:
{chr(10).join(validation.messages)}

Current extraction: {len(rows)} rows
Document states: {anchors}

The extracted rows are:
{_format_rows_for_llm(rows[:20])}

The document text around totals:
{_find_total_context(parsed_markdown)}

Identify the problem and suggest corrections. Return JSON:
{{
    "diagnosis": "what went wrong",
    "action": "remove_duplicates" | "add_missing" | "fix_values",
    "duplicate_indices": [list of 0-based indices to remove],
    "corrections": ["description of each fix"]
}}"""

        try:
            resp = await llm.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.0,
            )
            # Parse response — strip markdown fences if present
            resp_clean = resp.strip()
            if resp_clean.startswith("```"):
                resp_clean = re.sub(r"^```(?:json)?\s*", "", resp_clean)
                resp_clean = re.sub(r"\s*```$", "", resp_clean)

            fix = json_mod.loads(resp_clean)

            if fix.get("action") == "remove_duplicates" and fix.get("duplicate_indices"):
                indices = sorted(fix["duplicate_indices"], reverse=True)
                removed = 0
                for idx in indices:
                    if 0 <= idx < len(rows):
                        rows.pop(idx)
                        removed += 1
                corrections.append(f"Iteration {iteration + 1}: removed {removed} duplicate rows")

            corrections.extend(fix.get("corrections", []))
        except Exception as exc:
            logger.debug("Self-correction iteration %d failed: %s", iteration + 1, exc)
            corrections.append(
                f"Iteration {iteration + 1}: self-correction failed, keeping current result"
            )
            break

    return rows, corrections


def _format_rows_for_llm(rows: list[dict]) -> str:
    if not rows:
        return "(no rows)"
    lines = []
    for i, row in enumerate(rows):
        lines.append(f"  [{i}] {row}")
    return "\n".join(lines)


def _find_total_context(markdown: str, window: int = 500) -> str:
    m = re.search(r"(?i)(total|grand total|sum).{0,200}", markdown)
    if m:
        start = max(0, m.start() - 200)
        return markdown[start : m.end()]
    return markdown[-500:]
