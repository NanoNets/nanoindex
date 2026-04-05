"""Validate extracted table data against document anchors.

Compares row counts and numeric column totals to ground-truth values
found in the document text (e.g. "Total Claims: 35", "Grand Total: $229,370").
"""

from __future__ import annotations

from nanoindex.models import ValidationResult


def validate_extraction(
    rows: list[dict],
    anchors: dict,
    numeric_columns: list[str] | None = None,
) -> ValidationResult:
    """Validate extracted rows against document anchors.

    Parameters
    ----------
    rows:
        The extracted data rows.
    anchors:
        Dict from :func:`find_anchors` with keys like ``row_count``, ``total``.
    numeric_columns:
        Column names that should be summed and compared to ``anchors["total"]``.

    Returns
    -------
    ValidationResult
        Populated validation result with pass/fail status and messages.
    """
    result = ValidationResult()

    # Check row count
    if "row_count" in anchors:
        result.row_count_expected = anchors["row_count"]
        result.row_count_actual = len(rows)
        result.row_count_match = len(rows) == anchors["row_count"]
        if not result.row_count_match:
            result.messages.append(
                f"Row count mismatch: expected {anchors['row_count']}, got {len(rows)}"
            )

    # Check numeric totals
    if "total" in anchors and numeric_columns:
        for col in numeric_columns:
            computed = sum(_parse_number(row.get(col, 0)) for row in rows)
            expected = anchors["total"]
            if abs(computed - expected) > 0.01:
                result.total_mismatches.append(
                    {"column": col, "expected": expected, "computed": computed}
                )
                result.messages.append(
                    f"Total mismatch for '{col}': expected {expected}, got {computed}"
                )

    result.passed = (result.row_count_match is not False) and len(result.total_mismatches) == 0
    return result


def _parse_number(val) -> float:
    """Best-effort parse of a value into a float."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        cleaned = val.replace("$", "").replace(",", "").strip()
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return 0.0
    return 0.0
