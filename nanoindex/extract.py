"""Structured data extraction from PDFs.

Separate from tree indexing. For extracting tables, forms, and structured fields.

Usage::

    from nanoindex import NanoIndex
    ni = NanoIndex()
    result = ni.extract("invoice.pdf")
    print(result.fields)  # {'vendor': 'Acme', 'total': '$15,260'}

    result = ni.extract("loss_run.pdf")
    print(result.rows)  # [{'claim_no': 'WC-001', 'paid': 15000}, ...]
    result.to_csv("claims.csv")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

from nanoindex.models import ExtractionResult2, ValidationResult

if TYPE_CHECKING:
    from nanoindex import NanoIndex

logger = logging.getLogger(__name__)


async def extract_document(
    file_path: str | Path,
    ni: NanoIndex,
    *,
    mode: str = "auto",
    schema: dict[str, Any] | None = None,
) -> ExtractionResult2:
    """Extract structured data from a PDF.

    Parameters
    ----------
    file_path:
        Path to the PDF file.
    ni:
        A :class:`NanoIndex` instance (provides parser, LLM, config).
    mode:
        ``"auto"`` (detect), ``"table"``, or ``"form"``.
    schema:
        Optional dict describing expected columns/fields.

    Returns
    -------
    ExtractionResult2
        Extracted rows (table mode) or fields (form mode) with validation.
    """
    path = Path(file_path)

    # Step 1: Parse PDF
    from nanoindex.core.parsers import get_parser

    parser_kwargs: dict[str, Any] = {}
    if ni.config.parser == "nanonets":
        parser_kwargs["api_key"] = ni.config.require_nanonets_key()
        parser_kwargs["use_v2"] = ni.config.use_v2_api
    parser = get_parser(ni.config.parser, **parser_kwargs)
    parsed = await parser.parse(path)
    markdown = parsed.markdown

    # Step 2: Detect mode
    if mode == "auto":
        from nanoindex.core.document_classifier import classify_parsed

        detected = classify_parsed(parsed)
        if detected == "tabular":
            mode = "table"
        elif detected == "form":
            mode = "form"
        else:
            # Default to table for mixed/hierarchical with tables present
            mode = "table"
        logger.info("Auto-detected extraction mode: %s (classifier: %s)", mode, detected)

    # Step 3: Extract
    if mode == "table":
        return await _extract_table(markdown, path, ni, schema=schema)
    elif mode == "form":
        return _extract_form(markdown, path)
    else:
        raise ValueError(f"Unknown extraction mode: {mode!r}. Use 'auto', 'table', or 'form'.")


async def _extract_table(
    markdown: str,
    path: Path,
    ni: NanoIndex,
    *,
    schema: dict[str, Any] | None = None,
) -> ExtractionResult2:
    """Extract table rows, validate against anchors, and self-correct."""
    from nanoindex.core.table_extractor import tables_from_markdown
    from nanoindex.core.validation_anchor import find_anchors
    from nanoindex.core.table_validator import validate_extraction

    # Parse tables from markdown
    tables = tables_from_markdown(markdown)

    # Merge all table rows (for multi-table documents like loss runs)
    all_rows: list[dict[str, Any]] = []
    all_columns: list[str] = []
    source_pages: list[int] = []
    for table in tables:
        all_rows.extend(table.rows)
        if not all_columns and table.columns:
            all_columns = table.columns
        source_pages.extend(table.source_pages)

    # Find validation anchors
    anchors = find_anchors(markdown)
    logger.info("Found anchors: %s", anchors)

    # Detect numeric columns from schema or heuristic
    numeric_columns: list[str] | None = None
    if schema and "numeric_columns" in schema:
        numeric_columns = schema["numeric_columns"]
    elif all_rows:
        numeric_columns = _detect_numeric_columns(all_rows, all_columns)

    # Validate
    validation = validate_extraction(all_rows, anchors, numeric_columns)

    # Self-correct if validation failed and LLM is available
    corrections: list[str] = []
    if not validation.passed and anchors:
        try:
            llm = ni._get_reasoning_llm()
            from nanoindex.core.self_corrector import self_correcting_extract

            all_rows, corrections = await self_correcting_extract(
                path, markdown, all_rows, anchors, llm,
            )
            # Re-validate after corrections
            validation = validate_extraction(all_rows, anchors, numeric_columns)
        except Exception as exc:
            logger.warning("Self-correction unavailable: %s", exc)
            corrections.append(f"Self-correction skipped: {exc}")

    return ExtractionResult2(
        rows=all_rows,
        columns=all_columns,
        validation=validation,
        corrections=corrections,
        confidence=1.0 if validation.passed else 0.5,
        mode="table",
        source_pages=source_pages,
        doc_name=path.stem,
    )


def _extract_form(markdown: str, path: Path) -> ExtractionResult2:
    """Extract key-value form fields."""
    from nanoindex.core.form_extractor import extract_form_from_markdown

    form = extract_form_from_markdown(markdown)
    return ExtractionResult2(
        fields=form.fields,
        confidence=form.confidence if form.confidence else 0.5,
        mode="form",
        source_pages=form.source_pages,
        doc_name=path.stem,
    )


def _detect_numeric_columns(
    rows: list[dict[str, Any]], columns: list[str],
) -> list[str]:
    """Heuristic: find columns where most values look numeric."""
    import re

    numeric_cols = []
    check_cols = columns or (list(rows[0].keys()) if rows else [])
    for col in check_cols:
        numeric_count = 0
        total = 0
        for row in rows[:20]:  # sample
            val = str(row.get(col, ""))
            cleaned = val.replace("$", "").replace(",", "").strip()
            if cleaned and re.match(r"^-?\d+\.?\d*$", cleaned):
                numeric_count += 1
            total += 1
        if total > 0 and numeric_count / total > 0.5:
            numeric_cols.append(col)
    return numeric_cols
