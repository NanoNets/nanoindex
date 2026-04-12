"""Extract structured tables from documents using V2 API or markdown parsing."""

import csv
import io
import logging
from nanoindex.models import ExtractedTable

logger = logging.getLogger(__name__)


def tables_from_markdown(markdown: str) -> list[ExtractedTable]:
    """Parse markdown tables into ExtractedTable objects."""
    tables = []
    lines = markdown.split("\n")
    current_table_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and "|" in stripped[1:]:
            current_table_lines.append(stripped)
        else:
            if len(current_table_lines) >= 3:  # header + separator + at least 1 row
                table = _parse_md_table(current_table_lines)
                if table:
                    tables.append(table)
            current_table_lines = []

    # Handle trailing table
    if len(current_table_lines) >= 3:
        table = _parse_md_table(current_table_lines)
        if table:
            tables.append(table)

    return tables


def _parse_md_table(lines: list[str]) -> ExtractedTable | None:
    """Parse a sequence of markdown table lines into an ExtractedTable."""
    if len(lines) < 3:
        return None

    def split_row(line):
        cells = [c.strip() for c in line.strip("|").split("|")]
        return [c for c in cells if c]

    headers = split_row(lines[0])
    if not headers:
        return None

    # Skip separator line (line[1] is usually |---|---|)
    rows = []
    for line in lines[2:]:
        cells = split_row(line)
        if len(cells) == len(headers):
            row = dict(zip(headers, cells))
            rows.append(row)

    if not rows:
        return None

    # Try to detect name from context (use first header as fallback)
    return ExtractedTable(
        name=f"Table ({len(rows)} rows)",
        columns=headers,
        rows=rows,
    )


async def extract_tables_v2(file_id: str, client) -> list[ExtractedTable]:
    """Extract tables using V2 API CSV extraction."""
    try:
        result = await client.extract_csv(file_id)
        content = result.get("result", {}).get("content", "")
        if not content:
            return []

        # Parse CSV content into ExtractedTable
        reader = csv.DictReader(io.StringIO(content))
        rows = list(reader)
        if not rows:
            return []

        return [
            ExtractedTable(
                name=f"Extracted Table ({len(rows)} rows)",
                columns=list(rows[0].keys()),
                rows=rows,
            )
        ]
    except Exception as exc:
        logger.warning("V2 table extraction failed: %s", exc)
        return []
