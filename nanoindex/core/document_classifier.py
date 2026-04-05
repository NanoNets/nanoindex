"""Auto-detect document type using Nanonets V2 classify API or heuristics."""
import logging
import re
from nanoindex.models import ParsedDocument

logger = logging.getLogger(__name__)

DOCUMENT_TYPES = ["hierarchical", "tabular", "form", "mixed"]

def classify_from_markdown(markdown: str) -> str:
    """Heuristic classification from parsed markdown content."""
    lines = markdown.strip().split("\n")
    if not lines:
        return "hierarchical"

    heading_count = sum(1 for l in lines if l.startswith("#"))
    table_row_count = sum(1 for l in lines if "|" in l and l.strip().startswith("|"))
    kv_count = sum(1 for l in lines if re.match(r"^[A-Za-z][^:]{1,40}:\s", l))
    total = len(lines)

    if total == 0:
        return "hierarchical"

    table_ratio = table_row_count / total
    kv_ratio = kv_count / total
    heading_ratio = heading_count / total

    if table_ratio > 0.4:
        return "tabular"
    if kv_ratio > 0.3 and heading_ratio < 0.05:
        return "form"
    return "hierarchical"


async def classify_with_api(file_id: str, client) -> dict:
    """Classify using V2 API split mode. Returns per-page categories."""
    categories = [
        {"name": "report", "description": "Reports, filings, manuals, papers with sections and headings"},
        {"name": "table", "description": "Tables, ledgers, claims, transaction lists, spreadsheet-like data"},
        {"name": "form", "description": "Invoices, receipts, tax forms, applications, key-value documents"},
    ]
    try:
        result = await client.classify(file_id, categories=categories, mode="split")
        return result
    except Exception as exc:
        logger.warning("V2 classify failed: %s, falling back to heuristic", exc)
        return {}


def classify_parsed(parsed: ParsedDocument) -> str:
    """Classify a parsed document using heuristics on the markdown."""
    return classify_from_markdown(parsed.markdown)
