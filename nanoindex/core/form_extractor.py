"""Extract key-value form fields from documents using V2 API."""
import logging
from nanoindex.models import ExtractedForm

logger = logging.getLogger(__name__)

# Common schemas for auto-detection
COMMON_SCHEMAS = {
    "invoice": ["vendor", "invoice_number", "date", "due_date", "total", "tax", "subtotal", "line_items"],
    "receipt": ["store", "date", "items", "total", "tax", "payment_method"],
    "insurance_dec": ["policy_number", "insured", "effective_date", "expiration_date", "limits", "deductible", "premium"],
}


async def extract_form_v2(file_id: str, client, fields: list[str] | None = None) -> ExtractedForm:
    """Extract form fields using V2 API JSON extraction."""
    if fields is None:
        # Auto-detect: try common fields
        fields = ["document_type", "date", "total", "name", "number", "address"]

    try:
        result = await client.extract_json(file_id, fields)
        content = result.get("result", {}).get("content", {})

        if isinstance(content, str):
            import json
            try:
                content = json.loads(content)
            except Exception:
                content = {}

        if not isinstance(content, dict):
            content = {}

        return ExtractedForm(
            schema_name="auto",
            fields=content,
            confidence=result.get("result", {}).get("overall_confidence", 0) / 100.0,
        )
    except Exception as exc:
        logger.warning("V2 form extraction failed: %s", exc)
        return ExtractedForm()


def extract_form_from_markdown(markdown: str) -> ExtractedForm:
    """Heuristic form extraction from markdown using key: value pattern matching."""
    import re
    fields = {}
    for line in markdown.split("\n"):
        match = re.match(r"^([A-Za-z][^:]{1,50}):\s*(.+)$", line.strip())
        if match:
            key = match.group(1).strip().lower().replace(" ", "_")
            value = match.group(2).strip()
            fields[key] = value

    return ExtractedForm(
        schema_name="heuristic",
        fields=fields,
    )
