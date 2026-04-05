"""Find validation anchors (totals, counts) in document text.

Anchors are ground-truth values embedded in the document itself — row counts,
dollar totals, subtotals — that can be used to verify extracted data.
"""

from __future__ import annotations

import re


def find_anchors(markdown: str) -> dict:
    """Scan text for validation anchors like 'Total: $X', 'Count: N'.

    Returns a dict with optional keys:
      - ``row_count``: expected number of data rows
      - ``total``: grand total dollar amount
      - ``subtotal``: subtotal dollar amount
    """
    anchors: dict = {}

    # Row/item counts
    for pattern in [
        r"(?:total|count|number of)\s*(?:claims|items|records|rows|entries|transactions)[\s:]+(\d+)",
        r"(\d+)\s+(?:claims|items|records|transactions)\s+(?:total|found|listed)",
    ]:
        m = re.search(pattern, markdown, re.IGNORECASE)
        if m:
            anchors["row_count"] = int(m.group(1))

    # Dollar totals
    for pattern in [
        r"(?:grand total|total|total (?:paid|due|amount|balance))[\s:]*\$?([\d,]+\.?\d*)",
        r"(?:subtotal)[\s:]*\$?([\d,]+\.?\d*)",
    ]:
        for m in re.finditer(pattern, markdown, re.IGNORECASE):
            matched_text = m.group().lower()
            if "subtotal" in matched_text:
                key = "subtotal"
            else:
                key = "total"
            anchors[key] = float(m.group(1).replace(",", ""))

    return anchors
