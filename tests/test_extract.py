"""Tests for the self-correcting table extraction pipeline."""

from __future__ import annotations


def test_validation_anchors():
    from nanoindex.core.validation_anchor import find_anchors

    text = "Total Claims: 35\nGrand Total: $229,370.00\nTotal Reserve: $1,370,000"
    anchors = find_anchors(text)
    assert anchors["row_count"] == 35
    assert anchors["total"] == 229370.0


def test_validation_anchors_subtotal():
    from nanoindex.core.validation_anchor import find_anchors

    text = "Subtotal: $14,000\nTotal: $15,260.00"
    anchors = find_anchors(text)
    assert anchors["subtotal"] == 14000.0
    assert anchors["total"] == 15260.0


def test_validation_anchors_item_count_suffix():
    from nanoindex.core.validation_anchor import find_anchors

    text = "45 items total"
    anchors = find_anchors(text)
    assert anchors["row_count"] == 45


def test_validation_anchors_empty():
    from nanoindex.core.validation_anchor import find_anchors

    anchors = find_anchors("No structured data here.")
    assert anchors == {}


def test_validate_extraction_pass():
    from nanoindex.core.table_validator import validate_extraction

    rows = [{"amount": "100"}, {"amount": "200"}]
    anchors = {"row_count": 2}
    result = validate_extraction(rows, anchors)
    assert result.passed


def test_validate_extraction_fail():
    from nanoindex.core.table_validator import validate_extraction

    rows = [{"amount": "100"}]
    anchors = {"row_count": 2}
    result = validate_extraction(rows, anchors)
    assert not result.passed
    assert result.row_count_expected == 2
    assert result.row_count_actual == 1


def test_validate_extraction_total_mismatch():
    from nanoindex.core.table_validator import validate_extraction

    rows = [{"amount": "100"}, {"amount": "200"}]
    anchors = {"total": 500.0}
    result = validate_extraction(rows, anchors, numeric_columns=["amount"])
    assert not result.passed
    assert len(result.total_mismatches) == 1
    assert result.total_mismatches[0]["expected"] == 500.0
    assert result.total_mismatches[0]["computed"] == 300.0


def test_validate_extraction_total_pass():
    from nanoindex.core.table_validator import validate_extraction

    rows = [{"amount": "100"}, {"amount": "200"}]
    anchors = {"total": 300.0}
    result = validate_extraction(rows, anchors, numeric_columns=["amount"])
    assert result.passed


def test_validate_extraction_no_anchors():
    from nanoindex.core.table_validator import validate_extraction

    rows = [{"a": 1}]
    anchors = {}
    result = validate_extraction(rows, anchors)
    assert result.passed


def test_extraction_result_to_csv(tmp_path):
    from nanoindex.models import ExtractionResult2

    result = ExtractionResult2(
        rows=[{"name": "Alice", "amount": 100}, {"name": "Bob", "amount": 200}],
        columns=["name", "amount"],
        mode="table",
    )
    csv_path = tmp_path / "out.csv"
    result.to_csv(str(csv_path))
    assert csv_path.exists()
    content = csv_path.read_text()
    assert "Alice" in content and "Bob" in content


def test_extraction_result_to_json(tmp_path):
    from nanoindex.models import ExtractionResult2

    result = ExtractionResult2(
        fields={"vendor": "Acme", "total": "$100"},
        mode="form",
    )
    json_path = tmp_path / "out.json"
    result.to_json(str(json_path))
    assert json_path.exists()
    import json

    data = json.loads(json_path.read_text())
    assert data["fields"]["vendor"] == "Acme"
    assert data["mode"] == "form"


def test_parse_number():
    from nanoindex.core.table_validator import _parse_number

    assert _parse_number("$1,234.56") == 1234.56
    assert _parse_number(42) == 42.0
    assert _parse_number("N/A") == 0.0
    assert _parse_number(3.14) == 3.14
    assert _parse_number("") == 0.0
    assert _parse_number(None) == 0.0
