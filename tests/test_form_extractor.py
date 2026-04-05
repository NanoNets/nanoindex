from nanoindex.core.form_extractor import extract_form_from_markdown

def test_extract_kv():
    md = "Vendor: Acme Corp\nDate: 2024-01-15\nTotal: $1500.00\nInvoice Number: INV-001"
    form = extract_form_from_markdown(md)
    assert form.fields["vendor"] == "Acme Corp"
    assert form.fields["total"] == "$1500.00"
    assert form.fields["invoice_number"] == "INV-001"

def test_empty():
    form = extract_form_from_markdown("No key value pairs here")
    assert form.fields == {}
