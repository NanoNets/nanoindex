from nanoindex.core.document_classifier import classify_from_markdown

def test_hierarchical():
    md = "# Title\n## Section 1\nSome text\n## Section 2\nMore text\n### Subsection\nDetails"
    assert classify_from_markdown(md) == "hierarchical"

def test_tabular():
    md = "| Col1 | Col2 | Col3 |\n|---|---|---|\n" + "\n".join(f"| val{i} | val{i} | val{i} |" for i in range(20))
    assert classify_from_markdown(md) == "tabular"

def test_form():
    md = "\n".join(f"Field {i}: Value {i}" for i in range(20))
    assert classify_from_markdown(md) == "form"

def test_empty():
    assert classify_from_markdown("") == "hierarchical"
