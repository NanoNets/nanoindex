from nanoindex.core.table_extractor import tables_from_markdown

def test_parse_markdown_table():
    md = "| Name | Amount |\n|---|---|\n| Alice | 100 |\n| Bob | 200 |"
    tables = tables_from_markdown(md)
    assert len(tables) == 1
    assert len(tables[0].rows) == 2
    assert tables[0].columns == ["Name", "Amount"]

def test_no_tables():
    md = "Just some text\nNo tables here"
    assert tables_from_markdown(md) == []

def test_multiple_tables():
    md = "| A | B |\n|---|---|\n| 1 | 2 |\n\nSome text\n\n| C | D |\n|---|---|\n| 3 | 4 |"
    tables = tables_from_markdown(md)
    assert len(tables) == 2
