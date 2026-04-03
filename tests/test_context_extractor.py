"""Tests for nanoindex.core.context_extractor."""

from nanoindex.core.context_extractor import enrich_modal_contexts, extract_context
from nanoindex.models import ModalContent, ParsedDocument


def test_extract_context():
    parsed = ParsedDocument(
        pages=[
            "Page 1 intro text.",
            "Page 2 has a table about revenue.",
            "Page 3 conclusion.",
        ],
        page_count=3,
    )
    item = ModalContent(content_type="table", page=2, content="| A | B |")
    ctx = extract_context(item, parsed)
    assert "revenue" in ctx
    assert "Page 1" in ctx  # previous page included


def test_enrich_modal_contexts():
    parsed = ParsedDocument(
        pages=["Intro.", "Revenue table here.", "Conclusion."],
        page_count=3,
        modal_contents=[ModalContent(content_type="table", page=2)],
    )
    enrich_modal_contexts(parsed)
    assert parsed.modal_contents[0].surrounding_text != ""


def test_empty_pages():
    """Graceful handling when pages list is empty."""
    parsed = ParsedDocument(pages=[], page_count=0)
    item = ModalContent(content_type="image", page=1)
    ctx = extract_context(item, parsed)
    assert ctx == ""


def test_empty_pages_enrich():
    """enrich_modal_contexts handles empty pages without error."""
    parsed = ParsedDocument(
        pages=[],
        page_count=0,
        modal_contents=[ModalContent(content_type="image", page=1)],
    )
    enrich_modal_contexts(parsed)
    assert parsed.modal_contents[0].surrounding_text == ""


def test_page_zero_item():
    """Items with page=0 should return empty context."""
    parsed = ParsedDocument(
        pages=["Some text."],
        page_count=1,
    )
    item = ModalContent(content_type="image", page=0)
    ctx = extract_context(item, parsed)
    assert ctx == ""


def test_existing_surrounding_text_not_overwritten():
    """enrich_modal_contexts should not overwrite existing surrounding_text."""
    parsed = ParsedDocument(
        pages=["Intro.", "Table page.", "End."],
        page_count=3,
        modal_contents=[
            ModalContent(
                content_type="table",
                page=2,
                surrounding_text="Already set",
            )
        ],
    )
    enrich_modal_contexts(parsed)
    assert parsed.modal_contents[0].surrounding_text == "Already set"
