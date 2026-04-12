"""Tests for the parser interface and registry."""

from nanoindex.core.parsers import available_parsers, get_parser
from nanoindex.core.parsers.base import BaseParser
from nanoindex.models import ParsedDocument


def test_parser_registry():
    parsers = available_parsers()
    assert isinstance(parsers, list)


def test_base_parser_is_abstract():
    import pytest

    with pytest.raises(TypeError):
        BaseParser()


def test_nanonets_parser_registered():
    parsers = available_parsers()
    assert "nanonets" in parsers


def test_nanonets_parser_instantiation():
    parser = get_parser("nanonets", api_key="test-key")
    assert parser.name == "nanonets"


# ---------------------------------------------------------------------------
# PyMuPDF parser tests
# ---------------------------------------------------------------------------


def test_pymupdf_registered():
    """PyMuPDF parser must appear in the registry."""
    assert "pymupdf" in available_parsers()


def test_pymupdf_get_parser():
    """get_parser('pymupdf') should return a BaseParser subclass."""
    parser = get_parser("pymupdf")
    assert isinstance(parser, BaseParser)
    assert parser.name == "pymupdf"


import asyncio
from pathlib import Path
import pytest

FIXTURE_PDF = Path(__file__).parent / "fixtures" / "q1-fy25-earnings.pdf"


@pytest.mark.skipif(not FIXTURE_PDF.exists(), reason="Test PDF fixture not available")
def test_pymupdf_parse_pdf():
    """Parse the test fixture and verify basic output structure."""
    parser = get_parser("pymupdf")
    result: ParsedDocument = asyncio.run(parser.parse(FIXTURE_PDF))

    assert result.parser_name == "pymupdf"
    assert result.page_count > 0
    assert len(result.pages) == result.page_count
    assert len(result.page_dimensions) == result.page_count
    assert result.processing_time > 0
    # There should be some text in the document.
    assert len(result.markdown) > 0


# ---------------------------------------------------------------------------
# NanoIndex integration with pymupdf parser
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not FIXTURE_PDF.exists(), reason="Test PDF fixture not available")
def test_nanoindex_pymupdf_integration():
    """NanoIndex with parser='pymupdf' should index a PDF and produce a valid tree."""
    from nanoindex import NanoIndex
    from nanoindex.config import NanoIndexConfig
    from nanoindex.exceptions import ConfigError

    config = NanoIndexConfig(
        parser="pymupdf",
        add_summaries=False,
        add_doc_description=False,
        split_strategy="heuristic",
        use_hierarchy_api=False,
    )
    try:
        ni = NanoIndex(config=config)
        tree = ni.index(FIXTURE_PDF)
    except ConfigError:
        pytest.skip("No LLM API key available for tree refinement")
        return

    assert tree.doc_name == FIXTURE_PDF.stem
    assert len(tree.structure) > 0
