"""Tests for the modal processor registry and processor classes."""

from nanoindex.core.modal_processors import PROCESSORS, get_processor
from nanoindex.core.modal_processors.image_processor import ImageModalProcessor
from nanoindex.core.modal_processors.table_processor import TableModalProcessor


def test_processor_registry():
    """PROCESSORS dict should contain entries for 'image' and 'table'."""
    assert "image" in PROCESSORS
    assert "table" in PROCESSORS
    assert PROCESSORS["image"] is ImageModalProcessor
    assert PROCESSORS["table"] is TableModalProcessor


def test_get_processor():
    """get_processor should return correct processor instances."""
    img = get_processor("image")
    assert isinstance(img, ImageModalProcessor)
    assert img.content_type == "image"

    tbl = get_processor("table")
    assert isinstance(tbl, TableModalProcessor)
    assert tbl.content_type == "table"


def test_get_unknown():
    """get_processor should return None for an unknown content type."""
    assert get_processor("video") is None
    assert get_processor("") is None
    assert get_processor("equation") is None
