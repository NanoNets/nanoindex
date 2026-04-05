"""Tests for bounding box and page dimension propagation in citations."""

from nanoindex.models import BoundingBox, Citation, PageDimensions


def test_citation_has_bounding_boxes():
    bb = BoundingBox(page=1, x=0.1, y=0.2, width=0.5, height=0.3, region_type="text")
    c = Citation(node_id="0001", title="Test", pages=[1], bounding_boxes=[bb])
    assert len(c.bounding_boxes) == 1
    assert c.bounding_boxes[0].page == 1


def test_citation_has_page_dimensions():
    pd = PageDimensions(page=1, width=612, height=792)
    c = Citation(node_id="0001", title="Test", pages=[1], page_dimensions=[pd])
    assert len(c.page_dimensions) == 1
