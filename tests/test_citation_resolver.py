"""Tests for citation_resolver — narrowing bboxes to exact matches."""

from nanoindex.core.citation_resolver import _extract_key_phrases, _match_bboxes, resolve_citations
from nanoindex.models import Answer, BoundingBox, Citation, DocumentTree, TreeNode


def test_extract_dollar_amounts():
    phrases = _extract_key_phrases("Revenue was $127.4 million, up 23% YoY")
    assert any("127.4" in p for p in phrases)
    assert any("23" in p for p in phrases)


def test_extract_percentages():
    phrases = _extract_key_phrases("Gross margin improved to 74.2%")
    assert any("74.2" in p for p in phrases)


def test_extract_numbers():
    phrases = _extract_key_phrases("The company has 2,400 customers and ARR of $512 million")
    assert any("512" in p for p in phrases)


def test_match_bboxes_finds_match():
    bboxes = [
        BoundingBox(page=1, x=0.1, y=0.1, width=0.8, height=0.03, text="Company Overview"),
        BoundingBox(page=1, x=0.1, y=0.2, width=0.8, height=0.03, text="Revenue: $127.4 million"),
        BoundingBox(page=1, x=0.1, y=0.3, width=0.8, height=0.03, text="Net Income: $14.1 million"),
    ]
    matched = _match_bboxes(bboxes, ["127.4"])
    assert len(matched) == 1
    assert matched[0].text == "Revenue: $127.4 million"


def test_match_bboxes_multiple_matches():
    bboxes = [
        BoundingBox(page=1, x=0.1, y=0.1, width=0.8, height=0.03, text="Revenue $127.4M, growth 23%"),
        BoundingBox(page=1, x=0.1, y=0.2, width=0.8, height=0.03, text="Expenses $50M"),
        BoundingBox(page=1, x=0.1, y=0.3, width=0.8, height=0.03, text="Net income from 127.4 revenue"),
    ]
    matched = _match_bboxes(bboxes, ["127.4", "23"])
    # First bbox matches both phrases (score 2), third matches one (score 1)
    assert len(matched) == 2
    assert matched[0].text == "Revenue $127.4M, growth 23%"  # highest score


def test_match_bboxes_no_match():
    bboxes = [
        BoundingBox(page=1, x=0.1, y=0.1, width=0.8, height=0.03, text="Unrelated content"),
    ]
    matched = _match_bboxes(bboxes, ["127.4"])
    assert matched == []


def test_resolve_citations_narrows():
    tree = DocumentTree(
        doc_name="test",
        structure=[
            TreeNode(
                title="Financial Results",
                node_id="0001",
                start_index=2,
                end_index=2,
                bounding_boxes=[
                    BoundingBox(page=2, x=0.1, y=0.1, width=0.8, height=0.03, text="Section header"),
                    BoundingBox(page=2, x=0.1, y=0.2, width=0.8, height=0.03, text="Revenue: $127.4 million"),
                    BoundingBox(page=2, x=0.1, y=0.3, width=0.8, height=0.03, text="Gross Margin: 74.2%"),
                    BoundingBox(page=2, x=0.1, y=0.4, width=0.8, height=0.03, text="Operating expenses breakdown"),
                ],
            )
        ],
    )

    answer = Answer(
        content="Revenue was $127.4 million in Q3.",
        citations=[
            Citation(node_id="0001", title="Financial Results", pages=[2], bounding_boxes=[]),
        ],
    )

    resolved = resolve_citations(answer, tree)
    # Should narrow to just the bbox containing "127.4"
    assert len(resolved.citations[0].bounding_boxes) == 1
    assert "127.4" in resolved.citations[0].bounding_boxes[0].text


def test_resolve_no_bboxes_graceful():
    tree = DocumentTree(
        doc_name="test",
        structure=[TreeNode(title="Test", node_id="0001")],
    )
    answer = Answer(
        content="Some answer",
        citations=[Citation(node_id="0001", title="Test", pages=[1])],
    )
    resolved = resolve_citations(answer, tree)
    assert resolved.citations[0].bounding_boxes == []
