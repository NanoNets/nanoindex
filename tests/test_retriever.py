"""Tests for the retriever module — node-id parsing and outline generation."""

from __future__ import annotations

from nanoindex.core.retriever import _parse_node_ids, _outline_for_nodes
from nanoindex.models import TreeNode


class TestParseNodeIds:
    def test_clean_json(self):
        assert _parse_node_ids('["0000", "0001.0002"]') == ["0000", "0001.0002"]

    def test_json_with_surrounding_text(self):
        text = 'Based on the document, the relevant sections are: ["0001", "0003"].'
        ids = _parse_node_ids(text)
        assert "0001" in ids
        assert "0003" in ids

    def test_empty_array(self):
        assert _parse_node_ids("[]") == []

    def test_no_json_fallback_to_regex(self):
        text = "Nodes 0001 and 0002.0003 are relevant"
        ids = _parse_node_ids(text)
        assert "0001" in ids
        assert "0002.0003" in ids

    def test_garbage_input(self):
        assert _parse_node_ids("nothing useful here") == []


class TestOutlineForNodes:
    def test_simple(self):
        nodes = [
            TreeNode(title="Intro", node_id="0000", start_index=1, end_index=3, summary="Introduction"),
            TreeNode(title="Details", node_id="0001", start_index=4, end_index=10),
        ]
        outline = _outline_for_nodes(nodes)
        assert "0000" in outline
        assert "Intro" in outline
        assert "Introduction" in outline
        assert "0001" in outline

    def test_with_children_count(self):
        child = TreeNode(title="Sub", node_id="0000.0000", level=2)
        parent = TreeNode(title="Main", node_id="0000", level=1, nodes=[child])
        outline = _outline_for_nodes([parent])
        assert "1 sub-sections" in outline
