"""Tests for the extractor response parsers."""

from __future__ import annotations

from nanoindex.core.extractor import _parse_markdown_response, _parse_hierarchy_response


class TestParseMarkdownResponse:
    def test_basic_response(self):
        """Real Nanonets response shape: result.markdown.{content, metadata}"""
        resp = {
            "result": {
                "markdown": {
                    "content": "# Hello\n\nWorld",
                    "metadata": {
                        "bounding_boxes": {
                            "elements": [
                                {
                                    "content": "# Hello",
                                    "bounding_box": {
                                        "page": 1,
                                        "x": 0.1,
                                        "y": 0.05,
                                        "width": 0.8,
                                        "height": 0.05,
                                        "confidence": 0.98,
                                        "type": "heading",
                                    },
                                }
                            ],
                            "page_dimensions": {
                                "pages": [{"page": 1, "width": 612, "height": 792}],
                                "total_pages": 1,
                            },
                        }
                    },
                }
            }
        }
        md, bboxes, dims = _parse_markdown_response(resp)
        assert md == "# Hello\n\nWorld"
        assert len(bboxes) == 1
        assert bboxes[0].page == 1
        assert bboxes[0].confidence == 0.98
        assert len(dims) == 1

    def test_string_markdown_fallback(self):
        """When markdown value is a plain string (legacy or simplified response)."""
        resp = {"result": {"markdown": "# Title\n\nBody text"}}
        md, bboxes, dims = _parse_markdown_response(resp)
        assert md == "# Title\n\nBody text"
        assert bboxes == []
        assert dims == []

    def test_empty_response(self):
        md, bboxes, dims = _parse_markdown_response({})
        assert md == ""
        assert bboxes == []
        assert dims == []


class TestParseHierarchyResponse:
    def test_nested_sections(self):
        """Real Nanonets response shape: result.json.content.document"""
        resp = {
            "result": {
                "json": {
                    "content": {
                        "document": {
                            "sections": [
                                {
                                    "id": "s1",
                                    "title": "Intro",
                                    "level": 1,
                                    "content": "Hello",
                                    "subsections": [
                                        {
                                            "id": "s1.1",
                                            "title": "Background",
                                            "level": 2,
                                            "content": "Details",
                                            "subsections": [],
                                        }
                                    ],
                                }
                            ],
                            "tables": [
                                {
                                    "id": "t1",
                                    "title": "Revenue",
                                    "headers": ["Q1", "Q2"],
                                    "rows": [["100", "200"]],
                                }
                            ],
                            "key_value_pairs": [{"key": "Author", "value": "Jane"}],
                        }
                    },
                    "metadata": {},
                }
            }
        }
        sections, tables, kv = _parse_hierarchy_response(resp)
        assert len(sections) == 1
        assert sections[0].title == "Intro"
        assert len(sections[0].subsections) == 1
        assert len(tables) == 1
        assert tables[0].headers == ["Q1", "Q2"]
        assert len(kv) == 1
        assert kv[0].key == "Author"

    def test_heading_field_fallback(self):
        """Some hierarchy responses use 'heading' instead of 'title'."""
        resp = {
            "result": {
                "json": {
                    "content": {
                        "document": {
                            "sections": [
                                {"id": "s1", "heading": "Summary", "level": 1, "content": "text"}
                            ],
                        }
                    }
                }
            }
        }
        sections, _, _ = _parse_hierarchy_response(resp)
        assert len(sections) == 1
        assert sections[0].title == "Summary"

    def test_empty_response(self):
        sections, tables, kv = _parse_hierarchy_response({})
        assert sections == []
        assert tables == []
        assert kv == []
