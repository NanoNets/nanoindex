"""Shared pytest fixtures for NanoIndex tests."""

from __future__ import annotations

import os

import pytest


@pytest.fixture()
def api_key() -> str:
    """Provide NANONETS_API_KEY or skip the test."""
    key = os.environ.get("NANONETS_API_KEY", "")
    if not key:
        pytest.skip("NANONETS_API_KEY not set")
    return key


@pytest.fixture()
def sample_markdown() -> str:
    return (
        "# Annual Report 2024\n\n"
        "## Executive Summary\n\nThe company performed well.\n\n"
        "## Financial Results\n\n### Revenue\n\nRevenue was $500M.\n\n"
        "### Expenses\n\nExpenses were $300M.\n\n"
        "## Outlook\n\nWe expect growth.\n"
    )


@pytest.fixture()
def sample_hierarchy() -> dict:
    return {
        "document": {
            "title": "Annual Report 2024",
            "type": "document",
            "metadata": {"pages": "5"},
            "sections": [
                {
                    "id": "page_1_section_1",
                    "title": "Executive Summary",
                    "level": 1,
                    "content": "The company performed well.",
                    "subsections": [],
                },
                {
                    "id": "page_2_section_1",
                    "title": "Financial Results",
                    "level": 1,
                    "content": "",
                    "subsections": [
                        {
                            "id": "page_2_section_2",
                            "title": "Revenue",
                            "level": 2,
                            "content": "Revenue was $500M.",
                            "subsections": [],
                        },
                        {
                            "id": "page_3_section_1",
                            "title": "Expenses",
                            "level": 2,
                            "content": "Expenses were $300M.",
                            "subsections": [],
                        },
                    ],
                },
                {
                    "id": "page_4_section_1",
                    "title": "Outlook",
                    "level": 1,
                    "content": "We expect growth.",
                    "subsections": [],
                },
            ],
            "tables": [],
            "key_value_pairs": [],
        }
    }


@pytest.fixture()
def sample_bounding_boxes() -> list[dict]:
    return [
        {
            "content": "# Annual Report 2024",
            "bounding_box": {
                "x": 0.1,
                "y": 0.05,
                "width": 0.8,
                "height": 0.05,
                "confidence": 0.98,
                "page": 1,
                "type": "heading",
            },
        },
        {
            "content": "## Executive Summary",
            "bounding_box": {
                "x": 0.1,
                "y": 0.15,
                "width": 0.6,
                "height": 0.04,
                "confidence": 0.96,
                "page": 1,
                "type": "heading",
            },
        },
        {
            "content": "The company performed well.",
            "bounding_box": {
                "x": 0.1,
                "y": 0.20,
                "width": 0.8,
                "height": 0.10,
                "confidence": 0.95,
                "page": 1,
                "type": "paragraph",
            },
        },
        {
            "content": "## Financial Results",
            "bounding_box": {
                "x": 0.1,
                "y": 0.05,
                "width": 0.6,
                "height": 0.04,
                "confidence": 0.97,
                "page": 2,
                "type": "heading",
            },
        },
        {
            "content": "### Revenue",
            "bounding_box": {
                "x": 0.1,
                "y": 0.12,
                "width": 0.4,
                "height": 0.03,
                "confidence": 0.94,
                "page": 2,
                "type": "heading",
            },
        },
    ]
