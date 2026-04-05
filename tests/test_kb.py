"""Tests for the KnowledgeBase class."""

from __future__ import annotations

import json

from nanoindex.kb import KnowledgeBase
from nanoindex.models import KBConfig


def test_create_kb(tmp_path):
    """Creating a KB should set up the .nanoindex directory structure."""
    kb = KnowledgeBase(tmp_path / "my_kb")

    data_dir = tmp_path / "my_kb" / ".nanoindex"
    assert data_dir.exists()
    assert (data_dir / "trees").is_dir()
    assert (data_dir / "graphs").is_dir()
    assert (data_dir / "embeddings").is_dir()
    assert (data_dir / "queries").is_dir()
    assert (data_dir / "config.json").exists()


def test_status_empty(tmp_path):
    """An empty KB should report all-zero statistics."""
    kb = KnowledgeBase(tmp_path / "empty_kb")
    stats = kb.status()

    assert stats["documents"] == 0
    assert stats["concepts"] == 0
    assert stats["queries"] == 0
    assert stats["entities"] == 0
    assert stats["relationships"] == 0


def test_lint_empty(tmp_path):
    """An empty KB should produce no lint warnings."""
    kb = KnowledgeBase(tmp_path / "lint_kb")
    warnings = kb.lint()
    assert warnings == []


def test_config_save_load(tmp_path):
    """Config should round-trip through save/load."""
    kb = KnowledgeBase(tmp_path / "rt_kb")

    # Verify initial config was persisted
    config_path = tmp_path / "rt_kb" / ".nanoindex" / "config.json"
    assert config_path.exists()

    with open(config_path) as f:
        raw = json.load(f)

    loaded = KBConfig.model_validate(raw)
    assert loaded.version == "1"
    assert loaded.created_at != ""
    assert loaded.documents == []
    assert loaded.concept_index == {}

    # Mutate config and re-save, then reload via a new KB instance
    kb._config.concept_index["revenue"] = ["doc1", "doc2"]
    kb._save_config()

    kb2 = KnowledgeBase(tmp_path / "rt_kb")
    assert kb2._config.concept_index == {"revenue": ["doc1", "doc2"]}
