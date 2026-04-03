"""Configuration management for NanoIndex.

Resolution order (later wins):
  1. Shipped ``config.yaml`` defaults
  2. User-provided YAML file
  3. Environment variables (NANONETS_API_KEY, LLM_BASE_URL, …)
  4. Explicit keyword overrides passed to ``NanoIndexConfig(...)``
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

from nanoindex.exceptions import ConfigError

_PACKAGE_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG_PATH = _PACKAGE_DIR.parent / "config.yaml"

# Mapping: env-var name → config field name
_ENV_MAP: dict[str, str] = {
    "NANONETS_API_KEY": "nanonets_api_key",
    "LLM_BASE_URL": "llm_base_url",
    "LLM_API_KEY": "llm_api_key",
    "LLM_MODEL": "llm_model",
    "REASONING_LLM_BASE_URL": "reasoning_llm_base_url",
    "REASONING_LLM_API_KEY": "reasoning_llm_api_key",
    "REASONING_LLM_MODEL": "reasoning_llm_model",
}


class NanoIndexConfig(BaseModel):
    """Single configuration object threaded through the entire pipeline."""

    nanonets_api_key: str | None = None
    llm_base_url: str = "https://extraction-api.nanonets.com/v1"
    llm_api_key: str | None = None
    llm_model: str = "nanonets-ocr-3"

    # Reasoning LLM — used for retrieval + answer generation.
    # When None, falls back to the default LLM above.
    reasoning_llm_base_url: str | None = None
    reasoning_llm_api_key: str | None = None
    reasoning_llm_model: str | None = None

    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    min_node_tokens: int = Field(default=50, ge=0)
    max_node_tokens: int = Field(default=20_000, ge=100)
    max_node_pages: int = Field(default=10, ge=1)
    parser: str = "nanonets"  # "nanonets" or "pymupdf"
    split_strategy: str = "hybrid"  # "heuristic", "llm", or "hybrid"

    add_summaries: bool = True
    add_doc_description: bool = False
    add_node_text: bool = False
    summary_model: str | None = None

    # Graph + embedding settings (for fast retrieval mode)
    build_graph: bool = False         # Extract entities at index time
    build_embeddings: bool = False    # Embed node summaries at index time
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: str | None = None
    embedding_base_url: str = "https://api.openai.com/v1"
    graph_hops: int = 1               # Expansion depth for graph traversal
    fast_top_k_embed: int = 20        # Tier 1: embedding candidates
    fast_top_k_final: int = 10        # Tier 2: LLM final selection

    @model_validator(mode="after")
    def _resolve_defaults(self) -> "NanoIndexConfig":
        if self.llm_api_key is None and self.nanonets_api_key is not None:
            self.llm_api_key = self.nanonets_api_key
        return self

    def require_nanonets_key(self) -> str:
        if not self.nanonets_api_key:
            raise ConfigError(
                "nanonets_api_key is required. Set it via NANONETS_API_KEY env var, "
                "config.yaml, or pass it to NanoIndexConfig(nanonets_api_key=...)."
            )
        return self.nanonets_api_key

    def require_llm_key(self) -> str:
        key = self.llm_api_key or self.nanonets_api_key
        if not key:
            key = "no-key-required"
        return key


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _load_env() -> dict[str, Any]:
    load_dotenv()
    out: dict[str, Any] = {}
    for env_key, field_name in _ENV_MAP.items():
        val = os.environ.get(env_key)
        if val is not None:
            out[field_name] = val
    return out


def load_config(
    config_path: str | Path | None = None,
    **overrides: Any,
) -> NanoIndexConfig:
    """Build a ``NanoIndexConfig`` from YAML defaults → env vars → overrides."""
    merged: dict[str, Any] = {}

    merged.update(_load_yaml(_DEFAULT_CONFIG_PATH))

    if config_path is not None:
        merged.update(_load_yaml(Path(config_path)))

    merged.update(_load_env())

    # Explicit overrides (drop None so they don't overwrite real values)
    merged.update({k: v for k, v in overrides.items() if v is not None})

    # Drop null-valued YAML sentinels that Pydantic would reject
    merged = {k: v for k, v in merged.items() if v is not None}

    try:
        return NanoIndexConfig(**merged)
    except Exception as exc:
        raise ConfigError(f"Invalid configuration: {exc}") from exc
