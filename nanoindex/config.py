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
    llm_model: str = "nanonets/Nanonets-OCR-s"

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
    use_v2_api: bool = False  # V1 is default (better tree structure). V2 adds bboxes but weaker hierarchy.
    doc_mode: str = "auto"    # "auto", "tree", "table", "form" — auto-detects by default
    split_strategy: str = "hybrid"  # "heuristic", "llm", or "hybrid"

    add_summaries: bool = True
    add_doc_description: bool = False
    add_node_text: bool = False
    summary_model: str | None = None

    # Graph + embedding settings
    # Graph requires a reasoning LLM (set reasoning_llm_model to enable)
    # Embeddings use a local model (no API key needed)
    build_graph: bool = False         # Set True to build entity graph during indexing
    build_embeddings: bool = False    # Enable for fast mode retrieval
    embedding_model: str = "local:all-MiniLM-L6-v2"
    embedding_api_key: str | None = None
    embedding_base_url: str = "https://api.openai.com/v1"
    graph_hops: int = 2               # Expansion depth for graph traversal
    fast_top_k_embed: int = 20        # Tier 1: embedding candidates
    fast_top_k_final: int = 10        # Tier 2: LLM final selection

    @model_validator(mode="after")
    def _resolve_defaults(self) -> "NanoIndexConfig":
        if self.llm_api_key is None and self.nanonets_api_key is not None:
            self.llm_api_key = self.nanonets_api_key
        # Graph building is opt-in (set build_graph=True explicitly)
        return self

    def require_nanonets_key(self) -> str:
        if not self.nanonets_api_key:
            raise ConfigError(
                "NANONETS_API_KEY is required for document parsing.\n\n"
                "  Get a free key (10K pages) at https://docstrange.nanonets.com/app\n"
                "  Then: export NANONETS_API_KEY=your_key\n"
            )
        return self.nanonets_api_key

    def require_llm_key(self) -> str:
        key = self.llm_api_key or self.nanonets_api_key
        if not key:
            raise ConfigError(
                "No LLM API key found. NanoIndex requires two API keys:\n\n"
                "  1. NANONETS_API_KEY — for document parsing (OCR)\n"
                "     Get a free key at https://docstrange.nanonets.com/app\n\n"
                "  2. An LLM API key — for answering questions\n"
                "     Set one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY\n"
                "     Or pass llm='anthropic:claude-sonnet-4-6' to NanoIndex()\n\n"
                "Example:\n"
                "  ni = NanoIndex(llm='anthropic:claude-sonnet-4-6')\n"
            )
        return key

    def require_reasoning_llm(self) -> str:
        """Ensure a reasoning LLM is configured for querying."""
        if not self.reasoning_llm_model:
            raise ConfigError(
                "No LLM configured for answering questions.\n\n"
                "NanoIndex requires two API keys to work:\n\n"
                "  1. NANONETS_API_KEY — for document parsing (OCR)\n"
                "     Get a free key at https://docstrange.nanonets.com/app\n\n"
                "  2. An LLM API key — for answering questions\n"
                "     Set one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY\n"
                "     Or pass llm='anthropic:claude-sonnet-4-6' to NanoIndex()\n\n"
                "Example:\n"
                "  export NANONETS_API_KEY=your_nanonets_key\n"
                "  export ANTHROPIC_API_KEY=your_anthropic_key\n"
                "  ni = NanoIndex()  # auto-detects both keys\n"
            )
        return self.reasoning_llm_model


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
