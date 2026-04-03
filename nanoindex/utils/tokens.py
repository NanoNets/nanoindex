"""Token counting utilities backed by tiktoken."""

from __future__ import annotations

import tiktoken

_DEFAULT_ENCODING = "cl100k_base"

_encoder: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding(_DEFAULT_ENCODING)
    return _encoder


def count_tokens(text: str) -> int:
    """Return the number of tokens in *text* using cl100k_base."""
    return len(_get_encoder().encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate *text* to at most *max_tokens*."""
    enc = _get_encoder()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])
