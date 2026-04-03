"""Provider-agnostic LLM client.

Supports two backends transparently:
  - **OpenAI-compatible** (default): Nanonets, OpenAI, Gemini, vLLM, Ollama, etc.
  - **Anthropic** (auto-detected when model starts with ``claude``):
    Uses the ``anthropic`` SDK directly for native Claude support.

Completely independent from the Nanonets-specific client in ``client.py``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from nanoindex.exceptions import RetrievalError

logger = logging.getLogger(__name__)

_MAX_RETRIES = 4
_RETRY_BACKOFF = 2.0  # seconds, doubled each attempt


def _is_anthropic_model(model: str) -> bool:
    return model.startswith("claude")


class LLMClient:
    """Async wrapper around OpenAI-compatible or Anthropic chat completions."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://extraction-api.nanonets.com/v1",
        model: str = "nanonets/Nanonets-OCR-s",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._is_anthropic = _is_anthropic_model(model)

        if self._is_anthropic:
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                raise ImportError(
                    "pip install anthropic  — required for Claude models"
                )
            self._anthropic = AsyncAnthropic(api_key=api_key)
            self._openai = None
        else:
            self._openai = AsyncOpenAI(api_key=api_key, base_url=base_url)
            self._anthropic = None

    @property
    def model(self) -> str:
        return self._model

    # ------------------------------------------------------------------
    # Core call
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat completion request and return the assistant message.

        Retries on transient server errors (5xx, 429, overloaded) with
        exponential backoff.  Dispatches to Anthropic or OpenAI backend
        automatically.
        """
        effective_model = model or self._model
        if self._is_anthropic or _is_anthropic_model(effective_model):
            return await self._chat_anthropic(
                messages, model=effective_model,
                temperature=temperature, max_tokens=max_tokens,
            )
        return await self._chat_openai(
            messages, model=effective_model,
            temperature=temperature, max_tokens=max_tokens,
        )

    # ---- OpenAI-compatible backend ------------------------------------

    async def _chat_openai(
        self, messages, *, model, temperature, max_tokens,
    ) -> str:
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = await self._openai.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature if temperature is not None else self._temperature,
                    max_tokens=max_tokens or self._max_tokens,
                    stream=False,
                )
                content = resp.choices[0].message.content or ""
                return content.strip()
            except Exception as exc:
                last_exc = exc
                err_str = str(exc)
                is_transient = any(code in err_str for code in ("500", "502", "503", "429", "Server", "overloaded"))
                if is_transient and attempt < _MAX_RETRIES:
                    wait = _RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(
                        "LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                        attempt + 1, _MAX_RETRIES + 1, err_str[:120], wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise RetrievalError(f"LLM call failed after {attempt + 1} attempts: {exc}") from exc
        raise RetrievalError(f"LLM call failed: {last_exc}") from last_exc

    # ---- Anthropic (Claude) backend -----------------------------------

    @staticmethod
    def _convert_content_for_anthropic(content):
        """Convert OpenAI-style multimodal content blocks to Anthropic format.

        OpenAI uses ``{"type": "image_url", "image_url": {"url": "data:...;base64,..."}}``
        Anthropic uses ``{"type": "image", "source": {"type": "base64", ...}}``.
        """
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return content

        converted: list[dict] = []
        for block in content:
            if block.get("type") == "image_url":
                url = block["image_url"]["url"]
                if url.startswith("data:"):
                    header, data = url.split(",", 1)
                    media_type = header.split(";")[0].replace("data:", "")
                    converted.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": data,
                        },
                    })
                else:
                    converted.append({
                        "type": "image",
                        "source": {"type": "url", "url": url},
                    })
            elif block.get("type") == "text":
                converted.append({"type": "text", "text": block["text"]})
            else:
                converted.append(block)
        return converted

    async def _chat_anthropic(
        self, messages, *, model, temperature, max_tokens,
    ) -> str:
        system_parts: list[str] = []
        user_msgs: list[dict] = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(
                    msg["content"] if isinstance(msg["content"], str) else str(msg["content"])
                )
            else:
                user_msgs.append({
                    "role": msg["role"],
                    "content": self._convert_content_for_anthropic(msg["content"]),
                })

        if not user_msgs:
            user_msgs = [{"role": "user", "content": ""}]

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens or self._max_tokens,
            "messages": user_msgs,
        }
        temp = temperature if temperature is not None else self._temperature
        if temp > 0:
            kwargs["temperature"] = temp
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = await self._anthropic.messages.create(**kwargs)
                return resp.content[0].text.strip()
            except Exception as exc:
                last_exc = exc
                err_str = str(exc)
                is_transient = any(tok in err_str for tok in ("500", "502", "503", "529", "overloaded", "rate"))
                if is_transient and attempt < _MAX_RETRIES:
                    wait = _RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(
                        "Anthropic call failed (attempt %d/%d): %s — retrying in %.1fs",
                        attempt + 1, _MAX_RETRIES + 1, err_str[:120], wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise RetrievalError(f"Anthropic call failed after {attempt + 1} attempts: {exc}") from exc
        raise RetrievalError(f"Anthropic call failed: {last_exc}") from last_exc

    # ---- Streaming (OpenAI-compatible only) ---------------------------

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream a chat completion, yielding text deltas."""
        if self._is_anthropic:
            async for token in self._stream_anthropic(messages, model=model, temperature=temperature, max_tokens=max_tokens):
                yield token
            return
        try:
            stream = await self._openai.chat.completions.create(
                model=model or self._model,
                messages=messages,
                temperature=temperature if temperature is not None else self._temperature,
                max_tokens=max_tokens or self._max_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:
            raise RetrievalError(f"LLM streaming call failed: {exc}") from exc

    async def _stream_anthropic(self, messages, *, model, temperature, max_tokens):
        system_parts: list[str] = []
        user_msgs: list[dict] = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                user_msgs.append({"role": msg["role"], "content": msg["content"]})

        kwargs: dict[str, Any] = {
            "model": model or self._model,
            "max_tokens": max_tokens or self._max_tokens,
            "messages": user_msgs or [{"role": "user", "content": ""}],
        }
        temp = temperature if temperature is not None else self._temperature
        if temp > 0:
            kwargs["temperature"] = temp
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)

        try:
            async with self._anthropic.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as exc:
            raise RetrievalError(f"Anthropic streaming failed: {exc}") from exc

    async def close(self) -> None:
        if self._openai:
            await self._openai.close()
        if self._anthropic:
            await self._anthropic.close()
