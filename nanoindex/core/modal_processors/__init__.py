"""Modal processors — convert non-text content into graph entities.

Usage::

    from nanoindex.core.modal_processors import get_processor

    processor = get_processor("image")
    result = await processor.process(item, llm, parent_node_id)
"""

from __future__ import annotations

from nanoindex.core.modal_processors.base import BaseModalProcessor, ModalProcessorResult
from nanoindex.core.modal_processors.image_processor import ImageModalProcessor
from nanoindex.core.modal_processors.table_processor import TableModalProcessor

__all__ = [
    "BaseModalProcessor",
    "ModalProcessorResult",
    "PROCESSORS",
    "get_processor",
]

PROCESSORS: dict[str, type[BaseModalProcessor]] = {
    "image": ImageModalProcessor,
    "table": TableModalProcessor,
}


def get_processor(content_type: str) -> BaseModalProcessor | None:
    """Return a processor instance for *content_type*, or ``None`` if unsupported."""
    cls = PROCESSORS.get(content_type)
    if cls is None:
        return None
    return cls()
