"""Base class and result type for modal processors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from nanoindex.models import Entity, Relationship


@dataclass
class ModalProcessorResult:
    """Result of processing a single multimodal content item."""

    entity: Entity
    relationships: list[Relationship]


class BaseModalProcessor(ABC):
    """Abstract base for processors that convert non-text content into graph nodes."""

    content_type: str = ""

    @abstractmethod
    async def process(
        self,
        item: "ModalContent",  # noqa: F821 — forward ref to avoid circular import
        llm: "LLMClient",  # noqa: F821
        parent_node_id: str,
    ) -> ModalProcessorResult | None:
        """Process a ModalContent item and return an entity + relationships, or None on failure."""
        ...
