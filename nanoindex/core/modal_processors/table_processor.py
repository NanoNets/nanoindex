"""Table modal processor — sends table markdown to an LLM for analysis."""

from __future__ import annotations

import logging

from nanoindex.core.llm import LLMClient
from nanoindex.core.modal_processors.base import BaseModalProcessor, ModalProcessorResult
from nanoindex.models import Entity, ModalContent, Relationship

logger = logging.getLogger(__name__)

_PROMPT = """\
Analyze the following table and provide the information in the exact format below.

Context from surrounding document text:
{context}

Table content:
{table}

Respond ONLY in this format:
NAME: <short descriptive name for the table>
DESCRIPTION: <1-3 sentence description of what the table contains>
ENTITIES: <comma-separated list of key entities/concepts found in the table>
"""


class TableModalProcessor(BaseModalProcessor):
    """Sends table markdown to an LLM and creates a graph entity from the response."""

    content_type: str = "table"

    async def process(
        self,
        item: ModalContent,
        llm: LLMClient,
        parent_node_id: str,
    ) -> ModalProcessorResult | None:
        if not item.content:
            logger.warning("Table has no content, skipping")
            return None

        context = item.surrounding_text or item.caption or "(no context available)"
        prompt_text = _PROMPT.format(context=context, table=item.content)

        messages = [{"role": "user", "content": prompt_text}]

        try:
            response = await llm.chat(messages)
        except Exception:
            logger.exception("LLM call failed for table processing")
            return None

        name, description, entities_str = _parse_response(response)
        if not name:
            name = f"Table (page {item.page})"

        entity = Entity(
            name=name,
            entity_type="Table",
            description=description,
            source_node_ids=[parent_node_id],
        )

        relationship = Relationship(
            source=name,
            target=parent_node_id,
            keywords="belongs_to",
            description=f"Table '{name}' belongs to node {parent_node_id}",
            source_node_ids=[parent_node_id],
        )

        return ModalProcessorResult(entity=entity, relationships=[relationship])


def _parse_response(text: str) -> tuple[str, str, str]:
    """Parse NAME / DESCRIPTION / ENTITIES from the LLM response."""
    name = ""
    description = ""
    entities = ""
    for line in text.splitlines():
        line_stripped = line.strip()
        upper = line_stripped.upper()
        if upper.startswith("NAME:"):
            name = line_stripped[len("NAME:") :].strip()
        elif upper.startswith("DESCRIPTION:"):
            description = line_stripped[len("DESCRIPTION:") :].strip()
        elif upper.startswith("ENTITIES:"):
            entities = line_stripped[len("ENTITIES:") :].strip()
    return name, description, entities
