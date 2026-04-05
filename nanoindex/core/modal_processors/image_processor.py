"""Image modal processor — sends images to a VLM for description and entity extraction."""

from __future__ import annotations

import base64
import logging
import mimetypes
import os

from nanoindex.core.llm import LLMClient
from nanoindex.core.modal_processors.base import BaseModalProcessor, ModalProcessorResult
from nanoindex.models import Entity, ModalContent, Relationship

logger = logging.getLogger(__name__)

_PROMPT = """\
Analyze this image and provide the following information in the exact format below.

Context from surrounding document text:
{context}

Respond ONLY in this format:
NAME: <short descriptive name for the image>
DESCRIPTION: <1-3 sentence description of what the image shows>
ENTITIES: <comma-separated list of key entities/concepts visible in the image>
"""


class ImageModalProcessor(BaseModalProcessor):
    """Sends an image to a VLM and creates a graph entity from the response."""

    content_type: str = "image"

    async def process(
        self,
        item: ModalContent,
        llm: LLMClient,
        parent_node_id: str,
    ) -> ModalProcessorResult | None:
        if not item.image_path or not os.path.isfile(item.image_path):
            logger.warning("Image file not found: %s", item.image_path)
            return None

        # Read and base64-encode the image
        try:
            with open(item.image_path, "rb") as f:
                image_bytes = f.read()
        except OSError:
            logger.warning("Failed to read image file: %s", item.image_path)
            return None

        mime_type = mimetypes.guess_type(item.image_path)[0] or "image/png"
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64}"

        context = item.surrounding_text or item.caption or "(no context available)"
        prompt_text = _PROMPT.format(context=context)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        try:
            response = await llm.chat(messages)
        except Exception:
            logger.exception("VLM call failed for image: %s", item.image_path)
            return None

        name, description, entities_str = _parse_response(response)
        if not name:
            name = os.path.basename(item.image_path)


        entity = Entity(
            name=name,
            entity_type="Image",
            description=description,
            source_node_ids=[parent_node_id],
        )

        relationship = Relationship(
            source=name,
            target=parent_node_id,
            keywords="belongs_to",
            description=f"Image '{name}' belongs to node {parent_node_id}",
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
            name = line_stripped[len("NAME:"):].strip()
        elif upper.startswith("DESCRIPTION:"):
            description = line_stripped[len("DESCRIPTION:"):].strip()
        elif upper.startswith("ENTITIES:"):
            entities = line_stripped[len("ENTITIES:"):].strip()
    return name, description, entities
