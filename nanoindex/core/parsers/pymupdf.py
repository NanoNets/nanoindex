"""PyMuPDF-based document parser -- fully open-source, no API key needed."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import fitz  # PyMuPDF

from nanoindex.core.parsers.base import BaseParser
from nanoindex.models import (
    BoundingBox,
    ModalContent,
    PageDimensions,
    ParsedDocument,
)

logger = logging.getLogger(__name__)


class PyMuPDFParser(BaseParser):
    """Extract text, images, and tables from PDFs using PyMuPDF (fitz).

    This parser is entirely local and requires no external API keys.
    """

    name: str = "pymupdf"

    # Minimum image dimensions (pixels) to keep -- skip tiny icons/bullets.
    min_image_width: int = 50
    min_image_height: int = 50

    async def parse(self, file_path: Path) -> ParsedDocument:
        file_path = Path(file_path)
        t0 = time.perf_counter()

        doc = fitz.open(str(file_path))

        pages: list[str] = []
        modal_contents: list[ModalContent] = []
        bounding_boxes: list[BoundingBox] = []
        page_dimensions: list[PageDimensions] = []

        # Directory to store extracted images, sibling to the PDF.
        asset_dir = file_path.parent / f"{file_path.stem}_assets"

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_index = page_num + 1  # 1-based

            # -- page dimensions ------------------------------------------------
            rect = page.rect
            page_dimensions.append(
                PageDimensions(
                    page=page_index,
                    width=int(rect.width),
                    height=int(rect.height),
                )
            )

            # -- text -----------------------------------------------------------
            text = page.get_text("text")
            pages.append(text)

            # -- images ---------------------------------------------------------
            image_list = page.get_images(full=True)
            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                except Exception:
                    logger.debug("Could not extract image xref=%s on page %s", xref, page_index)
                    continue

                img_bytes = base_image["image"]
                img_ext = base_image.get("ext", "png")
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)

                if width < self.min_image_width or height < self.min_image_height:
                    continue

                # Save image to disk.
                asset_dir.mkdir(parents=True, exist_ok=True)
                img_filename = f"page{page_index}_img{img_idx}.{img_ext}"
                img_path = asset_dir / img_filename
                img_path.write_bytes(img_bytes)

                modal_contents.append(
                    ModalContent(
                        content_type="image",
                        page=page_index,
                        image_path=str(img_path),
                        metadata={"width": width, "height": height, "ext": img_ext},
                    )
                )

                bounding_boxes.append(
                    BoundingBox(
                        page=page_index,
                        x=0,
                        y=0,
                        width=width,
                        height=height,
                        region_type="image",
                    )
                )

            # -- tables ---------------------------------------------------------
            try:
                tables = page.find_tables()
                for tbl_idx, table in enumerate(tables):
                    # table.extract() returns list[list[str|None]]
                    rows = table.extract()
                    if not rows:
                        continue

                    # Build a simple markdown representation.
                    md_lines: list[str] = []
                    header = rows[0]
                    md_lines.append("| " + " | ".join(c or "" for c in header) + " |")
                    md_lines.append("| " + " | ".join("---" for _ in header) + " |")
                    for row in rows[1:]:
                        md_lines.append("| " + " | ".join(c or "" for c in row) + " |")
                    md_table = "\n".join(md_lines)

                    modal_contents.append(
                        ModalContent(
                            content_type="table",
                            page=page_index,
                            content=md_table,
                            metadata={"table_index": tbl_idx, "row_count": len(rows)},
                        )
                    )

                    # Bounding box from table bbox (x0, y0, x1, y1).
                    bbox = table.bbox
                    if bbox:
                        bounding_boxes.append(
                            BoundingBox(
                                page=page_index,
                                x=bbox[0] / rect.width if rect.width else 0,
                                y=bbox[1] / rect.height if rect.height else 0,
                                width=(bbox[2] - bbox[0]) / rect.width if rect.width else 0,
                                height=(bbox[3] - bbox[1]) / rect.height if rect.height else 0,
                                region_type="table",
                            )
                        )
            except Exception:
                logger.debug("Table extraction failed on page %s", page_index, exc_info=True)

        doc.close()

        # Combine per-page text into a single markdown string.
        markdown = "\n\n".join(pages)

        elapsed = time.perf_counter() - t0

        return ParsedDocument(
            markdown=markdown,
            pages=pages,
            page_count=len(pages),
            modal_contents=modal_contents,
            bounding_boxes=bounding_boxes,
            page_dimensions=page_dimensions,
            processing_time=elapsed,
            parser_name=self.name,
        )
