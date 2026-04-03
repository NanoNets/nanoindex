"""PDF utilities backed by PyMuPDF (fitz).

Page splitting for parallel extraction and page-image rendering for
the vision pipeline.
"""

from __future__ import annotations

import base64
from pathlib import Path

import fitz  # PyMuPDF

from nanoindex.models import BoundingBox


def get_page_count(pdf_path: str | Path) -> int:
    """Return the number of pages in a PDF file."""
    doc = fitz.open(str(pdf_path))
    count = len(doc)
    doc.close()
    return count


def split_pdf_pages(pdf_path: str | Path) -> list[tuple[int, bytes]]:
    """Split a PDF into single-page PDFs held in memory.

    Returns a list of ``(page_number, pdf_bytes)`` tuples where
    *page_number* is **1-based**.
    """
    doc = fitz.open(str(pdf_path))
    pages: list[tuple[int, bytes]] = []
    for idx in range(len(doc)):
        single = fitz.open()
        single.insert_pdf(doc, from_page=idx, to_page=idx)
        pages.append((idx + 1, single.tobytes()))
        single.close()
    doc.close()
    return pages


def render_page(
    pdf_path: str | Path,
    page_number: int,
    *,
    dpi: int = 150,
) -> bytes:
    """Render a single page (1-based) as a PNG byte string."""
    doc = fitz.open(str(pdf_path))
    page = doc[page_number - 1]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    png_bytes = pix.tobytes("png")
    doc.close()
    return png_bytes


def render_pages(
    pdf_path: str | Path,
    page_numbers: list[int],
    *,
    dpi: int = 150,
    output_dir: str | Path | None = None,
) -> list[str]:
    """Render multiple pages, saving PNGs to *output_dir* or returning base64 data URIs."""
    results: list[str] = []
    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for pn in page_numbers:
        page = doc[pn - 1]
        pix = page.get_pixmap(matrix=mat)
        png_bytes = pix.tobytes("png")

        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            fp = out / f"page_{pn}.png"
            fp.write_bytes(png_bytes)
            results.append(str(fp))
        else:
            b64 = base64.b64encode(png_bytes).decode()
            results.append(f"data:image/png;base64,{b64}")

    doc.close()
    return results


def render_region(
    pdf_path: str | Path,
    bbox: BoundingBox,
    *,
    dpi: int = 200,
) -> bytes:
    """Render a cropped region defined by a ``BoundingBox`` as a PNG."""
    doc = fitz.open(str(pdf_path))
    page = doc[bbox.page - 1]
    page_rect = page.rect

    clip = fitz.Rect(
        page_rect.width * bbox.x,
        page_rect.height * bbox.y,
        page_rect.width * (bbox.x + bbox.width),
        page_rect.height * (bbox.y + bbox.height),
    )

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, clip=clip)
    png_bytes = pix.tobytes("png")
    doc.close()
    return png_bytes
