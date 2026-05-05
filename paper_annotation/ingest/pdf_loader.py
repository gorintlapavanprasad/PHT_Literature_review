"""PDF loading utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import fitz  # pymupdf


@dataclass(frozen=True)
class PageText:
    """Represents text extracted from a page."""

    page_number: int
    text: str


def load_pdf_text(pdf_path: Path) -> List[PageText]:
    """Load text from a PDF file page by page."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: List[PageText] = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text")
            pages.append(PageText(page_number=i + 1, text=text))
    return pages


if __name__ == "__main__":
    sample = Path(__file__).resolve().parents[1] / "data" / "pdfs"
    pdfs = list(sample.glob("*.pdf"))
    if pdfs:
        pages = load_pdf_text(pdfs[0])
        print(f"Loaded {len(pages)} pages from {pdfs[0].name}")
        print(pages[0].text[:500])
    else:
        print("No PDFs found for testing.")
