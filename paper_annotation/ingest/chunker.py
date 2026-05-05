"""Chunking utilities."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterable, List, Sequence, Tuple

from ingest.pdf_loader import PageText


@dataclass(frozen=True)
class TextChunk:
    """Represents a chunk of text with metadata."""

    paper_id: str
    chunk_id: str
    text: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int


def _build_page_offsets(pages: Sequence[PageText]) -> List[Tuple[int, int, int]]:
    """Create (page_number, start_offset, end_offset) entries."""
    offsets: List[Tuple[int, int, int]] = []
    cursor = 0
    for page in pages:
        start = cursor
        cursor += len(page.text) + 1
        end = cursor
        offsets.append((page.page_number, start, end))
    return offsets


def chunk_text(paper_id: str, pages: Iterable[PageText], chunk_size: int, chunk_overlap: int) -> List[TextChunk]:
    """Chunk text from pages into overlapping segments."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    pages_list = list(pages)
    full_text = "\n".join(p.text for p in pages_list)
    logging.info("Chunker received %s pages and %s characters", len(pages_list), len(full_text))
    offsets = _build_page_offsets(pages_list)
    chunks: List[TextChunk] = []
    start = 0
    index = 0
    while start < len(full_text):
        end = min(start + chunk_size, len(full_text))
        chunk_text_value = full_text[start:end]
        covered_pages = [p for p, s, e in offsets if not (end <= s or start >= e)]
        page_start = min(covered_pages) if covered_pages else 1
        page_end = max(covered_pages) if covered_pages else 1
        chunks.append(
            TextChunk(
                paper_id=paper_id,
                chunk_id=f"chunk_{index}",
                text=chunk_text_value,
                page_start=page_start,
                page_end=page_end,
                char_start=start,
                char_end=end,
            )
        )
        index += 1
        next_start = end - chunk_overlap
        if next_start <= start:
            logging.warning("Chunker detected no forward progress at index %s; stopping.", index)
            break
        start = next_start
        if start < 0:
            start = 0
        if start >= len(full_text):
            break
    return chunks


if __name__ == "__main__":
    pages = [PageText(page_number=1, text="Page one text."), PageText(page_number=2, text="Page two text.")]
    chunks = chunk_text("sample", pages, chunk_size=20, chunk_overlap=5)
    print(len(chunks))
    print(chunks[0])
