"""Standalone script for ingesting and embedding papers only."""
from __future__ import annotations

import argparse
from pathlib import Path
import logging
import time

from dotenv import load_dotenv

from config import load_config
from ingest.pdf_loader import load_pdf_text
from ingest.text_cleaner import clean_text
from ingest.chunker import chunk_text, PageText
from ingest.embedder import build_faiss_index, save_index
from ingest.embedding_service import EmbeddingService, EmbeddingConfig
from pipeline.concurrency import run_bounded


def _paper_id_from_path(path: Path) -> str:
    """Derive a stable paper ID from the PDF filename."""
    return path.stem


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest and embed papers into FAISS indexes (no annotation)."
    )
    parser.add_argument(
        "--papers",
        type=str,
        help="Specific PDF file or directory to ingest (optional; defaults to config.pdf_dir).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed papers even if indexes already exist.",
    )
    return parser.parse_args()


def ingest_paper(pdf_path: Path, config, embedding_service: EmbeddingService, force: bool = False) -> Path:
    """Ingest a paper and persist its FAISS index and chunks."""
    paper_id = _paper_id_from_path(pdf_path)
    index_path = config.index_dir / f"{paper_id}.faiss"
    chunks_path = config.index_dir / f"{paper_id}.chunks.json"

    if index_path.exists() and chunks_path.exists() and not force:
        logging.info("FAISS index found for %s, skipping embedding step", pdf_path.name)
        return index_path

    if force:
        logging.info("Force re-embedding %s", pdf_path.name)

    step_start = time.perf_counter()
    pages = load_pdf_text(pdf_path)
    logging.info("Loaded %s pages for %s in %.2fs", len(pages), pdf_path.name, time.perf_counter() - step_start)

    step_start = time.perf_counter()
    cleaned_pages = [clean_text(p.text) for p in pages]
    cleaned_page_objs = [
        PageText(page_number=p.page_number, text=cleaned_pages[i]) for i, p in enumerate(pages)
    ]
    logging.info("Cleaned text for %s in %.2fs", pdf_path.name, time.perf_counter() - step_start)

    step_start = time.perf_counter()
    chunks = chunk_text(
        paper_id, cleaned_page_objs, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
    )
    logging.info("Chunked %s into %s chunks in %.2fs", pdf_path.name, len(chunks), time.perf_counter() - step_start)

    step_start = time.perf_counter()
    embedded = build_faiss_index(chunks, embedding_service=embedding_service)
    logging.info("Embedded %s chunks for %s in %.2fs", len(chunks), pdf_path.name, time.perf_counter() - step_start)

    step_start = time.perf_counter()
    save_index(embedded, index_path=index_path, chunks_path=chunks_path)
    logging.info("Saved index for %s in %.2fs", pdf_path.name, time.perf_counter() - step_start)
    return index_path


if __name__ == "__main__":
    load_dotenv()
    config = load_config()
    args = _parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Determine which PDFs to process
    if args.papers:
        papers_path = Path(args.papers)
        if papers_path.is_dir():
            pdfs = list(papers_path.glob("*.pdf"))
        else:
            pdfs = [papers_path]
    else:
        pdfs = list(config.pdf_dir.glob("*.pdf"))

    if not pdfs:
        raise SystemExit("No PDFs found.")

    logging.info("Found %s PDFs to ingest", len(pdfs))

    embedding_service = EmbeddingService(
        EmbeddingConfig(
            model_name=config.embedding_model_name,
            cache_path=config.embedding_cache_path,
        )
    )

    def _ingest_worker(path: Path) -> Path:
        pdf_start = time.perf_counter()
        logging.info("Ingesting %s", path.name)
        ingest_paper(path, config, embedding_service, force=args.force)
        ingest_seconds = time.perf_counter() - pdf_start
        logging.info("Ingestion time for %s: %.2fs", path.name, ingest_seconds)
        return path

    run_bounded(pdfs, _ingest_worker, max_workers=config.ingest_concurrency)

    print(f"Ingestion complete. Processed {len(pdfs)} papers.")
