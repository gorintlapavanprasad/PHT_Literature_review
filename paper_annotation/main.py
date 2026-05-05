"""Main entry point for paper annotation pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path
import logging
import time
from typing import List
import numpy as np

from dotenv import load_dotenv

from config import load_config, Config, MODELS, all_models, model_output_dir, model_provider
from ingest.pdf_loader import load_pdf_text, PageText
from ingest.text_cleaner import clean_text
from ingest.chunker import chunk_text
from ingest.embedder import build_faiss_index, load_index, save_index
from ingest.embedding_service import EmbeddingService, EmbeddingConfig
from llm.base import BaseLLM
from llm.openai_llm import OpenAILLM, OpenAISettings
from llm.huggingface_llm import HuggingFaceLLM, HuggingFaceSettings
from pipeline.annotate import annotate_question, AnnotationRecord
from pipeline.concurrency import run_bounded
from storage.writer import write_jsonl
from retrieval.retriever import HybridRetriever
from questions import PHT_FRAMEWORK_QUESTIONS


def _paper_id_from_path(path: Path) -> str:
    """Derive a stable paper ID from the PDF filename."""
    return path.stem


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Annotate papers with configured LLM models.")
    parser.add_argument(
        "--model",
        type=str,
        action="append",
        default=None,
        help=(
            "Optional model selector; can be repeated. "
            "Accepts exact model names or aliases: openai, llama, mistral, gemini."
        ),
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["ingest", "annotate", "all"],
        default="all",
        help=(
            "Which pipeline stage to run: "
            "'ingest' (embedding only), 'annotate' (LLM only, requires pre-embedded), or 'all' (default)."
        ),
    )
    parser.add_argument(
        "--force-ingest",
        action="store_true",
        help="Re-embed papers even if indexes already exist (only applicable for 'ingest' or 'all' stage).",
    )
    return parser.parse_args()


def _resolve_models(requested: list[str] | None, configured_models: list[str]) -> list[str]:
    """Resolve CLI selectors into concrete configured model names."""
    if not requested:
        return configured_models

    alias_map = {
        "openai":  MODELS["openai"],
        "gemini":  MODELS["gemini"],
        "groq":    MODELS.get("groq", []),
        "mistral": MODELS.get("mistral", []),
        # Legacy aliases for backwards compatibility with older scripts.
        "llama":   [m for m in MODELS.get("groq", []) if "llama" in m.lower()]
                   + [m for m in MODELS["huggingface"] if "llama" in m.lower()],
    }

    resolved: list[str] = []
    for selector in requested:
        selected_models = alias_map.get(selector, [selector])
        for model_name in selected_models:
            if model_name not in configured_models:
                raise SystemExit(
                    f"Unsupported model '{selector}'. Supported models: {', '.join(configured_models)}. "
                    "Aliases: openai, llama, mistral, gemini"
                )
            if model_name not in resolved:
                resolved.append(model_name)
    return resolved


def _build_llm(model_name: str, config: Config) -> BaseLLM:
    """Instantiate a provider-specific LLM from model name."""
    provider = model_provider(model_name)
    if provider == "openai":
        if not config.openai_api_key:
            raise EnvironmentError("OPENAI_API_KEY is required for OpenAI models.")
        return OpenAILLM(
            OpenAISettings(
                api_key=config.openai_api_key,
                model_name=model_name,
                temperature=config.temperature,
            )
        )
    if provider == "huggingface":
        if not config.huggingface_token:
            raise EnvironmentError("HF_TOKEN (or HUGGINGFACE_API_TOKEN) is required for Hugging Face models.")
        return HuggingFaceLLM(
            HuggingFaceSettings(
                token=config.huggingface_token,
                model_name=model_name,
                temperature=config.temperature,
                timeout_seconds=config.huggingface_request_timeout,
            )
        )
    if provider == "groq":
        from llm.groq_llm import GroqLLM, GroqSettings

        if not config.groq_api_key:
            raise EnvironmentError("GROQ_API_KEY is required for Groq models.")
        logging.info("Running Groq model: %s", model_name)
        return GroqLLM(
            GroqSettings(
                api_key=config.groq_api_key,
                model_name=model_name,
                temperature=config.temperature,
            )
        )
    if provider == "mistral":
        from llm.mistral_llm import MistralLLM, MistralSettings

        if not config.mistral_api_key:
            raise EnvironmentError("MISTRAL_API_KEY is required for Mistral models.")
        logging.info("Running Mistral model: %s", model_name)
        return MistralLLM(
            MistralSettings(
                api_key=config.mistral_api_key,
                model_name=model_name,
                temperature=config.temperature,
            )
        )
    if provider == "gemini":
        from llm.gemini_llm import GeminiLLM, GeminiSettings

        if not config.gemini_api_key:
            raise EnvironmentError("GEMINI_API_KEY is required for Gemini models.")
        logging.info("Running Gemini model: %s", model_name)
        return GeminiLLM(
            GeminiSettings(
                api_key=config.gemini_api_key,
                model=model_name,
                temperature=config.temperature,
            )
        )
    raise ValueError(f"Unsupported provider for model: {model_name}")


def ingest_paper(
    pdf_path: Path,
    config: Config,
    embedding_service: EmbeddingService,
    force: bool = False,
) -> Path:
    """Ingest a paper and persist its FAISS index and chunks."""
    paper_id = _paper_id_from_path(pdf_path)
    index_path = config.index_dir / f"{paper_id}.faiss"
    chunks_path = config.index_dir / f"{paper_id}.chunks.json"

    if index_path.exists() and chunks_path.exists() and not force:
        logging.info("FAISS index found for %s, skipping embedding step", pdf_path.name)
        return index_path

    if force:
        if index_path.exists():
            index_path.unlink()
        if chunks_path.exists():
            chunks_path.unlink()
        logging.info("Force re-embedding %s (existing index/chunks deleted)", pdf_path.name)

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
    chunks = chunk_text(paper_id, cleaned_page_objs, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    logging.info("Chunked %s into %s chunks in %.2fs", pdf_path.name, len(chunks), time.perf_counter() - step_start)

    step_start = time.perf_counter()
    embedded = build_faiss_index(chunks, embedding_service=embedding_service)
    logging.info("Embedded %s chunks for %s in %.2fs", len(chunks), pdf_path.name, time.perf_counter() - step_start)

    step_start = time.perf_counter()
    save_index(embedded, index_path=index_path, chunks_path=chunks_path)
    logging.info("Saved index for %s in %.2fs", pdf_path.name, time.perf_counter() - step_start)
    return index_path


def annotate_paper(
    pdf_path: Path,
    questions: List[str],
    question_embeddings: np.ndarray,
    llm: BaseLLM,
    embedding_service: EmbeddingService,
    config: Config,
    model_name: str,
) -> List[AnnotationRecord]:
    """Annotate a single paper with all questions."""
    paper_id = _paper_id_from_path(pdf_path)
    logging.info("Starting retrieval/annotation for %s", pdf_path.name)
    index_path = config.index_dir / f"{paper_id}.faiss"
    chunks_path = config.index_dir / f"{paper_id}.chunks.json"
    step_start = time.perf_counter()
    embedded = load_index(index_path=index_path, chunks_path=chunks_path)
    logging.info("Loaded index for %s in %.2fs", pdf_path.name, time.perf_counter() - step_start)

    # Pair each (question, embedding) with its index so the annotator can
    # pick the right dimension-specific few-shot example in the new prompt.
    question_payloads = list(enumerate(zip(questions, question_embeddings)))
    retriever = HybridRetriever(embedded, embedding_service, lexical_top_k=config.lexical_top_k)

    def _worker(payload: tuple[int, tuple[str, np.ndarray]]) -> AnnotationRecord:
        q_index, (question, embedding) = payload
        return annotate_question(
            paper_id=paper_id,
            embedded=embedded,
            retriever=retriever,
            question=question,
            query_embedding=embedding,
            llm=llm,
            model_name=model_name,
            prompt_version=config.prompt_version,
            top_k=config.top_k,
            question_index=q_index,
        )

    return run_bounded(question_payloads, _worker, max_workers=config.max_concurrency)


if __name__ == "__main__":
    load_dotenv()
    config = load_config()
    args = _parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    pdfs = list(config.pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("No PDFs found in papers/.")

    embedding_service = EmbeddingService(
        EmbeddingConfig(
            model_name=config.embedding_model_name,
            cache_path=config.embedding_cache_path,
        )
    )

    # Stage 1: Ingestion (embedding)
    if args.stage in ["ingest", "all"]:
        logging.info("Starting ingestion stage...")

        def _ingest_worker(path: Path) -> Path:
            pdf_start = time.perf_counter()
            logging.info("Ingesting %s", path.name)
            ingest_paper(path, config, embedding_service, force=args.force_ingest)

            ingest_seconds = time.perf_counter() - pdf_start
            logging.info("Ingestion time for %s: %.2fs (before API calls)", path.name, ingest_seconds)
            return path

        run_bounded(pdfs, _ingest_worker, max_workers=config.ingest_concurrency)
        logging.info("Ingestion stage complete.")

        if args.stage == "ingest":
            print("Embedding complete. Run with --stage annotate to perform annotations.")
            exit(0)

    # Stage 2: Annotation (requires embedded papers)
    if args.stage in ["annotate", "all"]:
        logging.info("Starting annotation stage...")
        
        questions = PHT_FRAMEWORK_QUESTIONS
        question_embeddings = embedding_service.embed_queries(questions)

        configured_models = all_models()
        models_to_run = _resolve_models(args.model, configured_models)
        llms = {model_name: _build_llm(model_name, config) for model_name in models_to_run}
        logging.info("Models selected: %s", ", ".join(models_to_run))

        for pdf_path in pdfs:
            paper_id = _paper_id_from_path(pdf_path)
            logging.info("Processing paper: %s", paper_id)
            for model_name in models_to_run:
                model_dir = model_output_dir(model_name)
                output_path = config.results_dir / model_dir / f"{paper_id}.jsonl"
                if output_path.exists():
                    logging.info("Skipping %s for %s (results already exist)", model_name, paper_id)
                    continue

                logging.info("Running annotation using %s for %s", model_name, paper_id)
                records = annotate_paper(
                    pdf_path=pdf_path,
                    questions=questions,
                    question_embeddings=question_embeddings,
                    llm=llms[model_name],
                    embedding_service=embedding_service,
                    config=config,
                    model_name=model_name,
                )
                write_jsonl(records, output_path)
                logging.info("Wrote results to %s", output_path)

        logging.info("Annotation stage complete.")

    print("Pipeline complete.")
