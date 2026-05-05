# Paper Annotation System (Non-Technical Guide)

This folder contains a complete system that reads research PDFs, breaks them into small pieces, searches for relevant evidence, and asks an AI model to answer a fixed set of questions. It saves the answers in a simple text format that can be checked later.

## Architecture Overview

Pipeline (high level):
1) Load PDF pages → clean text → chunk text (with page + character offsets)
2) Generate embeddings once via a central EmbeddingService (with cache)
3) Build and persist a FAISS index per paper
4) Hybrid retrieval: lexical filter (token overlap) → vector rerank
5) LLM answers questions with strict JSON output
6) Atomic JSONL write to results

Key additions:
- EmbeddingService: centralized embeddings and cache
- Hybrid retrieval: lexical + vector
- Atomic writes: prevent partial/corrupt outputs

## What each file does (simple descriptions)

Core entry points:
- [main.py](main.py): Runs the full pipeline (embedding + annotation) or individual stages.
- [ingest_papers.py](ingest_papers.py): Standalone script to embed/ingest papers only (optional).

Configuration:
- [config.py](config.py): Central settings (chunking, retrieval, concurrency, model names, cache path, API key).

Questions:
- [questions.py](questions.py): Framework questions replicated from worker_model.

PDF ingestion:
- [ingest/pdf_loader.py](ingest/pdf_loader.py): Reads each PDF and extracts text page by page.
- [ingest/text_cleaner.py](ingest/text_cleaner.py): Cleans the extracted text.
- [ingest/chunker.py](ingest/chunker.py): Splits text into overlapping chunks with page + character offsets.
- [ingest/embedding_service.py](ingest/embedding_service.py): Central embedding model + cache.
- [ingest/embedder.py](ingest/embedder.py): Builds FAISS indexes using EmbeddingService.

Retrieval:
- [retrieval/retriever.py](retrieval/retriever.py): Hybrid lexical + vector retrieval.

LLM layer:
- [llm/base.py](llm/base.py): Abstract LLM interface.
- [llm/openai_llm.py](llm/openai_llm.py): OpenAI implementation with retries.
- [llm/prompts.py](llm/prompts.py): Prompt builder (strict JSON output).

Pipeline and concurrency:
- [pipeline/annotate.py](pipeline/annotate.py): One record per question per paper.
- [pipeline/concurrency.py](pipeline/concurrency.py): Bounded concurrency with semaphore logs.

Storage and evaluation:
- [storage/writer.py](storage/writer.py): Atomic JSONL writes.
- [evaluation/metrics.py](evaluation/metrics.py): Evaluation metrics (citation coverage, hit rate, stability).

Data folders:
- [papers/](../papers/): Put your PDF files here.
- [data/indexes/](data/indexes/): Saved search indexes live here.
- [data/results/](data/results/): Final results are saved here.

## One-time setup (easy)

1) Make sure you have Python installed.
2) Install dependencies from [requirements.txt](requirements.txt).
3) Put your API keys in [paper_annotation/.env](paper_annotation/.env):
   - `OPENAI_API_KEY=...` (required for OpenAI models)
   - `GEMINI_API_KEY=...` (required for Gemini models)
   - `HF_TOKEN=...` (required for Hugging Face models)

## How to run the full system (end-to-end)

1) Put your PDF files into [papers/](../papers/).
2) From the paper_annotation folder, run `python main.py` to run the full pipeline.

### Running individual stages (modular execution)

The pipeline can be broken into two stages: **ingest** (embedding) and **annotate** (LLM).

#### Option 1: Use the standalone ingest script
```bash
# Embed all papers from the default directory
python ingest_papers.py

# Re-embed papers even if indexes already exist
python ingest_papers.py --force

# Embed specific papers
python ingest_papers.py --papers papers/myfile.pdf
python ingest_papers.py --papers papers/subfolder
```

#### Option 2: Use main.py with --stage flag
```bash
# Just embedding (skip annotation)
python main.py --stage ingest

# Just embedding with force re-embed
python main.py --stage ingest --force-ingest

# Just annotation (skip embedding, requires pre-embedded papers)
python main.py --stage annotate --model openai

# Full pipeline (both ingest and annotate, default)
python main.py --stage all
python main.py  # equivalent to --stage all
```

### Model selection examples
```bash
# Run one model
python main.py --model openai
python main.py --stage annotate --model gemini

# Run multiple model families
python main.py --model gemini --model llama
python main.py --stage annotate --model openai --model mistral

# Run a specific Hugging Face model
python main.py --model mistralai/Mistral-7B-Instruct-v0.3
```

### Typical workflow with stages
```bash
# Phase 1: Embed all papers once (can take a while)
python main.py --stage ingest

# Phase 2: Run different models independently (faster, can run in parallel)
python main.py --stage annotate --model openai
python main.py --stage annotate --model gemini
python main.py --stage annotate --model llama
```

### What the system does
1. **Ingest stage**: Reads each PDF, extracts text, cleans it, chunks it, generates embeddings, and builds a FAISS search index for each paper.
2. **Annotate stage**: Loads pre-built indexes, retrieves relevant chunks for each question, and asks the LLM to answer with supporting evidence.
3. Writes results into `data/results/<model_dir>/<paper_id>.jsonl` with one line per question.

### Expected output
- For each PDF/model pair, a JSONL file appears in `data/results/<model_dir>/` with one line per question.
- Directory names are normalized by model (for example: `openai`, `llama`, `mistral`).
- Each line contains the answer, confidence, evidence chunks, and reasoning.

## Configuration Notes

- Embedding configuration: [config.py](config.py) → `embedding_model_name` and `embedding_cache_path`.
- Supported models are centrally configured in [config.py](config.py) under `MODELS`.
- The pipeline skips model/paper work when `data/results/<model_dir>/<paper_id>.jsonl` already exists.
- Cache behavior: embeddings are cached by SHA-256 of text in a local SQLite file for reuse.

## Performance Notes

- Embedding cache: once cached, reruns avoid recomputing embeddings.
- Parallel ingestion: PDF loading and preprocessing run concurrently per paper.
- Hybrid retrieval: lexical filtering improves relevance before vector rerank.

## Extensibility Notes

- Add a new LLM backend: implement `BaseLLM` in [llm/base.py](llm/base.py) and swap in [main.py](main.py).
- Change vector store: update [ingest/embedder.py](ingest/embedder.py) and [retrieval/retriever.py](retrieval/retriever.py) to use a different index.
- Extend evaluation: add metrics in [evaluation/metrics.py](evaluation/metrics.py).

## How to verify the full flow step-by-step (non-technical)

1) Place a PDF in [papers/](../papers/).
2) Run python main.py.
3) Check that new files appear in [data/indexes/](data/indexes/).
4) Check that a JSONL file appears in [data/results/openai/](data/results/openai/).
5) Open the JSONL file in any text editor; each line is one answer with evidence and chunk metadata.

## Notes for non-technical users

- If the system says no PDFs found, make sure the PDF files are inside [papers/](../papers/).
- If the system says the API key is missing, set `OPENAI_API_KEY` in [paper_annotation/.env](paper_annotation/.env).
- If you rerun the system, it will reuse saved indexes and cached embeddings and run faster.
