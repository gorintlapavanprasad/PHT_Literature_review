# PHT Literature Review — Project Document

_Last updated 2026-11-10. See also `PAPER_NOTES.md` (paper review) and `reports/REPORT.md` (evaluation of the stored results)._

## 1. What this project is

A pipeline plus viewer for a **systematic literature review on Playful Health Technology (PHT)**.
215+ accessibility-research papers (PDFs) are coded on **8 design dimensions** of the PHT framework
(Duval 2022), each on a 1–5 scale:

| # | Dimension | 1 … 5 |
|---|-----------|-------|
| 1 | Fun-first vs utility-first | totally fun first … totally utility first |
| 2 | Play vs game | unstructured play … rigid-rule game |
| 3 | Skill vs chance | all skill … all chance |
| 4 | Solo vs social | solo … entirely social |
| 5 | Sequential vs simultaneous | turn-based … simultaneous |
| 6 | Synchronous vs asynchronous | sync … async |
| 7 | Competitive vs collaborative | competitive … collaborative |
| 8 | Symmetrical vs asymmetrical | symmetrical … asymmetrical |

Four LLMs (GPT-5.1, Gemini 2.5 Flash, Llama 3.3 70B via Groq, Mistral Small) answer each question per
paper using retrieval-augmented generation (RAG). Their answers are compared with each other and (planned)
with human consensus coding from Covidence. Goal of the study: can LLMs reliably assist human coders in
a systematic review, and where do they disagree.

## 2. Architecture

```
PDF ─▶ pdf_loader (PyMuPDF) ─▶ text_cleaner ─▶ chunker (900 chars, 150 overlap, page+offset metadata)
    ─▶ EmbeddingService (BGE-large, SQLite cache) ─▶ FAISS flat-L2 index per paper (data/indexes/)
Question ─▶ HybridRetriever (token-overlap filter ─▶ vector rerank, top_k=15)
         ─▶ prompts.py (role + framework + few-shot + calibrated confidence, strict JSON)
         ─▶ LLM wrapper (openai / gemini / groq / mistral / huggingface) with retries
         ─▶ annotate_question (full-context if ≤30 chunks; retry with 2x top_k on Unknown)
         ─▶ AnnotationRecord ─▶ atomic JSONL (data/results/<provider>/<paper>.jsonl)
pht_ui: FastAPI (REST + SSE streaming) + single-file HTML heatmap frontend
```

Key folders: `paper_annotation/` (pipeline, CLI `main.py`), `pht_ui/` (web UI),
`reference code/` (earlier prototypes), `tests/` (pytest suite added in this pass), `Final Report/`.

## 3. How to run (verified on Windows, Python 3.12)

```bash
py -3.12 -m venv .venv
.venv\Scripts\pip install -r paper_annotation/requirements.txt -r pht_ui/backend/requirements.txt pytest
copy pht_ui\.env.example paper_annotation\.env      # then add your API keys
# tests (no keys, no torch needed – embedder and LLMs are faked)
.venv\Scripts\python -m pytest
# UI
cd pht_ui && ..\.venv\Scripts\python -m uvicorn backend.server:app --port 8765
# batch CLI (needs keys): papers go in <repo>/papers/
cd paper_annotation && ..\.venv\Scripts\python main.py --stage all --model gemini
```

Set `EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2` for a fast ~90 MB model while developing (the default
`BAAI/bge-large-en-v1.5` is ~1.3 GB). Indexes built with one model are not compatible with another —
delete `data/indexes` and `data/embeddings_cache.sqlite` when switching.

## 4. State of the data (measured)

| Provider dir | Papers | Records | Prompt | Notes |
|---|---|---|---|---|
| openai (gpt-5.1) | 214 | 1712 | v1 | 40% `forced_choice` |
| gemini | 217 | 1722 | v1 | 138 forced_choice; **2 papers incomplete** |
| llama (HF) | 214 | 1712 | v1 | **53% forced_choice**, mean conf 0.70 |
| mistral (HF) | 215 | 1720 | v1 | **65% forced_choice**; contains 8 stray `llama-3.1-8b-instant` records |
| groq | **2** | 16 | **v2** | the only data from the new pipeline |

**Biggest finding:** ~85% of the corpus was produced by the *old* v1 prompt with the forced-choice fallback
that the April 2026 rewrite deliberately removed (it manufactures hallucinated 1–5 answers). The improved
v2 pipeline has barely been run. The UI therefore mostly shows the data the team already decided is unreliable.

## 5. Bugs found and fixed in this pass

| # | Problem | Fix |
|---|---|---|
| 1 | `server.py` looked for a non-existent sibling folder `literature-review-worker-model-main/` → `ModuleNotFoundError: config` on a fresh clone | Auto-detects `paper_annotation/` in the repo; `PIPELINE_ROOT` env still overrides |
| 2 | `main.py` imports `llm/mistral_llm.py`, which didn't exist → Mistral column could never run | Added `MistralLLM` (OpenAI-compatible, reuses Groq wrapper) |
| 3 | Chunker always emitted a redundant overlap-only tail chunk | Stop after the chunk that reaches end of text |
| 4 | Retriever lexical filter counted stopwords/question boilerplate and could return < top_k chunks | Stopword filter + backfill with vector hits to reach top_k |
| 5 | `_normalize_answer`: `True`→valid, `3.0`/`"3 - mostly skill"`/`"(3)"`→Unknown | Robust parsing |
| 6 | `_parse_llm_response` didn't handle ```json fences or prose around JSON | Fence/brace extraction |
| 7 | Non-numeric `confidence` or string `evidence` crashed the record builder | Defensive coercion |
| 8 | Image-only/empty PDF crashed with `need at least one array to stack` | Clear error: "No extractable text… run OCR" |
| 9 | Smoke test expected `llama` column; UI uses `groq` | Test updated |
| 10 | `huggingface_hub` imported but not in requirements; `.env.example` missing Groq/Mistral keys; `run.sh` documented old layout | Fixed |

## 6. Known issues / not working (not fixed)

1. **Human-consensus CSV missing** (`pht_ui/backend/data/human_consensus.csv`) → no human column, no accuracy numbers. This is the study's main validity metric and is entirely unmeasured.
2. `evaluation/metrics.py`: `inter_rater_reliability`, `accuracy_vs_human`, `confidence_calibration` are `NotImplementedError` stubs.
3. Corpus is v1/forced-choice (see §4). Re-running with v2 needs API keys + budget (~215 papers × 8 questions × 4 models ≈ 6,900 calls).
4. Mistral results dir mixes two models; `_load_existing_results` ignores `model_name`, so the UI can show a different model than labelled.
5. Server file-write race: two concurrent streams for the same paper both write the same JSONL (last writer wins).
6. Server comments/docstrings still say "9 questions"; there are 8.
7. `reference code/` and `Applications_of_LLM.pdf` (duplicated) add noise; `Config.model_name` default `gpt-4o-mini` is unused.
8. Retrieval is flat-L2 on un-normalised BGE vectors; no cross-encoder reranker; lexical stage is a bag-of-words overlap.
9. Scanned PDFs are unsupported (now fails clearly instead of obscurely).
10. CORS is `*` and there is no auth; fine for localhost only. Do not deploy as-is.
11. Not tested here: real LLM calls (no API keys available) and the browser UI interactions.

## 7. Tests added (`tests/`, run with `python -m pytest`) — 93 passing

- **Unit:** 30+ answer-normalisation variants, response-parsing variants, chunker invariants (6 size/overlap combos), cleaner, metrics, config.
- **Pipeline variations:** 12 synthetic PDFs (1 page, 5, 25, unicode, hyphenated, numbers only, repeated text, survey-not-game, mixed blank pages, empty, blank…) through real PDF parse → chunk → FAISS → retrieve → annotate → JSONL → reload; 10 LLM-response scenarios (fenced JSON, string answers, N/A, garbage, Unknown→retry→resolved, etc.).
- **API:** upload validation, full SSE stream for a newly uploaded paper, cache hit with zero LLM calls, resume of partial results, one provider failing without breaking the others, 404s.
- Embedder and LLMs are deterministic fakes, so the suite runs in ~15 s with no keys. Original `pht_ui/tests/smoke_test.py` still works against the real cached corpus.

## 8. Roadmap (recommended order)

1. **Get the human-consensus CSV** into `pht_ui/backend/data/` and implement `accuracy_vs_human`, Cohen's/Fleiss' kappa across models, and a calibration curve. Without this the study has no ground truth.
2. **Re-run all four models with the v2 prompt** (start with a 20-paper pilot, compare v1 vs v2 vs human), keeping v1 in a separate folder for the paper's ablation. Store `prompt_version` in the folder name or UI so versions never mix.
3. **Clean data:** move the 8 stray llama-3.1-8b records out of `mistral/`; re-run the 2 incomplete Gemini papers; make `_load_existing_results` filter by `model_name` + `prompt_version`.
4. **UI:** export CSV/JSON bundle; v1-vs-v2 toggle; disagreement filter; corpus-level distribution view per dimension; human column.
5. **Robustness:** per-paper lock for writes, OCR fallback for scanned PDFs, request/response logging with cost tracking, structured-output (schema) mode where providers support it.
6. **Retrieval quality:** normalise embeddings + cosine (or `IndexFlatIP`), BM25 instead of raw overlap, optional cross-encoder rerank; evaluate with `retrieval_hit_rate`.
7. **Research outputs:** the final report (`Final Report/Applications_of_LLM.pdf`) should be regenerated from the v2 + human-agreement results.
8. **Hygiene:** CI running `pytest`, pin dependency versions, drop `reference code/` into an archive branch.

## 9. Glossary

- **PHT** — Playful Health Technology. **Decision mode** — how a record was produced (`full_context`, `retrieval`, `expanded_retry`, `unresolved`, `not_applicable`, legacy `forced_choice`). **Consensus** — final human answer after Covidence reconciliation.


## 10. Update 2026-11-10: evaluation, training, deployment

**Added**

| Area | What | How to run |
|---|---|---|
| Evaluation | `paper_annotation/evaluation/analysis.py`, `report.py`: accuracy, quadratic kappa, chance baselines, MAE, ECE, Kruskal-Wallis, Dunn, rank-biserial, ensembles, selective prediction, calibration, inter-model agreement | `python -m evaluation.report --human consensus.csv` (from `paper_annotation/`) or `make report HUMAN=...` |
| Training | `training/stacker.py` (paper-grouped CV stacking meta-model, save/predict), `training/export_finetune.py` (train/val chat JSONL for LoRA / API fine-tuning) | `python -m training.stacker --human consensus.csv`; `python -m training.export_finetune --human consensus.csv` |
| Pipeline | iterative chunk expansion (`max_expansions`, default 2); full-context `Not Applicable` mode fixed | env `MAX_EXPANSIONS` |
| Server | `/api/health`, `CORS_ORIGINS` env, lock+merge persistence (concurrent writes no longer lose records) | |
| Deploy | `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `.github/workflows/ci.yml`, `requirements-ci.txt`, `Makefile` | `docker compose up --build` (Dockerfile not built here: Docker is not installed on this machine) |
| Tests | 130+ pytest tests; CI-equivalent clean venv without torch passes | `python -m pytest` |

**Result available now (no human labels needed):** models agree with each other at quadratic kappa 0.47-0.67 (exact match 29-55 %), i.e. they are consistent with one another even though the paper finds them nearly uncorrelated with the human coders. See `reports/`.

**Blocked on the user:** human consensus CSV, API keys, GPU/provider account for fine-tuning, and a decision on pushing this repo / creating a hosted deployment (nothing has been pushed or created in any cloud account).
