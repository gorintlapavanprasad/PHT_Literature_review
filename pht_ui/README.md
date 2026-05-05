# PHT ScholarLens — Coding Console

A local web UI for your paper-annotation pipeline. Drop a research PDF, watch
four LLMs code it against the nine PHT framework dimensions in real time, and
click any cell in the 9×4 heatmap to see the evidence, reasoning, and page
references that drove each answer.

This is a **thin wrapper around the existing `paper_annotation/` codebase** —
it does not fork or modify the pipeline. Any improvement you make to
chunking, retrieval, or LLM prompts carries over automatically.

## What you get

- **Drag-and-drop upload** — drops the PDF into `papers/` with the filename stem as `paper_id`, so the existing FAISS cache still hits on re-runs.
- **Browsable corpus** — filterable list of the ~215 papers you've already coded. Click any of them to load cached results instantly.
- **Live 9×4 heatmap** — rows are the PHT dimensions, columns are GPT-5.1, Gemini 2.5 Flash, Llama 3.3 70B, Mistral Small 3.1. Cells pulse while pending, fill in as server-sent events arrive, show answer + confidence.
- **Inter-model agreement strip** — for each dimension, surfaces which ones had full consensus, 3/4 majority, or a genuine split. The split rows are the ones worth reviewing by hand.
- **Evidence panel** — click any cell for full reasoning, cited chunks with page refs, all retrieved context, and raw JSON.
- **Activity log** — timestamped feed of every ingest event, model start, result, and error.

## Architecture

```
┌─────────────────────────────┐         ┌──────────────────────────────┐
│  frontend/index.html        │◀────────│  backend/server.py (FastAPI) │
│  vanilla JS · Tailwind CDN  │   SSE   │                              │
│  no build step              │─────────▶│  imports paper_annotation/   │
│                             │  JSON   │  main.py helpers directly    │
└─────────────────────────────┘         └──────────────────────────────┘
                                                       │
                                                       ▼
                                        ┌──────────────────────────────┐
                                        │ literature-review-worker-    │
                                        │ model-main/paper_annotation/ │
                                        │   ingest/   retrieval/       │
                                        │   pipeline/ llm/  storage/   │
                                        └──────────────────────────────┘
```

Four endpoints:

| Method | Path                       | What it does                                                         |
|--------|----------------------------|----------------------------------------------------------------------|
| GET    | `/api/meta`                | Boot payload: dimension labels, model aliases, existing paper IDs.  |
| GET    | `/api/papers`              | List of papers that have at least one cached result on disk.         |
| POST   | `/api/upload`              | Save a PDF into `papers/<stem>.pdf`.                                 |
| GET    | `/api/results/{paper_id}`  | All cached JSONL results for a paper, grouped by model × question.   |
| GET    | `/api/stream/{paper_id}`   | SSE. Runs ingest then fans out 4 models concurrently, streaming each (question × model) result as it lands. |

## Expected layout

```
<repo-root>/
├── literature-review-worker-model-main/
│   └── paper_annotation/
│       ├── .env                  ← OPENAI_API_KEY, GEMINI_API_KEY, HF_TOKEN
│       ├── main.py
│       ├── config.py
│       ├── questions.py
│       ├── data/
│       │   ├── indexes/          ← FAISS indexes (created on first ingest)
│       │   ├── results/          ← JSONL results by model
│       │   └── embeddings_cache.sqlite
│       └── ... (ingest/ retrieval/ pipeline/ llm/ storage/)
├── papers/                       ← your PDFs live here
└── pht_ui/                       ← this project
    ├── backend/
    │   ├── server.py
    │   └── requirements.txt
    ├── frontend/
    │   └── index.html
    ├── tests/
    │   └── smoke_test.py
    ├── run.sh
    └── README.md
```

If your repo doesn't match this layout, set `PIPELINE_ROOT` to point at your
`paper_annotation/` directory:

```bash
export PIPELINE_ROOT=/absolute/path/to/paper_annotation
```

## Setup

### 1. Install dependencies

The UI layer is thin but it imports the full pipeline, so you need both sets
of deps installed in the same Python environment.

```bash
# Pipeline deps (you probably already have these)
pip install -r literature-review-worker-model-main/paper_annotation/requirements.txt

# UI deps (FastAPI + uvicorn + python-multipart)
pip install -r pht_ui/backend/requirements.txt
```

### 2. Verify wiring with the headless smoke test

Before starting the server, sanity-check the integration:

```bash
cd pht_ui
python tests/smoke_test.py              # defaults to ahmetovic21
python tests/smoke_test.py duval18      # or any paper you've already coded
```

You should see ✓ marks for the meta, papers, and results endpoints plus a
replication of the frontend's inter-model agreement heuristic. This test
does not call any LLM APIs — it only exercises the JSONL parsing and the
endpoint wiring.

### 3. Run the server

```bash
./pht_ui/run.sh
# or, equivalently:
# cd pht_ui && uvicorn backend.server:app --host 127.0.0.1 --port 8765 --reload
```

Open <http://127.0.0.1:8765>. FastAPI serves the frontend from the same
origin, so no CORS config is needed.

## Using it

**Browse existing papers.** The left "Corpus" panel lists everything that has
at least one JSONL on disk. Click a paper → results load from disk instantly,
heatmap fills, no API calls fire.

**Upload a new paper.** Drop a PDF in the left panel. It gets saved to
`papers/<stem>.pdf`. Click **code this paper** — the pipeline ingests (text
extraction → cleaning → chunking → embedding → FAISS index), then all four
models run concurrently. As each (question × model) result lands, its heatmap
cell fills in. Expect 3–8 minutes total for a fresh paper; near-instant for
a fully cached one.

**Inspect an answer.** Click any heatmap cell. The slide-in panel shows the
full question text, the LLM's reasoning, the evidence chunks it cited (with
page and character offsets), all retrieved context, and the raw record JSON.
The `decision_mode` field tells you how the answer was produced —
`full_context` (small paper), `retrieval` (first pass worked),
`expanded_retry` (doubled top-k after Unknown), `forced_choice` (best-fit
fallback with capped confidence), or `api_failure`/`parse_failure`.

**Spot disagreements.** The bottom strip shows inter-model agreement per
dimension. Split rows (three or more different answers across the four
models) are the ones that deserve a closer look — those are the cases your
methods section should discuss.

## Design choices worth knowing

**Concurrency.** `config.py` sets `max_concurrency: 1` for rate-limit safety
*per model*. But OpenAI, Gemini, and HuggingFace are independent providers —
hitting them in parallel costs nothing. The server fans out the four models
simultaneously via `asyncio.run_in_executor`, which is what makes the wall
time feel like one model's duration instead of four.

**Cache-first streaming.** If a JSONL already exists on disk for a model,
those rows are re-emitted as SSE events immediately (marked `cached: true`),
and only the missing questions get computed. You can run the pipeline with
a new model added to `config.py` without recomputing the three you already
have.

**No pipeline modifications.** `backend/server.py` imports `main.py` and
pipeline modules directly. The only duplicated logic is a small re-
implementation of `annotate_paper` that yields one result at a time instead
of returning a batch — necessary for the streaming UX.

## Troubleshooting

**"No PDFs found in papers/"** — the pipeline looks in
`literature-review-worker-model-main/papers/`, not in `pht_ui/`. Upload
through the UI or copy files there directly.

**Model errors on specific providers.** Each model's `.env` key is checked
only when that model's LLM is constructed (lazily, on first use). So a
missing `HF_TOKEN` breaks Llama and Mistral but leaves OpenAI and Gemini
fully working — the heatmap will just show ERR for the HuggingFace columns.

**"gpt-5.1 not found" or similar.** `config.py` lists `gpt-5.1` as the
OpenAI model. If that name isn't live on your account, edit
`paper_annotation/config.py` `MODELS["openai"]` and restart. The UI picks
up the change on next boot.

**Large PDFs time out.** PyMuPDF extraction is fast; the bottleneck is
embedding ~200 chunks on first ingest (CPU-bound). Subsequent runs hit the
SQLite embedding cache and skip recomputation.

## Roadmap (reasonable next steps)

- Export current heatmap + evidence as a single CSV/JSON bundle for paper
  supplementary materials.
- Side-by-side diff view: pick two models and highlight every dimension
  where they disagreed, with both reasonings adjacent.
- Human-coder column: a 5th column where you can record your own rating
  and the UI computes accuracy vs each model (the
  `accuracy_vs_human` stub in `evaluation/metrics.py` is waiting for this).
- Corpus-level views: for a selected dimension, show the distribution of
  answers across all 215 papers as a spectrum heatmap.
