"""
PHT ScholarLens Coding Console — FastAPI backend.

Thin web layer over the existing paper_annotation pipeline. Does not modify any
pipeline code. Exposes three endpoints:

    POST /api/upload              -> save a PDF to papers/ directory
    GET  /api/results/{paper_id}  -> return already-computed results (if any)
    GET  /api/stream/{paper_id}   -> SSE: ingest + annotate with all 4 models,
                                     stream each result as it lands

The stream fires 36 events per paper (9 questions * 4 models) plus lifecycle
events (ingest_started, ingest_done, model_started, done, error).
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Import the existing pipeline without touching it.
# ---------------------------------------------------------------------------
PIPELINE_ROOT = Path(__file__).resolve().parents[1].parent / "literature-review-worker-model-main" / "paper_annotation"
if not PIPELINE_ROOT.exists():
    # Allow overriding via env for local dev or Docker paths.
    import os
    override = os.getenv("PIPELINE_ROOT")
    if override:
        PIPELINE_ROOT = Path(override)
sys.path.insert(0, str(PIPELINE_ROOT))

load_dotenv(PIPELINE_ROOT / ".env")

from config import load_config, all_models, model_output_dir, model_provider  # noqa: E402
from ingest.embedding_service import EmbeddingService, EmbeddingConfig  # noqa: E402
from questions import PHT_FRAMEWORK_QUESTIONS  # noqa: E402

# Re-use main.py helpers so we inherit exactly one source of truth for pipeline
# behavior (ingest, annotate, LLM construction).
import main as pipeline_main  # noqa: E402


# ---------------------------------------------------------------------------
# Short human-readable labels for the 9 PHT framework dimensions.
# These are UI metadata only — the full question text stays in questions.py.
# ---------------------------------------------------------------------------
PHT_DIMENSIONS = [
    {"id": 1, "short": "Fun vs Utility", "scale": "1=fun-first ... 5=utility-first"},
    {"id": 2, "short": "Play Structure", "scale": "1=unstructured ... 5=rigid rules"},
    {"id": 3, "short": "Skill vs Chance", "scale": "1=skill ... 5=chance"},
    {"id": 4, "short": "Solo vs Social", "scale": "1=solo ... 5=social"},
    {"id": 5, "short": "Turn Structure", "scale": "1=turn-based ... 5=simultaneous"},
    {"id": 6, "short": "Sync vs Async", "scale": "1=synchronous ... 5=asynchronous"},
    {"id": 7, "short": "Compete vs Collab", "scale": "1=competitive ... 5=collaborative"},
    {"id": 8, "short": "Symmetry", "scale": "1=symmetrical ... 5=asymmetrical"},
    {"id": 9, "short": "Dimension 9", "scale": ""},  # placeholder; see below
]

# questions.py ships with 8 questions. If a 9th gets added later, we pad
# PHT_DIMENSIONS automatically so the UI never desyncs from questions.py.
PHT_DIMENSIONS = PHT_DIMENSIONS[: len(PHT_FRAMEWORK_QUESTIONS)]

# Model aliases the UI cares about (stable column order in the heatmap).
# Each alias = one provider. Column header labels below are what users see.
UI_MODEL_ORDER = ["openai", "gemini", "groq", "mistral"]
UI_MODEL_LABELS = {
    "openai":  "GPT-5.1",
    "gemini":  "Gemini 2.5 Flash",
    "groq":    "Llama 3.3 70B (Groq)",
    "mistral": "Mistral Small",
}


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pht_ui")

app = FastAPI(title="PHT Literature Review")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local-only tool, single-user
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load human consensus answers at import time. If the CSV is missing, the app
# still works — the frontend just doesn't get a Human column.
from backend import human_consensus  # noqa: E402
_HUMAN_CSV_PATH = Path(__file__).resolve().parent / "data" / "human_consensus.csv"
human_consensus.load_human_answers(_HUMAN_CSV_PATH)

# Lazy singletons — embedding model and LLM clients are expensive to construct.
_config = None
_embedding_service: EmbeddingService | None = None
_llm_cache: dict[str, object] = {}


def _get_config():
    global _config
    if _config is None:
        _config = load_config()
    return _config


def _get_embedding_service() -> EmbeddingService:
    """Embedding model is heavy (~1GB). Construct once, reuse forever."""
    global _embedding_service
    if _embedding_service is None:
        cfg = _get_config()
        log.info("Loading embedding model %s ...", cfg.embedding_model_name)
        _embedding_service = EmbeddingService(
            EmbeddingConfig(
                model_name=cfg.embedding_model_name,
                cache_path=cfg.embedding_cache_path,
            )
        )
        log.info("Embedding model loaded.")
    return _embedding_service


def _get_llm(model_name: str):
    """Cache constructed LLM clients so we don't re-auth on every stream."""
    if model_name not in _llm_cache:
        _llm_cache[model_name] = pipeline_main._build_llm(model_name, _get_config())
    return _llm_cache[model_name]


def _results_path(paper_id: str, model_alias: str) -> Path:
    """data/results/<alias>/<paper_id>.jsonl"""
    return _get_config().results_dir / model_alias / f"{paper_id}.jsonl"


def _load_existing_results(paper_id: str) -> dict:
    """
    Read any JSONL files that already exist for this paper. Shape:
        { model_alias: { question_index: record_dict } }
    """
    out: dict[str, dict[int, dict]] = {}
    for alias in UI_MODEL_ORDER:
        path = _results_path(paper_id, alias)
        if not path.exists():
            continue
        records_by_q: dict[int, dict] = {}
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Match each record to its question index by string equality.
                q_text = rec.get("question", "")
                for idx, canonical in enumerate(PHT_FRAMEWORK_QUESTIONS):
                    if q_text == canonical:
                        records_by_q[idx] = rec
                        break
        if records_by_q:
            out[alias] = records_by_q
    return out


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
@app.get("/api/meta")
async def get_meta():
    """Frontend boot metadata: dimensions, models, paper count."""
    cfg = _get_config()
    existing = sorted({p.stem for p in cfg.pdf_dir.glob("*.pdf")})
    return {
        "dimensions": PHT_DIMENSIONS,
        "models": [{"alias": a, "label": UI_MODEL_LABELS[a]} for a in UI_MODEL_ORDER],
        "existing_papers": existing,
    }


@app.get("/api/papers")
async def list_papers():
    """List papers that have at least one cached result (fast browse mode)."""
    cfg = _get_config()
    seen: set[str] = set()
    for alias in UI_MODEL_ORDER:
        d = cfg.results_dir / alias
        if not d.exists():
            continue
        for f in d.glob("*.jsonl"):
            seen.add(f.stem)
    return {"papers": sorted(seen)}


@app.post("/api/upload")
async def upload_paper(file: UploadFile = File(...)):
    """Save a PDF into papers/ so the ingest stage can pick it up by stem."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only .pdf files are accepted.")
    cfg = _get_config()
    cfg.pdf_dir.mkdir(parents=True, exist_ok=True)
    # paper_id = filename stem; keep deterministic so reruns hit the cache.
    paper_id = Path(file.filename).stem
    dest = cfg.pdf_dir / f"{paper_id}.pdf"
    content = await file.read()
    dest.write_bytes(content)
    log.info("Uploaded %s (%d bytes)", dest.name, len(content))
    return {"paper_id": paper_id, "size_bytes": len(content)}


@app.get("/api/results/{paper_id}")
async def get_results(paper_id: str):
    """Return all cached results for a paper (empty dict if none)."""
    cfg = _get_config()
    pdf_path = cfg.pdf_dir / f"{paper_id}.pdf"
    return JSONResponse({
        "paper_id": paper_id,
        "pdf_exists": pdf_path.exists(),
        "results": _load_existing_results(paper_id),
        "human_answers": human_consensus.get_human_answers(paper_id),
    })


@app.get("/api/stream/{paper_id}")
async def stream_annotation(paper_id: str):
    """
    Run the full pipeline and stream each (question, model) result as an SSE
    event the moment it lands. Skips work that's already cached on disk.
    """
    cfg = _get_config()
    pdf_path = cfg.pdf_dir / f"{paper_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(404, f"No PDF at {pdf_path}")

    async def event_gen() -> AsyncGenerator[bytes, None]:
        def pack(event: str, payload: dict) -> bytes:
            return f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode("utf-8")

        loop = asyncio.get_running_loop()
        embed_svc = _get_embedding_service()

        # -------- Stage 1: ingest (build FAISS index if missing) ------------
        yield pack("ingest_started", {"paper_id": paper_id})
        t0 = time.perf_counter()
        try:
            await loop.run_in_executor(
                None,
                lambda: pipeline_main.ingest_paper(pdf_path, cfg, embed_svc, force=False),
            )
        except Exception as exc:
            log.exception("Ingest failed for %s", paper_id)
            yield pack("error", {"stage": "ingest", "message": str(exc)})
            return
        yield pack("ingest_done", {
            "paper_id": paper_id,
            "seconds": round(time.perf_counter() - t0, 2),
        })

        # -------- Stage 2: embed the 9 questions once ----------------------
        question_embeddings = await loop.run_in_executor(
            None, lambda: embed_svc.embed_queries(PHT_FRAMEWORK_QUESTIONS)
        )

        # -------- Stage 3: fan out to the 4 models in parallel -------------
        from config import MODELS  # local import to avoid clutter at top
        # One model per UI column. Each alias maps directly to one provider.
        models_by_alias = {
            "openai":  MODELS["openai"][0] if MODELS["openai"] else None,
            "gemini":  MODELS["gemini"][0] if MODELS["gemini"] else None,
            "groq":    MODELS.get("groq", [None])[0],
            "mistral": MODELS.get("mistral", [None])[0],
        }
        # Drop any aliases whose model couldn't be resolved (missing from config).
        models_by_alias = {k: v for k, v in models_by_alias.items() if v}

        # Queue that every per-model task pushes results into; the generator
        # drains it and forwards events downstream.
        queue: asyncio.Queue = asyncio.Queue()

        async def run_one_model(alias: str, full_name: str):
            """Process all 9 questions for one model, pushing each result."""
            out_path = _results_path(paper_id, alias)
            existing: dict[int, dict] = {}
            if out_path.exists():
                # Re-emit cached rows instantly, then only compute what's missing.
                try:
                    with out_path.open("r", encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            rec = json.loads(line)
                            q_text = rec.get("question", "")
                            for idx, canonical in enumerate(PHT_FRAMEWORK_QUESTIONS):
                                if q_text == canonical:
                                    existing[idx] = rec
                                    break
                except Exception:
                    existing = {}

            await queue.put(("model_started", {
                "paper_id": paper_id,
                "model": alias,
                "cached_count": len(existing),
            }))

            # Cached rows first.
            for idx, rec in existing.items():
                await queue.put(("result", {
                    "paper_id": paper_id,
                    "model": alias,
                    "question_index": idx,
                    "cached": True,
                    "record": rec,
                }))

            if len(existing) == len(PHT_FRAMEWORK_QUESTIONS):
                await queue.put(("model_done", {
                    "paper_id": paper_id, "model": alias, "computed": 0,
                }))
                return

            # Build LLM and run remaining questions one by one, streaming results.
            try:
                llm = await loop.run_in_executor(None, lambda: _get_llm(full_name))
            except Exception as exc:
                await queue.put(("model_error", {
                    "paper_id": paper_id, "model": alias, "message": str(exc),
                }))
                return

            # We duplicate a small part of main.annotate_paper so we can emit
            # one SSE event per question instead of waiting for all 9.
            from ingest.embedder import load_index
            from retrieval.retriever import HybridRetriever
            from pipeline.annotate import annotate_question

            index_path = cfg.index_dir / f"{paper_id}.faiss"
            chunks_path = cfg.index_dir / f"{paper_id}.chunks.json"
            embedded = await loop.run_in_executor(
                None, lambda: load_index(index_path=index_path, chunks_path=chunks_path)
            )
            retriever = HybridRetriever(embedded, embed_svc, lexical_top_k=cfg.lexical_top_k)

            new_records = []
            for idx, question in enumerate(PHT_FRAMEWORK_QUESTIONS):
                if idx in existing:
                    continue
                q_emb = question_embeddings[idx]
                try:
                    record = await loop.run_in_executor(
                        None,
                        lambda q=question, e=q_emb, qi=idx: annotate_question(
                            paper_id=paper_id,
                            embedded=embedded,
                            retriever=retriever,
                            question=q,
                            query_embedding=e,
                            llm=llm,
                            model_name=full_name,
                            prompt_version=cfg.prompt_version,
                            top_k=cfg.top_k,
                            question_index=qi,
                        ),
                    )
                except Exception as exc:
                    await queue.put(("result_error", {
                        "paper_id": paper_id, "model": alias,
                        "question_index": idx, "message": str(exc),
                    }))
                    continue
                new_records.append(record)
                await queue.put(("result", {
                    "paper_id": paper_id,
                    "model": alias,
                    "question_index": idx,
                    "cached": False,
                    "record": record.__dict__,
                }))

            # Persist the new records alongside any prior ones.
            if new_records or existing:
                try:
                    from pipeline.annotate import AnnotationRecord
                    merged: list[AnnotationRecord] = []
                    for idx in range(len(PHT_FRAMEWORK_QUESTIONS)):
                        if idx in existing:
                            # Rehydrate from dict — we only need dict-like for write.
                            merged.append(AnnotationRecord(**{
                                k: v for k, v in existing[idx].items()
                                if k in AnnotationRecord.__dataclass_fields__
                            }))
                        else:
                            match = next((r for r in new_records
                                          if PHT_FRAMEWORK_QUESTIONS.index(r.question) == idx), None)
                            if match:
                                merged.append(match)
                    from storage.writer import write_jsonl
                    await loop.run_in_executor(None, lambda: write_jsonl(merged, out_path))
                except Exception as exc:
                    log.warning("Persist failed for %s/%s: %s", alias, paper_id, exc)

            await queue.put(("model_done", {
                "paper_id": paper_id, "model": alias, "computed": len(new_records),
            }))

        # Launch all four model workers concurrently.
        tasks = [
            asyncio.create_task(run_one_model(alias, models_by_alias[alias]))
            for alias in UI_MODEL_ORDER
        ]

        # Drain the queue while any task is still alive.
        async def all_done():
            await asyncio.gather(*tasks, return_exceptions=True)
            await queue.put(("__sentinel__", {}))

        drain_task = asyncio.create_task(all_done())

        while True:
            event, payload = await queue.get()
            if event == "__sentinel__":
                break
            yield pack(event, payload)

        await drain_task
        yield pack("done", {"paper_id": paper_id})

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Static frontend — serve the SPA from the same origin.
# ---------------------------------------------------------------------------
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)
