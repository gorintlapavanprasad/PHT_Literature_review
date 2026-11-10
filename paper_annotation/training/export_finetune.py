"""Export a supervised fine-tuning dataset from human consensus labels.

For every human-labelled (paper, question) the example is:
    user      = the exact production prompt (llm.prompts.build_prompt) over that paper's chunks
    assistant = strict JSON  {"answer": <human 1-5>, "confidence": ..., "evidence": [...], "reasoning": ...}

The split is by PAPER (hash-based, deterministic) so no paper leaks between train and validation.
Output is OpenAI/HF "messages" JSONL, usable for OpenAI fine-tuning or LoRA with TRL/Unsloth/Axolotl:

    python -m training.export_finetune --human consensus.csv --out ../data/finetune

Evidence chunks default to those the best-agreeing LLM retrieved (so the model learns to ground in
retrievable text); if none are available the top chunks by position are used.
Training itself needs a GPU or a provider account and is intentionally not run here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
for p in (str(HERE), str(HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from ingest.chunker import TextChunk  # noqa: E402
from llm.prompts import build_prompt  # noqa: E402
from questions import PHT_FRAMEWORK_QUESTIONS  # noqa: E402


def split_of(paper_id: str, val_fraction: float = 0.2) -> str:
    h = int(hashlib.sha1(paper_id.encode()).hexdigest(), 16) % 1000 / 1000.0
    return "val" if h < val_fraction else "train"


def load_chunks(index_dir: Path, paper_id: str) -> list[TextChunk]:
    path = Path(index_dir) / f"{paper_id}.chunks.json"
    if not path.exists():
        return []
    return [TextChunk(**c) for c in json.loads(path.read_text(encoding="utf-8"))]


def _reference_evidence(results_dir: Path, paper_id: str, q: int, providers: list[str]) -> list[str]:
    for prov in providers:
        f = Path(results_dir) / prov / f"{paper_id}.jsonl"
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("question") == PHT_FRAMEWORK_QUESTIONS[q] and r.get("evidence"):
                return list(r["evidence"])
    return []


def build_examples(human_rows, index_dir: Path, results_dir: Path | None = None,
                   providers=("openai", "gemini", "llama", "mistral"), max_chunks: int = 15):
    """Yield (split, example_dict). `human_rows` is an iterable of (paper_id, q, human)."""
    cache: dict[str, list[TextChunk]] = {}
    for paper_id, q, human in human_rows:
        chunks = cache.setdefault(paper_id, load_chunks(index_dir, paper_id))
        if not chunks:
            continue
        evidence = _reference_evidence(results_dir, paper_id, q, list(providers)) if results_dir else []
        by_id = {c.chunk_id: c for c in chunks}
        selected = [by_id[e] for e in evidence if e in by_id]
        for c in chunks:                     # pad with the remaining chunks, in document order
            if len(selected) >= max_chunks:
                break
            if c not in selected:
                selected.append(c)
        selected = selected[:max_chunks]
        target = {
            "answer": int(human),
            "confidence": 0.9,
            "evidence": [e for e in evidence if e in by_id][:3],
            "reasoning": "Matches the human consensus coding for this framework dimension.",
        }
        yield split_of(paper_id), {
            "messages": [
                {"role": "user", "content": build_prompt(PHT_FRAMEWORK_QUESTIONS[q], selected, question_index=q)},
                {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
            ],
            "meta": {"paper_id": paper_id, "q": int(q)},
        }


def export(human_df, index_dir: Path, out_dir: Path, results_dir: Path | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(human_df[["paper_id", "q", "human"]].itertuples(index=False, name=None))
    counts = {"train": 0, "val": 0}
    with (out_dir / "train.jsonl").open("w", encoding="utf-8") as tr, \
            (out_dir / "val.jsonl").open("w", encoding="utf-8") as va:
        for split, ex in build_examples(rows, index_dir, results_dir):
            (tr if split == "train" else va).write(json.dumps(ex, ensure_ascii=False) + "\n")
            counts[split] += 1
    counts["skipped_no_chunks"] = len(rows) - counts["train"] - counts["val"]
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human", type=Path, required=True)
    ap.add_argument("--index-dir", type=Path, default=HERE / "data" / "indexes")
    ap.add_argument("--results", type=Path, default=HERE / "data" / "results")
    ap.add_argument("--out", type=Path, default=HERE.parent / "data" / "finetune")
    a = ap.parse_args()
    from evaluation.report import load_human

    print(export(load_human(a.human), a.index_dir, a.out, a.results))


if __name__ == "__main__":
    main()
