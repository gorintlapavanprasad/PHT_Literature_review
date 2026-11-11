"""Few-shot examples taken from the human-coded papers (issue #25).

The v2 prompt uses one fixed example per question. This builds examples from the consensus-coded papers instead,
with a hard leave-one-paper-out rule: the paper being annotated is NEVER used as its own example, otherwise the
test would leak the answer. Examples are chosen to cover different labels (so the model sees the scale, not just
the commonest answer) and the choice is deterministic.

Typical use (needs the human labels and the saved indexes):

    bank = FewShotBank.build(human_df, index_dir, results_dir)
    block = bank.block_for(paper_id="smith21", question_index=2, k=3)   # text to add to the prompt
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from ingest.chunker import TextChunk

MAX_EXCERPT_CHARS = 700


@dataclass(frozen=True)
class Example:
    paper_id: str
    q: int
    label: int
    excerpt: str


def _stable_rank(paper_id: str, q: int) -> int:
    """Deterministic pseudo-random order that does not depend on the run or on who else is in the bank."""
    return int(hashlib.sha1(f"{paper_id}|{q}".encode()).hexdigest(), 16)


def _excerpt_for(chunks: List[TextChunk], evidence_ids: Iterable[str]) -> str:
    by_id = {c.chunk_id: c for c in chunks}
    for cid in evidence_ids:
        if cid in by_id:
            return " ".join(by_id[cid].text.split())[:MAX_EXCERPT_CHARS]
    return " ".join(chunks[0].text.split())[:MAX_EXCERPT_CHARS] if chunks else ""


class FewShotBank:
    def __init__(self, examples: List[Example]) -> None:
        self.examples = examples

    @classmethod
    def build(cls, human_df, index_dir: Path, results_dir: Optional[Path] = None, providers=("openai", "gemini", "llama", "mistral")):
        """One example per (paper, question) that has a human label and a saved chunk index."""
        from training.export_finetune import _reference_evidence, load_chunks

        out: List[Example] = []
        cache: dict = {}
        for paper_id, q, label in human_df[["paper_id", "q", "human"]].itertuples(index=False, name=None):
            chunks = cache.setdefault(paper_id, load_chunks(Path(index_dir), paper_id))
            if not chunks:
                continue
            ev = _reference_evidence(Path(results_dir), paper_id, int(q), list(providers)) if results_dir else []
            excerpt = _excerpt_for(chunks, ev)
            if excerpt:
                out.append(Example(paper_id, int(q), int(label), excerpt))
        return cls(out)

    def select(self, paper_id: str, question_index: int, k: int = 3) -> List[Example]:
        """Up to k examples for this question, never from `paper_id`, covering as many distinct labels as possible."""
        pool = [e for e in self.examples if e.q == question_index and e.paper_id != paper_id]
        pool.sort(key=lambda e: _stable_rank(e.paper_id, e.q))
        chosen: List[Example] = []
        seen_labels = set()
        for e in pool:                       # first pass: one per label
            if e.label not in seen_labels and len(chosen) < k:
                chosen.append(e)
                seen_labels.add(e.label)
        for e in pool:                       # then fill up
            if len(chosen) >= k:
                break
            if e not in chosen:
                chosen.append(e)
        return sorted(chosen, key=lambda e: e.label)

    def block_for(self, paper_id: str, question_index: int, k: int = 3) -> str:
        ex = self.select(paper_id, question_index, k)
        if not ex:
            return ""
        lines = ["WORKED EXAMPLES (coded by the research team; the paper you are coding is NOT one of them)"]
        for i, e in enumerate(ex, 1):
            lines.append(f'Example {i}\nExcerpt: "{e.excerpt}"\nTeam answer: {e.label}')
        return "\n\n".join(lines)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps([e.__dict__ for e in self.examples]), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FewShotBank":
        return cls([Example(**d) for d in json.loads(Path(path).read_text(encoding="utf-8"))])
