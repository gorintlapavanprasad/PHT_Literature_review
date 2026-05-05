"""Annotation pipeline logic.

April 2026 rewrite: removed the forced_choice fallback pass. Shelby's analysis
showed that forcing a 1-5 answer when evidence is weak correlates with
inaccuracy — it manufactures the hallucination pattern warned about in
CS 599 lec. 22. The pipeline now keeps "Unknown" and "Not Applicable" as
honest outputs and surfaces them in the UI.

Decision modes written to each record:
  full_context    - paper was small enough to fit in one call; no retrieval
  retrieval       - standard retrieval worked, first pass produced an answer
  expanded_retry  - first retrieval returned Unknown; wider context resolved it
  unresolved      - even expanded retrieval returned Unknown (honest)
  not_applicable  - model judged the paper does not describe a game/play system
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from typing import List

from llm.base import BaseLLM
from llm.prompts import build_prompt
import numpy as np

from retrieval.retriever import HybridRetriever
from ingest.embedder import EmbeddedIndex


# Use full-paper context for small papers to maximize recall.
_FULL_CONTEXT_CHUNK_THRESHOLD = 30
# If first pass returns Unknown, retry once with wider retrieval context.
_UNKNOWN_RETRY_MULTIPLIER = 2


@dataclass(frozen=True)
class AnnotationRecord:
    """A single annotation record."""

    paper_id: str
    question: str
    model_name: str
    prompt_version: str
    answer: str | int
    confidence: float
    evidence: List[str]
    evidence_metadata: List[dict]
    retrieved_chunks: List[dict]
    reasoning: str
    decision_mode: str
    latency: float
    timestamp: str


def _penalize_confidence(confidence: float, evidence: List[str]) -> float:
    """Halve confidence when no evidence chunk ids are cited."""
    if not evidence:
        return max(0.0, confidence * 0.5)
    return confidence


def _parse_llm_response(raw_response: str) -> dict:
    """Parse JSON from the LLM, tolerant of providers that wrap it."""
    try:
        parsed = json.loads(raw_response)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return {
        "answer": "Unknown",
        "confidence": 0.0,
        "evidence": [],
        "reasoning": "Invalid JSON response from model.",
        "latency_seconds": 0.0,
    }


def _normalize_answer(answer) -> str | int:
    """Coerce the answer field into int 1-5, 'Unknown', or 'Not Applicable'."""
    if isinstance(answer, int) and 1 <= answer <= 5:
        return answer
    if isinstance(answer, str):
        s = answer.strip()
        # Strings like "3" or "3." that should be ints
        if s.rstrip(".").isdigit():
            n = int(s.rstrip("."))
            if 1 <= n <= 5:
                return n
        low = s.lower()
        if "not applicable" in low or low in ("n/a", "na"):
            return "Not Applicable"
        if low == "unknown":
            return "Unknown"
    return "Unknown"


def _is_resolved(answer) -> bool:
    """True if the answer is a committed 1-5 (not Unknown, not N/A)."""
    return isinstance(answer, int) and 1 <= answer <= 5


def annotate_question(
    paper_id: str,
    embedded: EmbeddedIndex,
    retriever: HybridRetriever,
    question: str,
    query_embedding: np.ndarray,
    llm: BaseLLM,
    model_name: str,
    prompt_version: str,
    top_k: int,
    question_index: int | None = None,
) -> AnnotationRecord:
    """Annotate a single question for a paper.

    Flow:
      1. If paper has <=30 chunks, pass full paper. One LLM call.
      2. Otherwise, retrieve top-k chunks. One LLM call.
      3. If that returned Unknown, retry with 2x top-k. One more LLM call.
      4. Accept whatever came back — including Unknown / Not Applicable.
         No forced guessing.
    """
    total_chunks = len(embedded.chunks)
    total_latency = 0.0
    decision_mode = "unresolved"

    if total_chunks <= _FULL_CONTEXT_CHUNK_THRESHOLD:
        # Small paper: provide full context to avoid missing evidence.
        selected_chunks = embedded.chunks
        prompt = build_prompt(question, selected_chunks, question_index=question_index)
        raw = llm.generate(prompt)
        response = _parse_llm_response(raw)
        total_latency += float(response.get("latency_seconds", 0.0))
        if _is_resolved(_normalize_answer(response.get("answer"))):
            decision_mode = "full_context"
    else:
        # Standard retrieval pass.
        retrieval = retriever.retrieve(
            query=question,
            query_embedding=query_embedding,
            top_k=top_k,
        )
        selected_chunks = retrieval.chunks
        prompt = build_prompt(question, selected_chunks, question_index=question_index)
        raw = llm.generate(prompt)
        response = _parse_llm_response(raw)
        total_latency += float(response.get("latency_seconds", 0.0))

        first_answer = _normalize_answer(response.get("answer"))
        if _is_resolved(first_answer):
            decision_mode = "retrieval"
        elif first_answer == "Not Applicable":
            decision_mode = "not_applicable"
        else:
            # First pass was Unknown — try again with a wider window.
            expanded_top_k = min(total_chunks, max(top_k + 1, top_k * _UNKNOWN_RETRY_MULTIPLIER))
            if expanded_top_k > top_k:
                retry_retrieval = retriever.retrieve(
                    query=question,
                    query_embedding=query_embedding,
                    top_k=expanded_top_k,
                )
                retry_chunks = retry_retrieval.chunks
                retry_prompt = build_prompt(question, retry_chunks, question_index=question_index)
                retry_raw = llm.generate(retry_prompt)
                retry_response = _parse_llm_response(retry_raw)
                total_latency += float(retry_response.get("latency_seconds", 0.0))
                retry_answer = _normalize_answer(retry_response.get("answer"))

                # Prefer the retry only when it resolves Unknown.
                if _is_resolved(retry_answer):
                    selected_chunks = retry_chunks
                    response = retry_response
                    decision_mode = "expanded_retry"
                elif retry_answer == "Not Applicable":
                    selected_chunks = retry_chunks
                    response = retry_response
                    decision_mode = "not_applicable"
                # If retry also returned Unknown, keep decision_mode = "unresolved"

    # Extract final fields.
    answer = _normalize_answer(response.get("answer"))
    confidence = float(response.get("confidence", 0.0))
    evidence = list(response.get("evidence", []))
    reasoning = response.get("reasoning", "")
    latency = total_latency

    # Only penalize confidence for *resolved* answers without evidence.
    # An honest Unknown doesn't need evidence.
    if _is_resolved(answer):
        confidence = _penalize_confidence(confidence, evidence)
    else:
        # For Unknown / Not Applicable, confidence is about the model's
        # certainty that the answer can't be produced. Leave it alone.
        confidence = max(0.0, min(1.0, confidence))

    retrieved_chunks = [
        {
            "chunk_id": c.chunk_id,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "char_start": c.char_start,
            "char_end": c.char_end,
        }
        for c in selected_chunks
    ]
    evidence_metadata = [item for item in retrieved_chunks if item["chunk_id"] in evidence]

    return AnnotationRecord(
        paper_id=paper_id,
        question=question,
        model_name=model_name,
        prompt_version=prompt_version,
        answer=answer,
        confidence=confidence,
        evidence=evidence,
        evidence_metadata=evidence_metadata,
        retrieved_chunks=retrieved_chunks,
        reasoning=reasoning,
        decision_mode=decision_mode,
        latency=latency,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


if __name__ == "__main__":
    print("Annotation pipeline ready.")
