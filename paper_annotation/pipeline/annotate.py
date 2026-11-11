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
import re
from typing import List

from llm.base import BaseLLM
from llm.prompts import build_prompt, build_forced_choice_prompt
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
    passes: int = 1  # number of LLM calls needed (the model's 'effort'): 1 = first pass sufficed


def _penalize_confidence(confidence: float, evidence: List[str]) -> float:
    """Halve confidence when no evidence chunk ids are cited."""
    if not evidence:
        return max(0.0, confidence * 0.5)
    return confidence


def _parse_llm_response(raw_response: str) -> dict:
    """Parse JSON from the LLM, tolerant of providers that wrap it."""
    text = (raw_response or "").strip()
    candidates = [text]
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        candidates.append(fence.group(1))
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return {
        "answer": "Unknown",
        "confidence": 0.0,
        "evidence": [],
        "reasoning": "Invalid JSON response from model.",
        "latency_seconds": 0.0,
    }


def _normalize_answer(answer) -> str | int:
    """Coerce the answer field into int 1-5, 'Unknown', or 'Not Applicable'."""
    if isinstance(answer, bool):  # bool is an int subclass; True must not become 1
        return "Unknown"
    if isinstance(answer, float) and answer.is_integer():
        answer = int(answer)
    if isinstance(answer, int) and 1 <= answer <= 5:
        return answer
    if isinstance(answer, str):
        s = answer.strip()
        # Strings like "3", "3.", "(3)", "3 - mostly skill", "3.0"
        m = re.match(r"^\(?\s*([1-5])(?:\.0)?\s*\)?(?:[\s.:\-–—)].*)?$", s, re.DOTALL)
        if m:
            return int(m.group(1))
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
    max_expansions: int = 1,
    force_choice: bool = False,
    fewshot=None,
) -> AnnotationRecord:
    """Annotate a single question for a paper.

    Flow:
      1. If paper has <=30 chunks, pass full paper. One LLM call.
      2. Otherwise, retrieve top-k chunks. One LLM call.
      3. If that returned Unknown, retry with 2x, 4x, ... top-k (up to max_expansions extra calls).
      4. Accept whatever came back — including Unknown / Not Applicable.
         No forced guessing.
    """
    extra = ""
    if fewshot is not None and question_index is not None:
        extra = fewshot(paper_id, question_index) or ""     # never contains this paper (leave-one-out)
    total_chunks = len(embedded.chunks)
    total_latency = 0.0
    passes = 0
    decision_mode = "unresolved"

    if total_chunks <= _FULL_CONTEXT_CHUNK_THRESHOLD:
        # Small paper: provide full context to avoid missing evidence.
        selected_chunks = embedded.chunks
        prompt = build_prompt(question, selected_chunks, question_index=question_index, extra_examples=extra)
        raw = llm.generate(prompt)
        passes += 1
        response = _parse_llm_response(raw)
        total_latency += float(response.get("latency_seconds", 0.0))
        full_answer = _normalize_answer(response.get("answer"))
        if _is_resolved(full_answer):
            decision_mode = "full_context"
        elif full_answer == "Not Applicable":
            decision_mode = "not_applicable"
    else:
        # Standard retrieval pass.
        retrieval = retriever.retrieve(
            query=question,
            query_embedding=query_embedding,
            top_k=top_k,
        )
        selected_chunks = retrieval.chunks
        prompt = build_prompt(question, selected_chunks, question_index=question_index, extra_examples=extra)
        raw = llm.generate(prompt)
        passes += 1
        response = _parse_llm_response(raw)
        total_latency += float(response.get("latency_seconds", 0.0))

        first_answer = _normalize_answer(response.get("answer"))
        if _is_resolved(first_answer):
            decision_mode = "retrieval"
        elif first_answer == "Not Applicable":
            decision_mode = "not_applicable"
        else:
            # First pass was Unknown: keep widening the window (k*2, k*4, ...), up to
            # `max_expansions` extra LLM calls, stopping at the first committed answer.
            current_k = top_k
            for _ in range(max_expansions):
                expanded_top_k = min(total_chunks, max(current_k + 1, current_k * _UNKNOWN_RETRY_MULTIPLIER))
                if expanded_top_k <= current_k:
                    break  # already using every chunk
                current_k = expanded_top_k
                retry_chunks = retriever.retrieve(
                    query=question, query_embedding=query_embedding, top_k=current_k
                ).chunks
                retry_raw = llm.generate(build_prompt(question, retry_chunks, question_index=question_index, extra_examples=extra))
                passes += 1
                retry_response = _parse_llm_response(retry_raw)
                total_latency += float(retry_response.get("latency_seconds", 0.0))
                retry_answer = _normalize_answer(retry_response.get("answer"))
                if _is_resolved(retry_answer):
                    selected_chunks, response, decision_mode = retry_chunks, retry_response, "expanded_retry"
                    break
                if retry_answer == "Not Applicable":
                    selected_chunks, response, decision_mode = retry_chunks, retry_response, "not_applicable"
                    break
                # still Unknown: keep the widest context seen so the record reflects what the model saw
                selected_chunks = retry_chunks
            else:
                pass  # exhausted expansions: decision_mode stays "unresolved" (honest Unknown)

    # Optional v1 behaviour (ablation only): still Unknown after every pass -> make the model pick a best fit.
    if force_choice and decision_mode == "unresolved":
        forced_raw = llm.generate(build_forced_choice_prompt(question, selected_chunks, question_index=question_index, extra_examples=extra))
        passes += 1
        forced_response = _parse_llm_response(forced_raw)
        total_latency += float(forced_response.get("latency_seconds", 0.0))
        if _is_resolved(_normalize_answer(forced_response.get("answer"))):
            response = forced_response
            decision_mode = "forced_choice"

    # Extract final fields.
    answer = _normalize_answer(response.get("answer"))
    try:
        confidence = float(response.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    raw_evidence = response.get("evidence", [])
    evidence = [str(e) for e in raw_evidence] if isinstance(raw_evidence, list) else []
    reasoning = response.get("reasoning", "")
    latency = total_latency

    # Only penalize confidence for *resolved* answers without evidence.
    # An honest Unknown doesn't need evidence.
    if _is_resolved(answer):
        confidence = _penalize_confidence(confidence, evidence)
        if decision_mode == "forced_choice":
            confidence = min(confidence, 0.6)   # a guess must not look as sure as a grounded answer
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
        passes=passes,
    )


if __name__ == "__main__":
    print("Annotation pipeline ready.")
