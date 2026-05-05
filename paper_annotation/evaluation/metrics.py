"""Evaluation metric stubs."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List


def inter_rater_reliability(predictions: Iterable[Dict[str, Any]]) -> float:
    """Placeholder for inter-rater reliability (e.g., Cohen's kappa)."""
    raise NotImplementedError("Inter-rater reliability not implemented yet.")


def accuracy_vs_human(predictions: Iterable[Dict[str, Any]], labels: Iterable[Dict[str, Any]]) -> float:
    """Placeholder for accuracy vs. human labels."""
    raise NotImplementedError("Accuracy vs human labels not implemented yet.")


def confidence_calibration(predictions: Iterable[Dict[str, Any]]) -> float:
    """Placeholder for confidence calibration metric."""
    raise NotImplementedError("Confidence calibration not implemented yet.")


def citation_coverage(predictions: Iterable[Dict[str, Any]]) -> float:
    """Compute fraction of answers that include at least one evidence citation."""
    items = list(predictions)
    if not items:
        return 0.0
    with_citations = sum(1 for item in items if item.get("evidence"))
    return with_citations / len(items)


def retrieval_hit_rate(predictions: Iterable[Dict[str, Any]]) -> float:
    """Compute fraction of evidence entries that appear in retrieved chunks."""
    items = list(predictions)
    if not items:
        return 0.0
    hits = 0
    total = 0
    for item in items:
        evidence = set(item.get("evidence", []))
        retrieved = {c.get("chunk_id") for c in item.get("retrieved_chunks", [])}
        total += max(len(evidence), 1)
        hits += sum(1 for e in evidence if e in retrieved)
    return hits / total


def answer_stability(predictions: Iterable[Dict[str, Any]]) -> float:
    """Estimate answer stability across repeated runs for the same paper/question/model."""
    grouped: Dict[tuple, List[str]] = defaultdict(list)
    for item in predictions:
        key = (item.get("paper_id"), item.get("question"), item.get("model_name"))
        grouped[key].append(str(item.get("answer")))
    if not grouped:
        return 0.0
    stability_scores = []
    for answers in grouped.values():
        if len(answers) == 1:
            stability_scores.append(1.0)
            continue
        mode_count = max(answers.count(a) for a in set(answers))
        stability_scores.append(mode_count / len(answers))
    return sum(stability_scores) / len(stability_scores)
