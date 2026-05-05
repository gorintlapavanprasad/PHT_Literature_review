"""
Human consensus loader.

Reads the Covidence export CSV once at startup and exposes a simple
`get_human_answers(paper_id) -> {question_index: 1-5}` lookup.

Paper IDs are derived from the CSV's "Study ID" field (e.g. "Ahmetovic 2021"
-> "ahmetovic21"), matching the filename convention used by paper_annotation
(ahmetovic21.pdf, duval18.pdf, etc.).
"""
from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Optional

log = logging.getLogger("pht_ui.human")

# Column name in the CSV -> question index (0-based, matching PHT_FRAMEWORK_QUESTIONS)
# Trailing spaces in CSV headers are preserved exactly as they appear in the export.
_PHT_COLUMNS = [
    "Fun-First vs. Utility-First (Initial Design Intention)",
    "Play vs. Game",
    "Skill vs. Chance",
    "Social vs. Solo",
    "Sequential vs. Simultaneous",
    "Synchronous vs. Asynchronous ",          # note trailing space in CSV
    "Competitive vs. Collaborative",
    "Symmetrical vs. Asymmetrical",
]

# For each column, map starting phrase of the "Consensus" answer -> 1-5 rubric score.
# The CSV stores the full natural-language option; we match on a unique prefix.
_VALUE_MAPPINGS: dict[str, dict[str, int]] = {
    "Fun-First vs. Utility-First (Initial Design Intention)": {
        "Totally fun first": 1,
        "Mostly fun first": 2,
        "Both fun and utility first": 3,
        "Mostly utility first": 4,
        "Totally utility first": 5,
    },
    "Play vs. Game": {
        "Unstructured play": 1,
        "Semi-structured play": 2,
        "Flexible structure with rules": 3,
        "Flexible game": 4,
        "Game with rigid rules": 5,
    },
    "Skill vs. Chance": {
        "Entirely skill-based": 1,
        "Mostly skill-based": 2,
        "Equally skill and chance based": 3,
        "Mostly chance": 4,
        "Entirely chance": 5,
    },
    "Social vs. Solo": {
        "Entirely Solo": 1,
        "Mostly Solo": 2,
        "Mix of solo and social": 3,
        "Mostly social": 4,
        "Entirely social": 5,
    },
    "Sequential vs. Simultaneous": {
        "Entirely Turn-based": 1,
        "Follows a set of steps": 2,
        "Turns are taken, but some actions": 3,
        "Most actions can be taken at any time": 4,
        "Entirely Simultaneous": 5,
    },
    "Synchronous vs. Asynchronous ": {
        "Entirely Synchronous": 1,
        "Mostly Synchronous": 2,
        "Equal mix of synchronous": 3,
        "Mostly asycnronous": 4,        # typo preserved as it appears in CSV
        "Entirely asynchrounous": 5,    # typo preserved as it appears in CSV
    },
    "Competitive vs. Collaborative": {
        "Entirely Competitive": 1,
        "Mostly Competitive": 2,
        "Mix of competitive and collaborative": 3,
        "Mostly Collaborative": 4,
        "Entirely Collaborative": 5,
    },
    "Symmetrical vs. Asymmetrical": {
        "Entirely Symetrical": 1,
        "Mostly Symmetrical": 2,
        "Both Symetrical and Asymetrical": 3,
        "Mostly Asymetrical": 4,
        "Entirely Asymetrical": 5,
    },
}


def _study_id_to_paper_id(study_id: str) -> Optional[str]:
    """
    "Ahmetovic 2021" -> "ahmetovic21"
    "Bar-El 2018"    -> "bar-el18"
    """
    if not study_id:
        return None
    parts = study_id.strip().split()
    if len(parts) < 2:
        return None
    # Author name = everything but the trailing year
    year = parts[-1]
    name_parts = parts[:-1]
    name = " ".join(name_parts).lower().replace(" ", "")
    m = re.search(r"\d{4}", year)
    if not m:
        return None
    full_year = m.group()
    yy = full_year[-2:]
    return f"{name}{yy}"


def _text_to_score(col: str, text: str) -> Optional[int]:
    if not text:
        return None
    t = text.strip()
    if not t or "Not Applicable" in t or "Undisclosed" in t:
        return None
    for prefix, score in _VALUE_MAPPINGS.get(col, {}).items():
        if t.startswith(prefix):
            return score
    return None


# In-memory cache: paper_id -> {question_index: 1-5}
_cache: dict[str, dict[int, int]] = {}
_loaded = False


def load_human_answers(csv_path: Path) -> dict[str, dict[int, int]]:
    """Parse the Covidence export and return {paper_id: {q_index: score}}."""
    global _cache, _loaded

    if not csv_path.exists():
        log.warning("Human consensus CSV not found at %s", csv_path)
        _loaded = True
        return {}

    out: dict[str, dict[int, int]] = {}
    try:
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Reviewer Name", "").strip() != "Consensus":
                    continue
                pid = _study_id_to_paper_id(row.get("Study ID", ""))
                if not pid:
                    continue
                scores: dict[int, int] = {}
                for qi, col in enumerate(_PHT_COLUMNS):
                    score = _text_to_score(col, row.get(col, ""))
                    if score is not None:
                        scores[qi] = score
                if scores:
                    out[pid] = scores
    except Exception as exc:
        log.exception("Failed to parse human consensus CSV: %s", exc)
        _loaded = True
        return {}

    _cache = out
    _loaded = True
    log.info("Loaded human consensus for %d papers from %s", len(out), csv_path.name)
    return out


def get_human_answers(paper_id: str) -> dict[int, int]:
    """Return {question_index: 1-5} for a paper, or empty dict if none."""
    return _cache.get(paper_id, {})


def all_paper_ids() -> list[str]:
    """List all paper IDs that have human answers loaded."""
    return sorted(_cache.keys())
