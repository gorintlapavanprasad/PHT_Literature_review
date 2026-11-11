"""Recompute every number the manuscript states about its Tables 4 and 5 (issue #9).

Table values are transcribed from the manuscript PDF (taccess_lit_review_llms.pdf, pages 9). Each "claim" is a
sentence in the text; the script recomputes it from the tables and flags mismatches. When the real results are
re-generated (`evaluation.report`), replace the transcription with the generated CSVs.

    python scripts/check_paper_numbers.py            # prints the check table, writes docs/paper_number_checks.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["GPT-5", "Llama3", "Gemini Pro", "Mistral"]

# Table 4: quadratic-weighted Cohen's kappa vs human consensus, F1..F8, and the printed Std. Dev. column
KAPPA = {
    "GPT-5": [0.144, 0.070, 0.545, 0.538, 0.130, 0.211, 0.230, 0.106],
    "Llama3": [0.116, -0.013, -0.075, 0.064, -0.049, -0.003, -0.077, -0.035],
    "Gemini Pro": [0.203, 0.012, -0.016, 0.080, 0.015, 0.101, -0.050, -0.006],
    "Mistral": [0.112, -0.005, -0.020, 0.051, 0.000, 0.033, -0.105, -0.001],
}
KAPPA_STD_PRINTED = {"GPT-5": 0.177, "Llama3": 0.064, "Gemini Pro": 0.076, "Mistral": 0.058}

# Table 5: accuracy vs human consensus
ACC = {
    "GPT-5": [0.234, 0.612, 0.561, 0.556, 0.290, 0.463, 0.397, 0.481],
    "Llama3": [0.210, 0.065, 0.154, 0.126, 0.117, 0.154, 0.117, 0.061],
    "Gemini Pro": [0.313, 0.065, 0.154, 0.159, 0.164, 0.154, 0.150, 0.079],
    "Mistral": [0.248, 0.056, 0.168, 0.173, 0.150, 0.168, 0.154, 0.079],
}
ACC_STD_PRINTED = {"GPT-5": 0.125, "Llama3": 0.046, "Gemini Pro": 0.070, "Mistral": 0.056}

TOL = 0.0015   # tables are rounded to 3 decimals


def _row_mean(t, m):
    return float(np.mean(t[m]))


def _col_mean(t, f):
    return float(np.mean([t[m][f] for m in MODELS]))


def _argext(t, fn):
    vals = [(t[m][f], m, f + 1) for m in MODELS for f in range(8)]
    return fn(vals, key=lambda x: x[0])


def checks() -> list[dict]:
    """[{claim, stated, computed, ok}] - one entry per checkable statement in the manuscript."""
    out = []

    def add(claim, stated, computed, tol=TOL):
        ok = abs(stated - computed) <= tol if isinstance(stated, (int, float)) else stated == computed
        out.append({"claim": claim, "stated": stated, "computed": computed, "ok": bool(ok)})

    # --- Section 4.1: kappa means per model
    for m, stated in zip(MODELS, [0.1795, -0.009, 0.0424, 0.0081]):
        add(f"mean kappa, {m}", stated, round(_row_mean(KAPPA, m), 4), tol=0.001)
    best = _argext(KAPPA, max)
    add("highest kappa value (GPT-5, F3 = 0.545)", 0.545, best[0])
    add("  ...belongs to GPT-5 / F3", "GPT-5 F3", f"{best[1]} F{best[2]}")
    low = _argext(KAPPA, min)
    add("lowest kappa value (text: Llama3, -0.077, F7)", "Llama3 F7", f"{low[1]} F{low[2]}")
    add("  ...its value", -0.077, low[0])
    add("average kappa for F4 (best framework)", 0.183, round(_col_mean(KAPPA, 3), 3))
    add("average kappa for F2 (worst framework)", 0.016, round(_col_mean(KAPPA, 1), 3))
    best_f = max(range(8), key=lambda f: _col_mean(KAPPA, f))
    worst_f = min(range(8), key=lambda f: _col_mean(KAPPA, f))
    add("best framework by mean kappa is F4", "F4", f"F{best_f + 1}")
    add("worst framework by mean kappa is F2", "F2", f"F{worst_f + 1}")
    for m in MODELS:
        add(f"printed Std. Dev. of kappa, {m} (matches population sd, ddof=0)", KAPPA_STD_PRINTED[m], round(float(np.std(KAPPA[m])), 3))

    # --- Section 4.1: accuracy means per model
    for m, stated in zip(MODELS, [0.449, 0.126, 0.136, 0.150]):
        add(f"mean accuracy, {m}", stated, round(_row_mean(ACC, m), 3))
    add("average accuracy for F1 (best framework)", 0.251, round(_col_mean(ACC, 0), 3))
    add("average accuracy for F5 (worst framework)", 0.180, round(_col_mean(ACC, 4), 3))
    add("GPT-5 highest accuracy, F2 (text: 61%)", 0.61, ACC["GPT-5"][1], tol=0.005)
    add("Mistral lowest accuracy, F2 (text: 5.6%)", 0.056, ACC["Mistral"][1])
    for m in MODELS:
        add(f"printed Std. Dev. of accuracy, {m} (matches population sd, ddof=0)", ACC_STD_PRINTED[m], round(float(np.std(ACC[m])), 3))

    return out


def chance_notes() -> list[str]:
    """Not a stated claim, but a reader will ask: is the accuracy above the 0.20 random-guess level for 5 options?"""
    return [f"{m}: mean accuracy {_row_mean(ACC, m):.3f} is {'above' if _row_mean(ACC, m) > 0.2 else 'BELOW'} the 0.20 chance level"
            for m in MODELS]


def to_markdown(rows: list[dict]) -> str:
    lines = ["# Paper number checks", "",
             "Every statement the manuscript makes about Tables 4-5, recomputed from the tables themselves.",
             "Mismatches are marked **NO**.", "",
             "| Claim | Stated | Computed | Match |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['claim']} | {r['stated']} | {r['computed']} | {'yes' if r['ok'] else '**NO**'} |")
    bad = [r for r in rows if not r["ok"]]
    lines += ["", f"{len(rows) - len(bad)} of {len(rows)} checks match; **{len(bad)} do not.**", "",
              "The printed Std. Dev. columns use the population formula (divide by 8), not the sample formula; say which in the paper.", "",
              "Chance level (5 options = 0.20 for random guessing), for context:", ""]
    lines += [f"- {n}" for n in chance_notes()]
    return "\n".join(lines) + "\n"


def main() -> int:
    rows = checks()
    md = to_markdown(rows)
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "paper_number_checks.md").write_text(md, encoding="utf-8")
    for r in rows:
        if not r["ok"]:
            print(f"MISMATCH: {r['claim']}: stated {r['stated']} vs computed {r['computed']}")
    print(f"\n{sum(r['ok'] for r in rows)}/{len(rows)} checks match -> docs/paper_number_checks.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
