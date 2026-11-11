"""Compare two annotation runs on the same papers (issue #24: v1 forced choice vs v2 honest Unknown).

Typical uses
  * v1 (forced-choice guesses) vs v2 (the model may say Unknown) for the same model;
  * the same prompt with two different models or two versions of one model.

Without human labels it reports how often the runs answer, abstain and agree. With human labels it answers the
question the ablation exists for: when v2 declined to answer, were v1's guesses on those cells actually wrong?

    python -m evaluation.compare --a data/results/openai --b ../pilot/results/openai_v2 [--human consensus.csv]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
for p in (str(HERE), str(HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluation import analysis as A  # noqa: E402
from questions import PHT_FRAMEWORK_QUESTIONS  # noqa: E402


def load_run(folder: Path) -> pd.DataFrame:
    """One results folder (data/results/<model>) as a tidy frame."""
    folder = Path(folder)
    df = A.load_results(folder.parent, PHT_FRAMEWORK_QUESTIONS, providers=[folder.name])
    if df.empty:
        raise SystemExit(f"No results in {folder}")
    return df


def _metrics(g: pd.DataFrame) -> dict:
    d = g.dropna(subset=["answer_num"])
    out = {"n": len(g), "answered": len(d), "answer_rate": len(d) / max(len(g), 1)}
    if "human" in g and len(d):
        out.update(accuracy=A.accuracy(d["human"], d["answer_num"]), qwk=A.qwk(d["human"], d["answer_num"]),
                   within_1=float(((d["answer_num"] - d["human"]).abs() <= 1).mean()))
    return out


def compare_runs(a: pd.DataFrame, b: pd.DataFrame, human: pd.DataFrame | None = None) -> dict:
    """Compare run A with run B on the (paper, question) cells both contain."""
    key = ["paper_id", "q"]
    m = a.merge(b, on=key, suffixes=("_a", "_b"))
    if m.empty:
        raise ValueError("The two runs share no (paper, question) cells")
    if human is not None:
        m = m.merge(human, on=key, how="left")
    both = m.dropna(subset=["answer_num_a", "answer_num_b"])
    res: dict = {
        "shared_cells": len(m),
        "answer_rate_a": float(m["answer_num_a"].notna().mean()),
        "answer_rate_b": float(m["answer_num_b"].notna().mean()),
        "exact_agreement_where_both_answered": float((both["answer_num_a"] == both["answer_num_b"]).mean()) if len(both) else float("nan"),
        "qwk_between_runs": A.qwk(both["answer_num_a"], both["answer_num_b"]) if len(both) > 1 else float("nan"),
        "mode_counts_a": m["decision_mode_a"].value_counts().to_dict(),
        "mode_counts_b": m["decision_mode_b"].value_counts().to_dict(),
    }
    # The cells where B abstained but A answered: those are the guesses B avoided.
    avoided = m[m["answer_num_b"].isna() & m["answer_num_a"].notna()]
    res["cells_b_abstained_a_answered"] = len(avoided)
    if human is not None and "human" in m:
        ma, mb = m.assign(answer_num=m["answer_num_a"]), m.assign(answer_num=m["answer_num_b"])
        res["human_a"], res["human_b"] = _metrics(ma.dropna(subset=["human"])), _metrics(mb.dropna(subset=["human"]))
        lab = avoided.dropna(subset=["human"])
        if len(lab):
            res["avoided_guesses_accuracy"] = float((lab["answer_num_a"] == lab["human"]).mean())
            res["avoided_guesses_within_1"] = float(((lab["answer_num_a"] - lab["human"]).abs() <= 1).mean())
            res["avoided_guesses_n"] = len(lab)
        # fair comparison: only cells where BOTH answered
        lb = both.dropna(subset=["human"]) if "human" in both else both.iloc[0:0]
        if len(lb):
            res["both_answered_accuracy_a"] = A.accuracy(lb["human"], lb["answer_num_a"])
            res["both_answered_accuracy_b"] = A.accuracy(lb["human"], lb["answer_num_b"])
    return res


def format_report(r: dict, name_a: str = "A", name_b: str = "B") -> str:
    L = [f"# Run comparison: {name_a} vs {name_b}", "", f"Shared (paper, question) cells: {r['shared_cells']}", "",
         f"- Answer rate: {name_a} {r['answer_rate_a']:.1%}, {name_b} {r['answer_rate_b']:.1%}",
         f"- Exact agreement where both answered: {r['exact_agreement_where_both_answered']:.1%}; kappa between runs {r['qwk_between_runs']:.3f}",
         f"- Cells where {name_b} abstained but {name_a} answered: {r['cells_b_abstained_a_answered']}"]
    if "human_a" in r:
        for nm, k in ((name_a, "human_a"), (name_b, "human_b")):
            h = r[k]
            L.append(f"- vs human, {nm}: answered {h['answered']}/{h['n']}, accuracy {h.get('accuracy', float('nan')):.3f}, "
                     f"kappa {h.get('qwk', float('nan')):.3f}, within 1: {h.get('within_1', float('nan')):.3f}")
        if "avoided_guesses_n" in r:
            L.append(f"- The {r['avoided_guesses_n']} guesses {name_b} avoided were right only "
                     f"{r['avoided_guesses_accuracy']:.1%} of the time exactly ({r['avoided_guesses_within_1']:.1%} within 1)")
        if "both_answered_accuracy_a" in r:
            L.append(f"- Where both answered: accuracy {name_a} {r['both_answered_accuracy_a']:.3f}, {name_b} {r['both_answered_accuracy_b']:.3f}")
    L += ["", f"Decision modes {name_a}: {r['mode_counts_a']}", f"Decision modes {name_b}: {r['mode_counts_b']}"]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=Path, required=True, help="results folder of run A (e.g. the v1 run)")
    ap.add_argument("--b", type=Path, required=True, help="results folder of run B (e.g. the v2 run)")
    ap.add_argument("--human", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    from evaluation.report import load_human

    human = load_human(a.human) if a.human else None
    ra, rb = load_run(a.a), load_run(a.b)
    try:
        text = format_report(compare_runs(ra, rb, human), a.a.name, a.b.name)
    except ValueError as exc:
        print(f"{exc}. A has {ra.paper_id.nunique()} papers, B has {rb.paper_id.nunique()}; run both on the same papers.", file=sys.stderr)
        return 2
    if a.out:
        a.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
