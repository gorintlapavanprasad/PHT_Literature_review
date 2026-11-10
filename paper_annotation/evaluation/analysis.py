"""Reproducible evaluation of LLM annotations against human consensus.

Implements the analyses reported in the paper (Hagemann et al.):
  * accuracy and quadratic-weighted Cohen's kappa per model x framework question
  * decision-mode / confidence vs correctness (Kruskal-Wallis, Dunn post-hoc, rank-biserial)
and adds the analyses the paper lists as future work / that make results *better*:
  * chance-level baselines (so accuracy is interpretable)
  * selective prediction (accuracy vs coverage when filtering on confidence / decision mode)
  * ensembles across models, confidence calibration (ECE)
  * inter-model agreement (works even without human labels)

Everything works on tidy pandas DataFrames so it is easy to test and reuse.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

LABELS = [1, 2, 3, 4, 5]
DECISION_MODES = ["full_context", "retrieval", "expanded_retry", "forced_choice", "unresolved", "not_applicable"]


# --------------------------------------------------------------------------- loading
def load_results(
    results_dir: Path,
    questions: list[str],
    providers: Optional[Iterable[str]] = None,
    prompt_version: Optional[str] = None,
    model_names: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Load every JSONL under results_dir/<provider>/ into one tidy frame.

    Columns: paper_id, provider, model_name, q (0-based), answer (raw), answer_num (1-5 or NaN),
    confidence, decision_mode, prompt_version, n_evidence.
    Filtering by prompt_version / model_names prevents mixing runs (a known data-hygiene issue).
    """
    qindex = {q: i for i, q in enumerate(questions)}
    rows = []
    results_dir = Path(results_dir)
    wanted = set(providers) if providers else None
    models = set(model_names) if model_names else None
    for pdir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        if wanted and pdir.name not in wanted:
            continue
        for f in sorted(pdir.glob("*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                q = qindex.get(r.get("question"))
                if q is None:
                    continue
                if prompt_version and r.get("prompt_version") != prompt_version:
                    continue
                if models and r.get("model_name") not in models:
                    continue
                ans = r.get("answer")
                num = float(ans) if isinstance(ans, int) and not isinstance(ans, bool) and 1 <= ans <= 5 else np.nan
                rows.append({
                    "paper_id": r.get("paper_id") or f.stem,
                    "provider": pdir.name,
                    "model_name": r.get("model_name"),
                    "q": q,
                    "answer": ans,
                    "answer_num": num,
                    "confidence": float(r.get("confidence") or 0.0),
                    "decision_mode": r.get("decision_mode"),
                    "prompt_version": r.get("prompt_version"),
                    "n_evidence": len(r.get("evidence") or []),
                })
    df = pd.DataFrame(rows, columns=[
        "paper_id", "provider", "model_name", "q", "answer", "answer_num",
        "confidence", "decision_mode", "prompt_version", "n_evidence"])
    # one record per (paper, provider, question): keep the last written
    return df.drop_duplicates(["paper_id", "provider", "q"], keep="last").reset_index(drop=True)


def load_human_tidy(csv_path: Path) -> pd.DataFrame:
    """Read a tidy human-label CSV with columns paper_id,q,human (q 0-based, human 1-5)."""
    h = pd.read_csv(csv_path)
    missing = {"paper_id", "q", "human"} - set(h.columns)
    if missing:
        raise ValueError(f"human CSV missing columns: {sorted(missing)}")
    return h[["paper_id", "q", "human"]].dropna().astype({"q": int, "human": int})


def human_from_covidence(csv_path: Path) -> pd.DataFrame:
    """Parse the Covidence export (Reviewer Name == 'Consensus') into the tidy human frame."""
    from pht_ui.backend import human_consensus  # namespace package; repo root must be on sys.path

    answers = human_consensus.load_human_answers(Path(csv_path))
    rows = [{"paper_id": pid, "q": q, "human": s} for pid, d in answers.items() for q, s in d.items()]
    return pd.DataFrame(rows, columns=["paper_id", "q", "human"])


def attach_human(preds: pd.DataFrame, human: pd.DataFrame) -> pd.DataFrame:
    out = preds.merge(human, on=["paper_id", "q"], how="inner")
    out["diff"] = (out["answer_num"] - out["human"]).abs()
    out["correct"] = (out["answer_num"] == out["human"]).astype(float).where(out["answer_num"].notna())
    return out


# --------------------------------------------------------------------------- metrics
def accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ok = ~np.isnan(y_pred)
    return float((y_true[ok] == y_pred[ok]).mean()) if ok.any() else float("nan")


def qwk(y_true, y_pred) -> float:
    """Quadratic-weighted Cohen's kappa on the fixed 1-5 label set (NaN if undefined)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ok = ~np.isnan(y_pred) & ~np.isnan(y_true)
    if ok.sum() < 2:
        return float("nan")
    k = cohen_kappa_score(y_true[ok].astype(int), y_pred[ok].astype(int), weights="quadratic", labels=LABELS)
    return float(k) if np.isfinite(k) else float("nan")


def chance_accuracy(y_true, y_pred) -> float:
    """Expected accuracy if the model guessed independently with its own answer distribution."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ok = ~np.isnan(y_pred)
    if not ok.any():
        return float("nan")
    ph = np.array([(y_true[ok] == k).mean() for k in LABELS])
    pm = np.array([(y_pred[ok] == k).mean() for k in LABELS])
    return float((ph * pm).sum())


def metric_table(df: pd.DataFrame, metric=accuracy, by: str = "provider") -> pd.DataFrame:
    """Rows = provider, columns = F1..F8 + overall + std (the layout of paper Tables 4/5)."""
    out = {}
    for name, g in df.groupby(by):
        row = {f"F{q + 1}": metric(sub["human"], sub["answer_num"]) for q, sub in g.groupby("q")}
        row["overall"] = metric(g["human"], g["answer_num"])
        vals = [v for k, v in row.items() if k.startswith("F") and np.isfinite(v)]
        row["std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
        out[name] = row
    return pd.DataFrame(out).T


def summary_by_model(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in df.groupby("provider"):
        rows.append({
            "provider": name,
            "n": len(g),
            "answered": float(g["answer_num"].notna().mean()),
            "accuracy": accuracy(g["human"], g["answer_num"]),
            "chance_accuracy": chance_accuracy(g["human"], g["answer_num"]),
            "qwk": qwk(g["human"], g["answer_num"]),
            "mae": float(g["diff"].mean()),
            "mean_confidence": float(g["confidence"].mean()),
            "ece": expected_calibration_error(g["confidence"], g["correct"]),
        })
    return pd.DataFrame(rows).set_index("provider")


def expected_calibration_error(conf, correct, bins: int = 10) -> float:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    ok = ~np.isnan(correct)
    conf, correct = conf[ok], correct[ok]
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(ece)


# --------------------------------------------------------------------------- statistics from the paper
def kruskal_by_mode(df: pd.DataFrame, value: str = "diff") -> dict:
    """Kruskal-Wallis of `value` (|model-human| or confidence) across decision modes."""
    groups = {m: g[value].dropna().to_numpy() for m, g in df.groupby("decision_mode") if g[value].notna().sum() > 0}
    groups = {m: v for m, v in groups.items() if len(v) >= 2}
    if len(groups) < 2:
        return {"H": float("nan"), "p": float("nan"), "groups": list(groups)}
    H, p = stats.kruskal(*groups.values())
    return {"H": float(H), "p": float(p), "groups": list(groups)}


def dunn_by_mode(df: pd.DataFrame, value: str = "diff") -> pd.DataFrame:
    """Dunn post-hoc p-value matrix (Bonferroni-adjusted) between decision modes."""
    import scikit_posthocs as sp

    d = df.dropna(subset=[value])
    counts = d["decision_mode"].value_counts()
    d = d[d["decision_mode"].isin(counts[counts >= 2].index)]
    return sp.posthoc_dunn(d, val_col=value, group_col="decision_mode", p_adjust="bonferroni")


def rank_biserial(a, b) -> float:
    """Rank-biserial correlation r = 2U/(n1 n2) - 1 (>0 means `a` tends to be larger than `b`)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    U = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(2 * U / (len(a) * len(b)) - 1)


def mode_comparisons(df: pd.DataFrame, value: str = "diff") -> pd.DataFrame:
    """Pairwise Mann-Whitney p and rank-biserial between decision modes (paper Tables 7/9)."""
    rows = []
    modes = [m for m in DECISION_MODES if (df["decision_mode"] == m).sum() >= 2]
    for m1, m2 in combinations(modes, 2):
        a = df.loc[df["decision_mode"] == m1, value].dropna()
        b = df.loc[df["decision_mode"] == m2, value].dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
        rows.append({"a": m1, "b": m2, "p": float(p), "rank_biserial": rank_biserial(a, b)})
    return pd.DataFrame(rows, columns=["a", "b", "p", "rank_biserial"])


# --------------------------------------------------------------------------- making results better
def selective_curve(df: pd.DataFrame, by: str = "confidence", steps: int = 10) -> pd.DataFrame:
    """Accuracy/QWK/MAE when keeping only the top fraction of answers ranked by `by`.

    If an answer's confidence (or decision mode) is informative, accuracy rises as coverage falls.
    """
    d = df.dropna(subset=["answer_num"]).copy()
    if by == "decision_mode":
        rank = {"full_context": 0, "retrieval": 1, "expanded_retry": 2, "forced_choice": 3}
        d["_score"] = -d["decision_mode"].map(rank).fillna(4) + d["confidence"] * 0.01
    else:
        d["_score"] = d[by]
    d = d.sort_values("_score", ascending=False).reset_index(drop=True)
    rows = []
    for frac in np.linspace(1.0, 1.0 / steps, steps):
        k = max(2, int(round(len(d) * frac)))
        top = d.iloc[:k]
        rows.append({"coverage": k / len(d), "n": k, "accuracy": accuracy(top["human"], top["answer_num"]),
                     "qwk": qwk(top["human"], top["answer_num"]), "mae": float(top["diff"].mean())})
    return pd.DataFrame(rows)


def ensemble(df: pd.DataFrame, method: str = "median") -> pd.DataFrame:
    """Combine providers per (paper, q). Methods: median, mode, conf_weighted (confidence-weighted vote)."""
    out = []
    for (pid, q), g in df.dropna(subset=["answer_num"]).groupby(["paper_id", "q"]):
        a = g["answer_num"].to_numpy()
        if method == "median":
            pred = float(np.round(np.median(a)))
        elif method == "mode":
            vals, cnt = np.unique(a, return_counts=True)
            best = vals[cnt == cnt.max()]
            pred = float(np.round(np.median(best)))
        elif method == "conf_weighted":
            w = {k: g.loc[g["answer_num"] == k, "confidence"].sum() for k in np.unique(a)}
            pred = float(max(w, key=w.get))
        else:
            raise ValueError(method)
        out.append({"paper_id": pid, "q": q, "provider": f"ensemble_{method}", "answer_num": pred,
                    "confidence": float(g["confidence"].mean()), "decision_mode": "ensemble"})
    return pd.DataFrame(out)


def pairwise_model_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """QWK between every pair of providers on shared (paper, q) cells - no human labels needed."""
    wide = df.pivot_table(index=["paper_id", "q"], columns="provider", values="answer_num", aggfunc="first")
    provs = list(wide.columns)
    m = pd.DataFrame(np.nan, index=provs, columns=provs)
    for a in provs:
        for b in provs:
            if a == b:
                m.loc[a, b] = 1.0
                continue
            both = wide[[a, b]].dropna()
            m.loc[a, b] = qwk(both[a], both[b]) if len(both) >= 2 else np.nan
    return m


def calibrate_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """Per-provider isotonic recalibration of self-reported confidence -> P(correct).

    Addresses the paper's observation that confidence is always >0.5 yet only weakly informative.
    Returns a copy with a `calibrated_confidence` column (fit on the same frame; use CV for reporting).
    """
    from sklearn.isotonic import IsotonicRegression

    out = df.copy()
    out["calibrated_confidence"] = np.nan
    for prov, g in out.groupby("provider"):
        g = g.dropna(subset=["correct"])
        if len(g) < 10 or g["confidence"].nunique() < 2:
            continue
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(g["confidence"], g["correct"])
        m = out["provider"] == prov
        out.loc[m, "calibrated_confidence"] = iso.predict(out.loc[m, "confidence"])
    return out
