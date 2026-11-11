"""Trust score: combine the pipeline's transparency signals into one interpretable probability.

The paper's finding: an LLM answer is more likely to be wrong when the model needed more search
passes and when its self-reported confidence is lower. This module turns those two signals into

    trust = P( |LLM answer - human answer| <= 1 )     (target "near", default)   or   P(exact match)

so a researcher can (1) fit it on a SMALL human-coded sample of their own review (e.g. 20-30 papers),
(2) score every remaining LLM answer, and (3) route only the low-trust answers to human coders.

It is a plain logistic regression on a handful of features, stored as JSON (coefficients are
readable and portable - no pickles). No LLM fine-tuning is involved.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import analysis as A
from evaluation.robust import auroc

MODES = ["full_context", "retrieval", "expanded_retry", "forced_choice", "unresolved"]
FEATURES = ["confidence", "passes_extra", *[f"mode_{m}" for m in MODES]]


def trust_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix. `passes_extra` = passes beyond the first (0 when unknown)."""
    x = pd.DataFrame(index=df.index)
    x["confidence"] = df["confidence"].astype(float).fillna(0.0)
    passes = df["passes"] if "passes" in df else pd.Series(np.nan, index=df.index)
    x["passes_extra"] = (passes.astype(float) - 1).clip(lower=0).fillna(0.0)
    for m in MODES:
        x[f"mode_{m}"] = (df["decision_mode"] == m).astype(float)
    return x[FEATURES]


@dataclass
class TrustModel:
    coef: dict
    intercept: float
    mean: dict
    std: dict
    target: str = "near"

    def score(self, df: pd.DataFrame) -> pd.Series:
        x = trust_features(df)
        z = pd.Series(self.intercept, index=df.index, dtype=float)
        for f in FEATURES:
            z = z + self.coef[f] * (x[f] - self.mean[f]) / self.std[f]
        return 1.0 / (1.0 + np.exp(-z))

    def explain(self) -> pd.DataFrame:
        """Standardised coefficients: positive = raises trust, negative = lowers it."""
        return pd.DataFrame({"coef_per_sd": self.coef}).sort_values("coef_per_sd")

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "TrustModel":
        return TrustModel(**json.loads(Path(path).read_text(encoding="utf-8")))


def _target(df: pd.DataFrame, target: str) -> pd.Series:
    if target == "near":
        return (df["diff"] <= 1).astype(float)
    if target == "exact":
        return (df["diff"] == 0).astype(float)
    raise ValueError("target must be 'near' or 'exact'")


def fit_trust(df: pd.DataFrame, target: str = "near", C: float = 1.0) -> TrustModel:
    """Fit on rows that have a human label and a numeric answer."""
    from sklearn.linear_model import LogisticRegression

    d = df.dropna(subset=["answer_num", "diff"])
    x = trust_features(d)
    mean = x.mean().to_dict()
    std = {k: (v if v > 1e-9 else 1.0) for k, v in x.std(ddof=0).to_dict().items()}
    z = (x - pd.Series(mean)) / pd.Series(std)
    y = _target(d, target)
    clf = LogisticRegression(C=C, max_iter=2000).fit(z.to_numpy(), y.to_numpy())
    return TrustModel(coef=dict(zip(FEATURES, map(float, clf.coef_[0]))), intercept=float(clf.intercept_[0]),
                      mean={k: float(v) for k, v in mean.items()}, std={k: float(v) for k, v in std.items()}, target=target)


def cross_validated_trust(df: pd.DataFrame, target: str = "near", folds: int = 5, seed: int = 0) -> pd.DataFrame:
    """Out-of-fold trust scores (grouped by paper, so no paper is in both train and test)."""
    from sklearn.model_selection import GroupKFold

    d = df.dropna(subset=["answer_num", "diff"]).reset_index(drop=True)
    out = pd.Series(np.nan, index=d.index)
    gkf = GroupKFold(n_splits=max(2, min(folds, d["paper_id"].nunique())))
    for tr, te in gkf.split(d, groups=d["paper_id"]):
        out.iloc[te] = fit_trust(d.iloc[tr], target).score(d.iloc[te]).to_numpy()
    return d.assign(trust=out)


def evaluate_signals(df: pd.DataFrame, target: str = "near") -> dict:
    """Does the combined score separate good answers from bad ones better than either signal alone? (out-of-fold AUROC)"""
    d = cross_validated_trust(df, target)
    y = _target(d, target)
    effort = -A.effort_rank(d).fillna(0)
    return {
        "n": int(len(d)),
        "base_rate": float(y.mean()),
        "auroc_trust": auroc(d["trust"], y),
        "auroc_confidence_only": auroc(d["confidence"], y),
        "auroc_effort_only": auroc(effort, y),
        "ece_trust": A.expected_calibration_error(d["trust"], y),
    }


def tiers(scored: pd.DataFrame, high: float = 0.8, low: float = 0.5, score_col: str = "trust") -> pd.DataFrame:
    """High / Medium / Low trust tiers and the observed quality inside each (needs human labels)."""
    s = scored.copy()
    s["tier"] = pd.cut(s[score_col], bins=[-1, low, high, 2], labels=["low", "medium", "high"], right=False)
    rows = []
    for t in ["high", "medium", "low"]:
        g = s[s["tier"] == t]
        rows.append({"tier": t, "n": len(g), "share": len(g) / max(len(s), 1),
                     "exact_accuracy": float((g["diff"] == 0).mean()) if len(g) else float("nan"),
                     "within_1": float((g["diff"] <= 1).mean()) if len(g) else float("nan"),
                     "mean_abs_error": float(g["diff"].mean()) if len(g) else float("nan")})
    return pd.DataFrame(rows).set_index("tier")


def review_plan(scored: pd.DataFrame, review_fraction: float, score_col: str = "trust") -> dict:
    """Send the `review_fraction` lowest-trust answers to humans; accept the rest automatically.

    With human labels available (validation), also reports the quality of the auto-accepted set.
    """
    s = scored.sort_values(score_col).reset_index(drop=True)
    k = int(round(len(s) * review_fraction))
    review, accept = s.iloc[:k], s.iloc[k:]
    out = {"review_n": int(len(review)), "auto_accept_n": int(len(accept))}
    if "diff" in s and s["diff"].notna().any():
        out["auto_accept_within_1"] = float((accept["diff"] <= 1).mean()) if len(accept) else float("nan")
        out["auto_accept_exact"] = float((accept["diff"] == 0).mean()) if len(accept) else float("nan")
        out["overall_within_1"] = float((s["diff"] <= 1).mean())
        out["overall_exact"] = float((s["diff"] == 0).mean())
    out["review_cells"] = review[[c for c in ("paper_id", "q", "provider", score_col) if c in review]]
    return out
