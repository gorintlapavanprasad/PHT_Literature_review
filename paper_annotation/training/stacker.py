"""Train a stacking meta-model that predicts the HUMAN consensus label.

Inputs per (paper, question): every provider's answer, confidence and decision mode, plus the
question id. Output: a calibrated classifier that is typically better aligned with the human
baseline than any single LLM (the paper's central weakness: kappa ~0).

Validation is grouped by paper (GroupKFold) so the model never sees a paper in both train and test.

    python -m training.stacker --human consensus.csv --out ../models/stacker.joblib
"""
from __future__ import annotations

import argparse
import json
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

MODES = ["full_context", "retrieval", "expanded_retry", "forced_choice", "unresolved", "not_applicable"]


def build_features(preds: pd.DataFrame, providers: list[str]) -> pd.DataFrame:
    """Wide feature frame indexed by (paper_id, q). Missing provider cells -> NaN (handled by the model)."""
    base = preds[["paper_id", "q"]].drop_duplicates().set_index(["paper_id", "q"])
    feats = base.copy()
    for prov in providers:
        g = preds[preds.provider == prov].set_index(["paper_id", "q"])
        feats[f"{prov}_ans"] = g["answer_num"]
        feats[f"{prov}_conf"] = g["confidence"]
        for m in MODES:
            feats[f"{prov}_mode_{m}"] = (g["decision_mode"] == m).astype(float)
    ans = [f"{p}_ans" for p in providers]
    feats["ans_mean"] = feats[ans].mean(axis=1)
    feats["ans_median"] = feats[ans].median(axis=1)
    feats["ans_std"] = feats[ans].std(axis=1).fillna(0)
    feats["n_providers"] = feats[ans].notna().sum(axis=1)
    for qi in range(len(PHT_FRAMEWORK_QUESTIONS)):
        feats[f"q_{qi}"] = (feats.index.get_level_values("q") == qi).astype(float)
    return feats.fillna(-1)  # tree models treat -1 as "missing"


def _make_model(kind: str, seed: int):
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "gbm":
        return GradientBoostingClassifier(n_estimators=120, max_depth=2, learning_rate=0.08, random_state=seed)
    if kind == "rf":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=-1)
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    raise ValueError(kind)


def cross_validate(feats: pd.DataFrame, y: pd.Series, groups: pd.Series, kind: str = "rf",
                   folds: int = 5, seed: int = 0) -> pd.Series:
    """Out-of-fold predictions with GroupKFold by paper."""
    from sklearn.model_selection import GroupKFold

    n_groups = groups.nunique()
    folds = max(2, min(folds, n_groups))
    oof = pd.Series(np.nan, index=feats.index)
    for tr, te in GroupKFold(n_splits=folds).split(feats, y, groups):
        classes = np.unique(y.iloc[tr])
        if len(classes) < 2:
            oof.iloc[te] = classes[0]
            continue
        model = _make_model(kind, seed).fit(feats.iloc[tr], y.iloc[tr])
        oof.iloc[te] = model.predict(feats.iloc[te])
    return oof


def evaluate(preds: pd.DataFrame, human: pd.DataFrame, providers: list[str], kinds=("logreg", "rf", "gbm"),
             seed: int = 0) -> dict:
    feats = build_features(preds, providers)
    h = human.set_index(["paper_id", "q"])["human"]
    idx = feats.index.intersection(h.index)
    feats, y = feats.loc[idx], h.loc[idx].astype(int)
    groups = pd.Series(idx.get_level_values("paper_id"), index=idx)
    out = {"n": int(len(idx)), "papers": int(groups.nunique()), "baselines": {}, "stacker": {}}
    for prov in providers:
        col = feats[f"{prov}_ans"].where(feats[f"{prov}_ans"] > 0)
        out["baselines"][prov] = {"accuracy": A.accuracy(y, col), "qwk": A.qwk(y, col)}
    med = feats["ans_median"].where(feats["ans_median"] > 0).round()
    out["baselines"]["ensemble_median"] = {"accuracy": A.accuracy(y, med), "qwk": A.qwk(y, med)}
    majority = float(y.value_counts(normalize=True).max())
    out["baselines"]["always_most_common_human_label"] = {"accuracy": majority, "qwk": 0.0}
    for kind in kinds:
        oof = cross_validate(feats, y, groups, kind, seed=seed)
        out["stacker"][kind] = {"accuracy": A.accuracy(y, oof), "qwk": A.qwk(y, oof)}
    return out


def fit_and_save(preds, human, providers, out_path: Path, kind: str = "rf", seed: int = 0) -> Path:
    import joblib

    feats = build_features(preds, providers)
    h = human.set_index(["paper_id", "q"])["human"]
    idx = feats.index.intersection(h.index)
    model = _make_model(kind, seed).fit(feats.loc[idx], h.loc[idx].astype(int))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "providers": providers, "columns": list(feats.columns), "kind": kind}, out_path)
    return out_path


def predict(bundle_path: Path, preds: pd.DataFrame) -> pd.DataFrame:
    """Apply a saved stacker to a fresh predictions frame (e.g. newly imported papers)."""
    import joblib

    b = joblib.load(bundle_path)
    feats = build_features(preds, b["providers"]).reindex(columns=b["columns"], fill_value=-1)
    proba = b["model"].predict_proba(feats)
    out = pd.DataFrame({"stacked_answer": b["model"].classes_[proba.argmax(1)],
                        "stacked_confidence": proba.max(1)}, index=feats.index)
    return out.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=HERE / "data" / "results")
    ap.add_argument("--human", type=Path, required=True)
    ap.add_argument("--providers", nargs="*", default=["openai", "gemini", "llama", "mistral"])
    ap.add_argument("--prompt-version", default=None)
    ap.add_argument("--out", type=Path, default=HERE.parent / "models" / "stacker.joblib")
    ap.add_argument("--kind", default="rf", choices=["rf", "gbm", "logreg"])
    a = ap.parse_args()
    from evaluation.report import load_human

    preds = A.load_results(a.results, PHT_FRAMEWORK_QUESTIONS, providers=a.providers, prompt_version=a.prompt_version)
    human = load_human(a.human)
    res = evaluate(preds, human, a.providers)
    print(json.dumps(res, indent=2))
    print("saved:", fit_and_save(preds, human, a.providers, a.out, a.kind))


if __name__ == "__main__":
    main()
