"""Tests for evaluation/analysis.py, evaluation/report.py and training/stacker.py."""
import json

import numpy as np
import pandas as pd
import pytest

from evaluation import analysis as A
from evaluation.report import build_report
from questions import PHT_FRAMEWORK_QUESTIONS as QS
from training import stacker


# ---------------------------------------------------------------- synthetic data helpers
def synth(n_papers=120, accs=None, seed=0):
    """Truth labels + providers that match the truth with given probability, else guess uniformly."""
    rng = np.random.default_rng(seed)
    accs = accs or {"openai": 0.55, "gemini": 0.40, "groq": 0.35, "mistral": 0.30}
    rows, truth = [], []
    for p in range(n_papers):
        for q in range(8):
            t = int(rng.integers(1, 6))
            truth.append({"paper_id": f"p{p}", "q": q, "human": t})
            for prov, acc in accs.items():
                hit = rng.random() < acc
                ans = t if hit else int(rng.integers(1, 6))
                conf = float(np.clip(0.5 + 0.4 * hit + rng.normal(0, 0.05), 0, 1))
                mode = rng.choice(["full_context", "retrieval", "expanded_retry", "forced_choice"],
                                  p=[0.3, 0.3, 0.1, 0.3] if hit else [0.1, 0.2, 0.1, 0.6])
                rows.append({"paper_id": f"p{p}", "provider": prov, "model_name": prov, "q": q, "answer": ans,
                             "answer_num": float(ans), "confidence": conf, "decision_mode": mode,
                             "prompt_version": "t", "n_evidence": 1})
    return pd.DataFrame(rows), pd.DataFrame(truth)


# ---------------------------------------------------------------- metrics
def test_accuracy_qwk_known_values():
    y = [1, 2, 3, 4, 5]
    assert A.accuracy(y, y) == 1.0 and A.qwk(y, y) == pytest.approx(1.0)
    assert A.qwk(y, [5, 4, 3, 2, 1]) < 0                       # reversed => negative agreement
    assert A.accuracy([1, 1, 2], [1, np.nan, 3]) == 0.5        # abstentions are excluded
    assert np.isnan(A.accuracy([1], [np.nan]))
    assert np.isnan(A.qwk([1], [1]))                          # undefined for <2 rows


def test_qwk_penalises_distance_more_than_accuracy():
    y = [1, 2, 3, 4, 5] * 20
    near = [2, 3, 4, 5, 4] * 20      # always off by one
    far = [5, 5, 1, 1, 1] * 20       # far off
    assert A.accuracy(y, near) == A.accuracy(y, far) == 0.0
    assert A.qwk(y, near) > A.qwk(y, far)


def test_chance_accuracy_all_same_label():
    assert A.chance_accuracy([3, 3, 3, 3], [3, 3, 3, 3]) == 1.0
    assert A.chance_accuracy([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(0.2)


def test_rank_biserial_sign_and_bounds():
    assert A.rank_biserial([5, 6, 7], [1, 2, 3]) == 1.0
    assert A.rank_biserial([1, 2, 3], [5, 6, 7]) == -1.0
    assert abs(A.rank_biserial([1, 2, 3, 4], [1, 2, 3, 4])) < 1e-9


def test_ece_perfect_and_overconfident():
    assert A.expected_calibration_error([1.0] * 10, [1.0] * 10) == 0.0
    assert A.expected_calibration_error([0.9] * 10, [0.0] * 10) == pytest.approx(0.9)


# ---------------------------------------------------------------- frame-level analyses
@pytest.fixture(scope="module")
def data():
    preds, human = synth()
    return preds, human, A.attach_human(preds, human)


def test_metric_tables_shape(data):
    _, _, df = data
    t = A.metric_table(df, A.accuracy)
    assert list(t.index) == sorted(df.provider.unique())
    assert {f"F{i}" for i in range(1, 9)} <= set(t.columns) and "overall" in t.columns
    assert t.loc["openai", "overall"] > t.loc["mistral", "overall"]    # 0.55 vs 0.30 generative accuracy
    s = A.summary_by_model(df)
    assert (s.accuracy > s.chance_accuracy - 0.05).all()


def test_decision_mode_is_informative(data):
    _, _, df = data
    k = A.kruskal_by_mode(df, "diff")
    assert k["p"] < 0.001 and k["H"] > 10
    d = A.dunn_by_mode(df, "diff")
    assert d.loc["forced_choice", "retrieval"] < 0.05
    mc = A.mode_comparisons(df, "diff")
    fc = mc[(mc.a == "retrieval") & (mc.b == "forced_choice")].iloc[0]
    assert fc.rank_biserial < 0            # retrieval errors are smaller than forced-choice errors


def test_selective_prediction_improves_accuracy(data):
    _, _, df = data
    for by in ("confidence", "decision_mode"):
        c = A.selective_curve(df, by)
        assert c.coverage.iloc[0] == pytest.approx(1.0)
        assert c.accuracy.iloc[-1] > c.accuracy.iloc[0] + 0.1


def test_ensembles_beat_weak_models(data):
    _, _, df = data
    for method in ("median", "mode", "conf_weighted"):
        e = A.ensemble(df, method)
        e = A.attach_human(e, df[["paper_id", "q", "human"]].drop_duplicates())
        assert len(e) == df[["paper_id", "q"]].drop_duplicates().shape[0]
        assert A.accuracy(e.human, e.answer_num) > A.accuracy(df[df.provider == "mistral"].human,
                                                              df[df.provider == "mistral"].answer_num)
    with pytest.raises(ValueError):
        A.ensemble(df, "nope")


def test_pairwise_agreement_symmetric_unit_diagonal(data):
    preds, _, _ = data
    m = A.pairwise_model_agreement(preds)
    assert np.allclose(np.diag(m), 1.0)
    assert np.allclose(m.values, m.values.T, equal_nan=True)


def test_calibration_reduces_ece(data):
    _, _, df = data
    c = A.calibrate_confidence(df)
    g = c[c.provider == "openai"]
    assert A.expected_calibration_error(g.calibrated_confidence, g.correct) <= \
           A.expected_calibration_error(g.confidence, g.correct) + 1e-9


# ---------------------------------------------------------------- loader + report
def _write_results(root, preds):
    for prov, g in preds.groupby("provider"):
        for pid, gp in g.groupby("paper_id"):
            d = root / prov
            d.mkdir(parents=True, exist_ok=True)
            with (d / f"{pid}.jsonl").open("w", encoding="utf-8") as fh:
                for _, r in gp.iterrows():
                    fh.write(json.dumps({"paper_id": pid, "question": QS[int(r.q)], "model_name": r.model_name,
                                         "prompt_version": r.prompt_version, "answer": int(r.answer),
                                         "confidence": r.confidence, "evidence": ["chunk_1"],
                                         "decision_mode": r.decision_mode}) + "\n")


def test_loader_filters_and_dedups(tmp_path, data):
    preds, _, _ = data
    _write_results(tmp_path / "res", preds[preds.paper_id.isin(["p0", "p1"])])
    df = A.load_results(tmp_path / "res", QS)
    assert len(df) == 2 * 8 * 4
    assert A.load_results(tmp_path / "res", QS, providers=["openai"]).provider.unique().tolist() == ["openai"]
    assert A.load_results(tmp_path / "res", QS, prompt_version="other").empty
    assert A.load_results(tmp_path / "res", QS, model_names=["mistral"]).provider.unique().tolist() == ["mistral"]


def test_report_with_and_without_human(tmp_path, data):
    preds, human, _ = data
    _write_results(tmp_path / "res", preds)
    r1 = build_report(tmp_path / "res", tmp_path / "o1")
    txt = r1.read_text(encoding="utf-8")
    assert "Inter-model agreement" in txt and "No human consensus supplied" in txt
    human.to_csv(tmp_path / "h.csv", index=False)
    r2 = build_report(tmp_path / "res", tmp_path / "o2", tmp_path / "h.csv")
    txt = r2.read_text(encoding="utf-8")
    for s in ("Overall vs human", "Kruskal-Wallis", "Dunn", "Ensembles", "Selective prediction",
              "Robust statistics", "Krippendorff", "McNemar", "Trust score", "Review plan"):
        assert s in txt
    assert (tmp_path / "o2" / "qwk_by_question.csv").exists()
    assert (tmp_path / "o2" / "trust_model.json").exists() and (tmp_path / "o2" / "signal_quality.csv").exists()


def test_report_bad_human_ids_fail_clearly(tmp_path, data):
    preds, human, _ = data
    _write_results(tmp_path / "res", preds.head(64))
    human.assign(paper_id="zzz").to_csv(tmp_path / "h.csv", index=False)
    with pytest.raises(SystemExit):
        build_report(tmp_path / "res", tmp_path / "o", tmp_path / "h.csv")


def test_human_tidy_validation(tmp_path):
    pd.DataFrame({"paper_id": ["a"], "q": [0]}).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError):
        A.load_human_tidy(tmp_path / "bad.csv")


# ---------------------------------------------------------------- training
def test_stacker_beats_best_single_model_grouped_cv():
    preds, human = synth(n_papers=150, seed=3)
    res = stacker.evaluate(preds, human, ["openai", "gemini", "groq", "mistral"], kinds=("logreg", "rf"))
    best_single = max(v["accuracy"] for k, v in res["baselines"].items() if k in ("openai", "gemini", "groq", "mistral"))
    assert res["stacker"]["rf"]["accuracy"] > best_single
    assert res["stacker"]["rf"]["qwk"] > 0.5
    assert res["n"] == 150 * 8 and res["papers"] == 150


def test_stacker_save_and_predict_on_new_papers(tmp_path):
    preds, human = synth(n_papers=80, seed=5)
    provs = ["openai", "gemini", "groq", "mistral"]
    path = stacker.fit_and_save(preds, human, provs, tmp_path / "m.joblib", kind="rf")
    new_preds, new_human = synth(n_papers=20, seed=99)
    new_preds["paper_id"] = "new_" + new_preds["paper_id"]
    new_human["paper_id"] = "new_" + new_human["paper_id"]
    out = stacker.predict(path, new_preds)
    assert len(out) == 20 * 8 and out.stacked_answer.between(1, 5).all()
    merged = out.merge(new_human, on=["paper_id", "q"])
    assert A.accuracy(merged.human, merged.stacked_answer) > 0.4


def test_stacker_handles_missing_provider_cells():
    preds, human = synth(n_papers=60, seed=7)
    preds = preds[~((preds.provider == "mistral") & (preds.q < 4))]      # a provider missing half the data
    res = stacker.evaluate(preds, human, ["openai", "gemini", "groq", "mistral"], kinds=("rf",))
    assert np.isfinite(res["stacker"]["rf"]["accuracy"])
