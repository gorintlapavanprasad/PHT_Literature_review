"""Tests for evaluation/robust.py (data-type-appropriate statistics) and evaluation/trust.py."""
import numpy as np
import pandas as pd
import pytest

from evaluation import analysis as A
from evaluation import robust as R
from evaluation import trust as T
from test_analysis import synth      # synthetic data where signals are informative


@pytest.fixture(scope="module")
def df():
    preds, human = synth(n_papers=150, seed=11)
    # give every record a passes count consistent with its decision mode
    passes = {"full_context": 1, "retrieval": 1, "expanded_retry": 2, "forced_choice": 3}
    preds["passes"] = preds["decision_mode"].map(passes)
    return A.attach_human(preds, human)


# ---------------------------------------------------------------- primitives vs known answers
def test_auroc_known_values():
    assert R.auroc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0
    assert R.auroc([4, 3, 2, 1], [0, 0, 1, 1]) == 0.0
    assert R.auroc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5
    assert np.isnan(R.auroc([1, 2], [1, 1]))


def test_spearman_and_kendall_monotone():
    x = [1, 2, 3, 4, 5, 6]
    assert R.spearman(x, [2, 4, 6, 8, 10, 12])[0] == pytest.approx(1.0)
    assert R.kendall_tau_b(x, [6, 5, 4, 3, 2, 1])[0] == pytest.approx(-1.0)
    assert np.isnan(R.spearman([1, 1, 1, 1], [1, 2, 3, 4])[0])          # constant input -> undefined, not a crash


def test_holm_matches_hand_computation_and_dominates_bonferroni():
    p = [0.01, 0.04, 0.03]
    assert R.holm(p) == pytest.approx([0.03, 0.06, 0.06])
    assert all(h <= min(1, 3 * x) + 1e-12 for h, x in zip(R.holm(p), p))


def test_krippendorff_alpha_extremes():
    perfect = np.array([[1, 2, 3, 4, 5, 3, 2], [1, 2, 3, 4, 5, 3, 2]], float)
    assert R.krippendorff_alpha_ordinal(perfect) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    rand = rng.integers(1, 6, size=(2, 500)).astype(float)
    assert abs(R.krippendorff_alpha_ordinal(rand)) < 0.1
    with_missing = perfect.copy()
    with_missing[1, 0] = np.nan
    assert R.krippendorff_alpha_ordinal(with_missing) == pytest.approx(1.0)


def test_mcnemar_detects_difference_and_symmetry(df):
    r = R.mcnemar(df, "openai", "mistral")            # 0.55 vs 0.30 synthetic accuracy
    assert r["p"] < 1e-6 and r["a_only_correct"] > r["b_only_correct"]
    same = R.mcnemar(df, "openai", "openai")
    assert same["p"] == 1.0
    pm = R.pairwise_mcnemar(df)
    assert len(pm) == 6 and (pm.p_holm >= pm.p - 1e-12).all()


# ---------------------------------------------------------------- clustering-aware inference
def test_cluster_bootstrap_ci_contains_estimate_and_is_wider_than_naive(df):
    g = df[df.provider == "openai"]
    ci = R.cluster_bootstrap(g, lambda x: A.accuracy(x["human"], x["answer_num"]), n_boot=300, seed=1)
    assert ci["lo"] <= ci["estimate"] <= ci["hi"]
    assert 0.55 < ci["estimate"] < 0.75   # 0.55 hit rate + 20% lucky guesses on misses ~= 0.64
    assert ci["hi"] - ci["lo"] < 0.2


def test_clustering_inflates_uncertainty_when_papers_are_correlated():
    """If a paper's 8 questions share one outcome, row-level CIs are far too narrow; cluster CIs are not."""
    rng = np.random.default_rng(0)
    rows = []
    for p in range(60):
        v = rng.integers(1, 6)
        hit = rng.random() < 0.5                       # one outcome shared by all 8 questions of the paper
        for q in range(8):
            rows.append({"paper_id": f"p{p}", "q": q, "human": v, "answer_num": v if hit else (v % 5) + 1})
    d = pd.DataFrame(rows)
    d["diff"] = (d.answer_num - d.human).abs()
    cl = R.cluster_bootstrap(d, lambda x: float((x["diff"] == 0).mean()), n_boot=400, seed=0)
    naive_se = np.sqrt(0.25 / len(d))
    assert (cl["hi"] - cl["lo"]) / 2 > 1.0 * 1.96 * naive_se


def test_effort_vs_error_finds_informative_signal(df):
    r = R.effort_vs_error(df, n_boot=200)
    assert r["spearman_rho"] > 0.1 and r["rho_ci_lo"] > 0 and r["p_cluster_bootstrap"] < 0.05
    c = R.confidence_vs_error(df, n_boot=200)
    assert c["spearman_rho"] < -0.1 and c["rho_ci_hi"] < 0


def test_effort_vs_error_null_when_signal_is_noise():
    rng = np.random.default_rng(2)
    n = 1500
    d = pd.DataFrame({"paper_id": [f"p{i % 150}" for i in range(n)], "diff": rng.integers(0, 5, n).astype(float),
                      "decision_mode": rng.choice(["retrieval", "expanded_retry", "forced_choice"], n),
                      "confidence": rng.random(n)})
    r = R.effort_vs_error(d, n_boot=300)
    assert abs(r["spearman_rho"]) < 0.1 and r["rho_ci_lo"] < 0 < r["rho_ci_hi"]


def test_signal_quality_auroc_above_chance(df):
    sq = R.signal_quality(df, n_boot=100)
    conf = sq[(sq.signal == "confidence") & (sq.predicts == "near")].iloc[0]
    assert conf.auroc > 0.55 and conf.lo > 0.5
    assert set(sq.signal) == {"confidence", "-effort", "-passes"}


def test_robust_summary_shape(df):
    s = R.robust_summary(df, n_boot=100)
    assert set(s["per_model"]) == {"openai", "gemini", "groq", "mistral"}
    assert s["per_model"]["openai"]["accuracy"]["lo"] < s["per_model"]["openai"]["accuracy"]["hi"]
    assert s["krippendorff_alpha"]["openai"] > s["krippendorff_alpha"]["mistral"]
    assert 0 <= s["kruskal_epsilon_squared_mode_vs_error"] <= 1


# ---------------------------------------------------------------- trust score
def test_trust_is_better_than_either_signal_alone_and_calibrated(df):
    ev = T.evaluate_signals(df, "near")
    assert ev["auroc_trust"] > 0.6
    assert ev["auroc_trust"] >= max(ev["auroc_confidence_only"], ev["auroc_effort_only"]) - 0.01
    assert ev["ece_trust"] < 0.08


def test_trust_model_directions_and_roundtrip(df, tmp_path):
    m = T.fit_trust(df, "near")
    assert m.coef["confidence"] > 0                      # more confident -> more trust
    assert m.coef["mode_forced_choice"] < m.coef["mode_retrieval"]
    m.save(tmp_path / "trust.json")
    m2 = T.TrustModel.load(tmp_path / "trust.json")
    assert np.allclose(m.score(df), m2.score(df))
    assert m.score(df).between(0, 1).all()
    assert m.explain().index[0] in {f for f in T.FEATURES}


def test_trust_scores_new_unlabelled_data(df):
    m = T.fit_trust(df.iloc[: len(df) // 2], "near")
    new = df.iloc[len(df) // 2:].drop(columns=["human", "diff", "correct"])
    s = m.score(new)
    assert len(s) == len(new) and s.notna().all()


def test_tiers_are_ordered_by_quality(df):
    scored = T.cross_validated_trust(df, "near")
    t = T.tiers(scored)
    assert t.loc["high", "within_1"] > t.loc["low", "within_1"]
    assert t.loc["high", "exact_accuracy"] > t.loc["low", "exact_accuracy"]
    assert t.share.sum() == pytest.approx(1.0)


def test_review_plan_improves_auto_accepted_quality(df):
    scored = T.cross_validated_trust(df, "near")
    plan = T.review_plan(scored, 0.3)
    assert plan["auto_accept_within_1"] > plan["overall_within_1"]
    assert plan["review_n"] + plan["auto_accept_n"] == len(scored)
    assert len(plan["review_cells"]) == plan["review_n"]
    assert T.review_plan(scored, 0.0)["review_n"] == 0


def test_trust_target_validation(df):
    with pytest.raises(ValueError):
        T.fit_trust(df, "bogus")


def test_features_work_without_passes_column(df):
    t = T.trust_features(df.drop(columns=["passes"]))
    assert (t["passes_extra"] == 0).all()
