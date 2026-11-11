"""Statistics that fit the data types of this study (see STATS_REVIEW.md).

Data types and the methods used here:
  * answers / human labels ...... ordinal 1-5            -> quadratic-weighted kappa, Krippendorff's alpha (ordinal), MAE
  * |model - human| ............. ordinal 0-4, many ties -> rank methods only (Spearman, Kendall tau-b, Kruskal-Wallis, Dunn)
  * decision mode / passes ...... ordinal effort           -> trend test (Spearman / Kendall), not just an unordered K-W
  * confidence .................. ordinal in practice      -> rank correlation + AUROC (NOT Pearson; it is clumped at 0.6/0.8/0.9)
  * correct (exact match) ....... binary                   -> proportions: McNemar (paired models), bootstrap CI, AUROC
  * independence ................ 8 questions x 4 models per paper => observations are CLUSTERED by paper, so
                                  p-values from row-level tests are too small. Every headline number therefore gets
                                  a paper-level cluster bootstrap CI.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from evaluation import analysis as A


# --------------------------------------------------------------------------- clustered uncertainty
def cluster_bootstrap(df: pd.DataFrame, stat: Callable[[pd.DataFrame], float], cluster: str = "paper_id",
                      n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile CI of `stat(df)` resampling whole papers (not rows) with replacement."""
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in df.groupby(cluster)}
    keys = np.array(list(groups))
    est = stat(df)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(keys, size=len(keys), replace=True)
        val = stat(pd.concat([groups[k] for k in pick], ignore_index=True))
        if np.isfinite(val):
            boots.append(val)
    if len(boots) < 20:
        return {"estimate": est, "lo": float("nan"), "hi": float("nan"), "p_two_sided_vs_0": float("nan"), "n_boot": len(boots)}
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    b = np.asarray(boots)
    p0 = 2 * min((b <= 0).mean(), (b >= 0).mean())
    return {"estimate": float(est), "lo": float(lo), "hi": float(hi), "p_two_sided_vs_0": float(min(1.0, p0)), "n_boot": len(boots)}


# --------------------------------------------------------------------------- association (ordinal-safe)
def spearman(x, y) -> tuple[float, float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~np.isnan(x) & ~np.isnan(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return float("nan"), float("nan")
    r = stats.spearmanr(x[ok], y[ok])
    return float(r.statistic), float(r.pvalue)


def kendall_tau_b(x, y) -> tuple[float, float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~np.isnan(x) & ~np.isnan(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return float("nan"), float("nan")
    r = stats.kendalltau(x[ok], y[ok], variant="b")
    return float(r.statistic), float(r.pvalue)


def effort_vs_error(df: pd.DataFrame, n_boot: int = 500, seed: int = 0) -> dict:
    """Does more search effort go with larger distance from the human answer? (the team's key finding)

    Uses the ordinal effort scale (retrieval < expanded_retry < forced_choice) and Spearman rho with a
    paper-level cluster-bootstrap CI. Row-level p is reported only for reference (it ignores clustering).
    """
    d = df.assign(effort=A.effort_rank(df)).dropna(subset=["effort", "diff"])
    rho, p = spearman(d["effort"], d["diff"])
    tau, _ = kendall_tau_b(d["effort"], d["diff"])
    ci = cluster_bootstrap(d, lambda g: spearman(g["effort"], g["diff"])[0], n_boot=n_boot, seed=seed)
    return {"n": int(len(d)), "spearman_rho": rho, "p_row_level": p, "kendall_tau_b": tau,
            "rho_ci_lo": ci["lo"], "rho_ci_hi": ci["hi"], "p_cluster_bootstrap": ci["p_two_sided_vs_0"]}


def confidence_vs_error(df: pd.DataFrame, n_boot: int = 500, seed: int = 0) -> dict:
    d = df.dropna(subset=["confidence", "diff"])
    rho, p = spearman(d["confidence"], d["diff"])
    ci = cluster_bootstrap(d, lambda g: spearman(g["confidence"], g["diff"])[0], n_boot=n_boot, seed=seed)
    return {"n": int(len(d)), "spearman_rho": rho, "p_row_level": p,
            "rho_ci_lo": ci["lo"], "rho_ci_hi": ci["hi"], "p_cluster_bootstrap": ci["p_two_sided_vs_0"]}


def epsilon_squared(df: pd.DataFrame, value: str = "diff", group: str = "decision_mode") -> float:
    """Effect size for Kruskal-Wallis: epsilon^2 = H / (n - 1)  (0 = none, 1 = perfect separation)."""
    k = A.kruskal_by_mode(df.rename(columns={group: "decision_mode"}), value)
    n = int(df[value].notna().sum())
    return float(k["H"] / (n - 1)) if n > 1 and np.isfinite(k["H"]) else float("nan")


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (uniformly more powerful than Bonferroni, same error control)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


# --------------------------------------------------------------------------- discrimination (is the signal useful?)
def auroc(score, positive) -> float:
    """P(score of a random positive > score of a random negative); ties count 0.5. 0.5 = useless."""
    s, y = np.asarray(score, float), np.asarray(positive, float)
    ok = ~np.isnan(s) & ~np.isnan(y)
    s, y = s[ok], y[ok].astype(bool)
    if y.all() or (~y).all():
        return float("nan")
    ranks = stats.rankdata(s)
    n1, n0 = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def signal_quality(df: pd.DataFrame, n_boot: int = 300, seed: int = 0) -> pd.DataFrame:
    """AUROC (with cluster CI) of each signal for predicting (a) exact match and (b) near-miss (|diff| <= 1)."""
    d = df.dropna(subset=["answer_num"]).copy()
    d["exact"] = (d["diff"] == 0).astype(float)
    d["near"] = (d["diff"] <= 1).astype(float)
    d["neg_effort"] = -A.effort_rank(d)            # higher = less effort = expected better
    d["neg_passes"] = -d["passes"]
    rows = []
    for sig in ("confidence", "neg_effort", "neg_passes"):
        sub = d.dropna(subset=[sig])
        if sub.empty:
            continue
        for target in ("exact", "near"):
            ci = cluster_bootstrap(sub, lambda g, s=sig, t=target: auroc(g[s], g[t]), n_boot=n_boot, seed=seed)
            rows.append({"signal": sig.replace("neg_", "-"), "predicts": target, "n": len(sub),
                         "auroc": ci["estimate"], "lo": ci["lo"], "hi": ci["hi"]})
    return pd.DataFrame(rows, columns=["signal", "predicts", "n", "auroc", "lo", "hi"])


# --------------------------------------------------------------------------- agreement / accuracy
def mcnemar(df: pd.DataFrame, a: str, b: str) -> dict:
    """Exact McNemar test: do providers a and b differ in how often they match the human (paired by paper+question)?"""
    w = df.pivot_table(index=["paper_id", "q"], columns="provider", values="correct", aggfunc="first").dropna(subset=[a, b])
    only_a = int(((w[a] == 1) & (w[b] == 0)).sum())
    only_b = int(((w[a] == 0) & (w[b] == 1)).sum())
    n = only_a + only_b
    p = float(stats.binomtest(only_a, n, 0.5).pvalue) if n else 1.0
    return {"a": a, "b": b, "n_pairs": int(len(w)), "a_only_correct": only_a, "b_only_correct": only_b, "p": p}


def pairwise_mcnemar(df: pd.DataFrame) -> pd.DataFrame:
    provs = sorted(df["provider"].unique())
    rows = []
    for i, a in enumerate(provs):
        for b in provs[i + 1:]:
            rows.append(mcnemar(df, a, b))
    out = pd.DataFrame(rows, columns=["a", "b", "n_pairs", "a_only_correct", "b_only_correct", "p"])
    if len(out):
        out["p_holm"] = holm(out["p"].tolist())
    return out


def krippendorff_alpha_ordinal(matrix: np.ndarray, levels=(1, 2, 3, 4, 5)) -> float:
    """Krippendorff's alpha, ordinal metric. `matrix`: raters x units, NaN = missing. The HCI-standard IRR for >2 raters."""
    m = np.asarray(matrix, float)
    levels = list(levels)
    idx = {v: i for i, v in enumerate(levels)}
    k = len(levels)
    o = np.zeros((k, k))
    for u in range(m.shape[1]):
        vals = m[:, u][~np.isnan(m[:, u])]
        if len(vals) < 2:
            continue
        for i, vi in enumerate(vals):
            for j, vj in enumerate(vals):
                if i != j:
                    o[idx[int(vi)], idx[int(vj)]] += 1.0 / (len(vals) - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()
    if n <= 1:
        return float("nan")
    # ordinal distance: delta^2(c,k) = (sum_{g=c..k} n_g - (n_c + n_k)/2)^2
    delta = np.zeros((k, k))
    for c in range(k):
        for kk in range(k):
            if c != kk:
                lo, hi = min(c, kk), max(c, kk)
                delta[c, kk] = (n_c[lo:hi + 1].sum() - (n_c[lo] + n_c[hi]) / 2) ** 2
    d_o = (o * delta).sum() / n
    d_e = (np.outer(n_c, n_c) * delta).sum() / (n * (n - 1))
    return float(1 - d_o / d_e) if d_e > 0 else float("nan")


def alpha_models_vs_human(df: pd.DataFrame) -> dict:
    """Krippendorff alpha across [human + all providers] and for each provider alone with the human."""
    w = df.pivot_table(index=["paper_id", "q"], columns="provider", values="answer_num", aggfunc="first")
    h = df.drop_duplicates(["paper_id", "q"]).set_index(["paper_id", "q"])["human"].reindex(w.index)
    out = {"all_raters": krippendorff_alpha_ordinal(np.vstack([h.to_numpy(), w.to_numpy().T]))}
    for prov in w.columns:
        out[prov] = krippendorff_alpha_ordinal(np.vstack([h.to_numpy(), w[prov].to_numpy()]))
    out["models_only"] = krippendorff_alpha_ordinal(w.to_numpy().T)
    return out


def robust_summary(df: pd.DataFrame, n_boot: int = 500, seed: int = 0) -> dict:
    """Everything the paper's results section needs, with clustered uncertainty."""
    per_model = {}
    for prov, g in df.groupby("provider"):
        acc = cluster_bootstrap(g, lambda x: A.accuracy(x["human"], x["answer_num"]), n_boot=n_boot, seed=seed)
        k = cluster_bootstrap(g, lambda x: A.qwk(x["human"], x["answer_num"]), n_boot=n_boot, seed=seed)
        per_model[prov] = {"accuracy": acc, "qwk": k, "chance_accuracy": A.chance_accuracy(g["human"], g["answer_num"])}
    return {
        "per_model": per_model,
        "effort_vs_error": effort_vs_error(df, n_boot, seed),
        "confidence_vs_error": confidence_vs_error(df, n_boot, seed),
        "kruskal_epsilon_squared_mode_vs_error": epsilon_squared(df, "diff"),
        "krippendorff_alpha": alpha_models_vs_human(df),
    }
