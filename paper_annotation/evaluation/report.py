"""CLI: build the full evaluation report (Markdown + CSVs).

    python -m evaluation.report --out ../reports                      # no human labels: agreement & modes
    python -m evaluation.report --human consensus.csv --out ../reports  # + accuracy/QWK/stats/ensembles
        (--human accepts the Covidence export or a tidy paper_id,q,human CSV)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
for p in (str(HERE), str(HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluation import analysis as A  # noqa: E402
from evaluation import robust as R  # noqa: E402
from evaluation import trust as T  # noqa: E402
from questions import PHT_FRAMEWORK_QUESTIONS  # noqa: E402


def _md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    return df.to_markdown(floatfmt=floatfmt) if hasattr(df, "to_markdown") and _has_tabulate() else "```\n" + df.round(3).to_string() + "\n```"


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def load_human(path: Path) -> pd.DataFrame:
    head = Path(path).read_text(encoding="utf-8-sig").splitlines()[0]
    if "Reviewer Name" in head:
        return A.human_from_covidence(path)
    return A.load_human_tidy(path)


def build_report(results_dir: Path, out_dir: Path, human_csv: Path | None = None,
                 providers: list[str] | None = None, prompt_version: str | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = A.load_results(results_dir, PHT_FRAMEWORK_QUESTIONS, providers=providers, prompt_version=prompt_version)
    if preds.empty:
        raise SystemExit("No results found for the given filters.")
    lines = ["# PHT LLM evaluation report", "",
             f"Records: {len(preds)} | papers: {preds.paper_id.nunique()} | providers: {', '.join(sorted(preds.provider.unique()))}",
             f"Prompt versions present: { {k: int(v) for k, v in preds.prompt_version.value_counts().items()} }", ""]

    modes = preds.groupby(["provider", "decision_mode"]).agg(n=("q", "size"), median_conf=("confidence", "median")).reset_index()
    modes.to_csv(out_dir / "decision_modes.csv", index=False)
    lines += ["## Decision modes and self-reported confidence", _md(modes.set_index(["provider", "decision_mode"])), ""]

    agree = A.pairwise_model_agreement(preds)
    agree.to_csv(out_dir / "inter_model_qwk.csv")
    lines += ["## Inter-model agreement (quadratic-weighted kappa, no human labels needed)", _md(agree), ""]

    if human_csv is None:
        lines += ["> No human consensus supplied: accuracy / kappa vs humans, significance tests, ensembles and",
                  "> calibration were skipped. Re-run with `--human <file>`.", ""]
    else:
        df = A.attach_human(preds, load_human(human_csv))
        if df.empty:
            raise SystemExit("Human labels matched no (paper, question) rows - check paper_id naming.")
        lines += [f"Matched with human consensus: {len(df)} rows over {df.paper_id.nunique()} papers.", ""]
        summ = A.summary_by_model(df)
        summ.to_csv(out_dir / "summary_by_model.csv")
        lines += ["## Overall vs human", _md(summ), ""]
        for name, metric in (("accuracy", A.accuracy), ("qwk", A.qwk)):
            t = A.metric_table(df, metric)
            t.to_csv(out_dir / f"{name}_by_question.csv")
            lines += [f"## {name} by framework question (paper Tables 4/5 layout)", _md(t), ""]
        kw = A.kruskal_by_mode(df, "diff")
        kc = A.kruskal_by_mode(df, "confidence")
        lines += ["## Decision mode vs correctness (Kruskal-Wallis)", f"|diff|: H={kw['H']:.2f}, p={kw['p']:.3g}",
                  f"confidence: H={kc['H']:.2f}, p={kc['p']:.3g}", ""]
        try:
            dunn = A.dunn_by_mode(df, "diff")
            dunn.to_csv(out_dir / "dunn_diff.csv")
            lines += ["### Dunn post-hoc (Bonferroni) on |model - human|", _md(dunn, ".4f"), ""]
        except Exception as exc:  # pragma: no cover - optional dependency
            lines += [f"(Dunn post-hoc skipped: {exc})", ""]
        mc = A.mode_comparisons(df, "diff")
        mc.to_csv(out_dir / "mode_comparisons.csv", index=False)
        lines += ["### Rank-biserial between modes (>0: first mode has larger error)", _md(mc.set_index(["a", "b"]) if len(mc) else mc, ".4f"), ""]

        # ---- statistics matched to the data types (see STATS_REVIEW.md) ----------------------------
        rs = R.robust_summary(df, n_boot=300)
        lines += ["## Robust statistics (paper-level cluster bootstrap, 95% CI)",
                  "| provider | accuracy [CI] | chance acc. | QWK [CI] |", "|---|---|---|---|"]
        for prov, v in rs["per_model"].items():
            a, k = v["accuracy"], v["qwk"]
            lines.append(f"| {prov} | {a['estimate']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}] | {v['chance_accuracy']:.3f} "
                         f"| {k['estimate']:.3f} [{k['lo']:.3f}, {k['hi']:.3f}] |")
        ev, cv = rs["effort_vs_error"], rs["confidence_vs_error"]
        lines += ["", f"- Effort vs |model-human| (Spearman, retrieval < expanded < forced): rho={ev['spearman_rho']:.3f} "
                      f"[{ev['rho_ci_lo']:.3f}, {ev['rho_ci_hi']:.3f}], n={ev['n']}",
                  f"- Confidence vs |model-human| (Spearman): rho={cv['spearman_rho']:.3f} [{cv['rho_ci_lo']:.3f}, {cv['rho_ci_hi']:.3f}]",
                  f"- Kruskal-Wallis effect size (epsilon^2), decision mode vs error: {rs['kruskal_epsilon_squared_mode_vs_error']:.4f}",
                  "- Krippendorff alpha (ordinal): " + ", ".join(f"{k}={v:.3f}" for k, v in rs["krippendorff_alpha"].items()), ""]
        sq = R.signal_quality(df, n_boot=150)
        sq.to_csv(out_dir / "signal_quality.csv", index=False)
        lines += ["### Can the signals tell good answers from bad ones? (AUROC, 0.5 = useless)", _md(sq.set_index(["signal", "predicts"])), ""]
        pm = R.pairwise_mcnemar(df)
        pm.to_csv(out_dir / "mcnemar_models.csv", index=False)
        lines += ["### Paired comparison of models (exact McNemar, Holm-adjusted)", _md(pm.set_index(["a", "b"]) if len(pm) else pm, ".4f"), ""]

        # ---- trust score -------------------------------------------------------------------------
        if df["answer_num"].notna().sum() >= 50:
            tev = T.evaluate_signals(df, "near")
            scored = T.cross_validated_trust(df, "near")
            tt = T.tiers(scored)
            tt.to_csv(out_dir / "trust_tiers.csv")
            T.fit_trust(df, "near").save(out_dir / "trust_model.json")
            lines += ["## Trust score (effort + confidence -> P(answer within 1 of the human))",
                      f"Out-of-fold AUROC: combined={tev['auroc_trust']:.3f}, confidence only={tev['auroc_confidence_only']:.3f}, "
                      f"effort only={tev['auroc_effort_only']:.3f}; base rate within-1={tev['base_rate']:.3f}; ECE={tev['ece_trust']:.3f}",
                      "", _md(tt), "",
                      "Review plan (send the least-trusted answers to humans):"]
            for frac in (0.1, 0.25, 0.5):
                rp = T.review_plan(scored, frac)
                lines.append(f"- review {frac:.0%}: auto-accepted answers are within-1 {rp['auto_accept_within_1']:.3f} "
                             f"(overall {rp['overall_within_1']:.3f}), exact {rp['auto_accept_exact']:.3f} (overall {rp['overall_exact']:.3f})")
            lines.append("")

        ens = pd.concat([A.ensemble(df, m) for m in ("median", "mode", "conf_weighted")])
        ens = A.attach_human(ens, df[["paper_id", "q", "human"]].drop_duplicates())
        es = A.summary_by_model(ens.assign(confidence=ens["confidence"].fillna(0)))
        es.to_csv(out_dir / "ensemble_summary.csv")
        lines += ["## Ensembles across models", _md(es), ""]

        for by in ("confidence", "decision_mode"):
            c = A.selective_curve(df, by)
            c.to_csv(out_dir / f"selective_{by}.csv", index=False)
            lines += [f"## Selective prediction by {by} (accuracy as coverage shrinks)", _md(c), ""]

    path = out_dir / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=HERE / "data" / "results")
    ap.add_argument("--human", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=HERE.parent / "reports")
    ap.add_argument("--providers", nargs="*", default=None)
    ap.add_argument("--prompt-version", default=None)
    a = ap.parse_args()
    print("Report written to", build_report(a.results, a.out, a.human, a.providers, a.prompt_version))


if __name__ == "__main__":
    main()
