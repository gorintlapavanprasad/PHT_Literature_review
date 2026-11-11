# Statistics review: do the tests fit the data types?

_Requested in the team meeting: review whether the tests suit the data types, and rerun if not._
_Status: methods reviewed and implemented + unit-tested on synthetic data with known answers. **Not yet re-run on the real
results**, because the human consensus file is not on this machine - one command re-runs everything once it is:_
`cd paper_annotation && python -m evaluation.report --human <consensus.csv> --out ../reports`

## 1. Variable types (Shelby's mapping, with two refinements)

| Variable | Type | Note |
|---|---|---|
| Model answer, human answer (1-5) | ordinal | spacing between levels is not guaranteed equal |
| `abs(model - human)` (0-4) | ordinal, many ties | only 5 values; rank methods only |
| Decision mode | categorical **and ordered by effort** | full_context is *not* on the effort scale (set by paper length) |
| Passes (number of LLM calls) | count / ordinal | now recorded per answer (`passes` field) |
| Confidence (0-1) | **ordinal in practice**, not ratio | self-reported, clumped at 0.6 / 0.8 / 0.9; the distance 0.6->0.8 is not "the same" as 0.8->1.0 |
| Accuracy | proportion of binary outcomes | not an observation per row; compare with paired / proportion methods |
| Kappa / IRR | a *statistic* of paired ordinal labels | quadratic weights are the right choice for 1-5 scales |

## 2. Verdict per test

| Test in the manuscript | Suitable? | Problem | What the repo now does |
|---|---|---|---|
| Kruskal-Wallis (decision mode vs `abs(diff)`) | **Yes, with caveats** | (a) Rows are not independent: every paper contributes 8 questions x 4 models, so p-values are far too small (the quoted H=5322, n~6,900 is a symptom); (b) it ignores that modes are ordered by effort - a trend test answers the real question and is more powerful; (c) full_context vs retrieval differ because of paper *length*, not effort | `robust.effort_vs_error`: Spearman rho + Kendall tau-b on the ordinal effort scale (retrieval < expanded < forced), **paper-level cluster-bootstrap CI**; full_context kept as its own stratum; Kruskal effect size epsilon^2 reported |
| Kruskal-Wallis (decision mode vs confidence) | Yes | same independence issue; confidence has huge ties (tie-corrected in SciPy) | same treatment (`confidence_vs_error`) |
| Dunn post-hoc (Bonferroni) | Yes | Bonferroni is conservative; independence caveat | Bonferroni kept (paper tables unchanged) and Holm provided (`robust.holm`) |
| Rank-biserial between mode pairs | Yes as an effect size (equals Cliff's delta) | sign convention not defined; no CI | convention fixed: r>0 means the first group has larger error; use with CIs |
| Pearson-type reasoning on confidence | **No** (not used, but avoid) | confidence is ordinal | rank correlation + **AUROC** (`signal_quality`): does confidence/effort separate correct from wrong answers? 0.5 = useless |
| Accuracy per model/question | Fine as a descriptive | no uncertainty; 3 of 4 models are *below* the 0.20 chance rate, which is suspicious; accuracy ignores distance | cluster-bootstrap CI, **chance accuracy** next to every value, MAE, McNemar for paired model comparison (Holm-adjusted) |
| Quadratic-weighted kappa | **Yes** (right choice for ordinal) | kappa depends on label prevalence ("kappa paradox") when human labels are skewed; no CI | CI by cluster bootstrap; **Krippendorff's alpha (ordinal)** alongside, also across all raters (human + models) |

## 3. The three issues that matter most for the paper

1. **Non-independence.** Treating 6,880 rows as independent overstates certainty. Resample *papers* (done) - or, for the final
   paper, fit an ordinal mixed model with a random intercept for paper (R `ordinal::clmm`; not implemented here).
2. **Large n makes every p-value tiny.** Lead with effect sizes and CIs (rho, epsilon^2, AUROC), not "p < 0.0001".
3. **Chance level.** Report accuracy next to the accuracy expected by chance given each model's own answer distribution
   (`chance_accuracy`). Llama/Gemini/Mistral accuracy of 0.13-0.15 against a 0.20 chance rate should be re-checked (see PAPER_NOTES.md section 4).

## 4. Does effort predict error? (the team's key finding)

Recomputed with `robust.effort_vs_error` once human labels are available. The code is validated on synthetic data where the
true relationship is known: it recovers a real effect (CI excludes 0) and returns CI-spanning-0 when effort is noise.

## 5. Combining effort and confidence into something other researchers can use

`evaluation/trust.py` fits a tiny logistic regression (effort + confidence -> P(answer within 1 of the human)), stored as
readable JSON coefficients. A researcher codes ~20-30 papers by hand, fits it, then uses the score on the rest to
(a) tier answers high / medium / low trust and (b) choose which answers to send back to human coders (`review_plan`). Out-of-fold
AUROC is reported against each signal alone, so the paper can say whether combining helps. It does **not** require training or
fine-tuning a language model.
