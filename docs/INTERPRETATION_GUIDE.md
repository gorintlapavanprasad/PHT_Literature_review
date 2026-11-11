# How to interpret LLM annotations from this pipeline

For researchers who want to use an LLM to help code papers on an ordinal scale and need to decide **which answers to
trust**. The pipeline's contribution is not "the LLM is accurate" (on this study's data it mostly was not); it is that
every answer comes with evidence you can check and with signals that say how much to trust it.

> The numbers quoted from the study are the paper's own. Anything marked **(pending)** will be filled in from
> `reports/REPORT.md` once the human consensus file is available (issue #5, #7).

## What you get with every answer

| Field | What it tells you |
|---|---|
| `answer` | 1-5, or `Unknown` / `Not Applicable` (only if the honest-Unknown mode is on) |
| `reasoning` | the model's argument, written before the answer |
| `evidence` + `retrieved_chunks` | which pieces of the paper it relied on, with page numbers, so you can open them |
| `confidence` | the model's self-reported certainty, 0 to 1 |
| `decision_mode` | how much searching was needed (below) |
| `passes` | how many times the model had to be asked (the "effort") |

## Reading the decision mode

| Mode | Meaning | Reading it |
|---|---|---|
| `full_context` | short paper: the model saw all of it | the answer had every chance to be grounded; best reliability in the study |
| `retrieval` | found in the first search pass | usually trustworthy |
| `expanded_retry` | needed a wider search | treat with more care |
| `forced_choice` | still not found, so the model was made to guess (old pipeline only) | the study found these to be the least accurate; treat as a guess |
| `unresolved` | still not found, model said Unknown (new pipeline) | needs a human, by design |

`full_context` depends on paper length, not on difficulty, so compare it only with itself. Effort (retrieval, then
expanded retry, then forced choice) is the ordered scale to use when asking "does more effort mean more error?".

## Reading confidence

- Models in the study almost never reported confidence below 0.6, even when wrong. That is a known tendency to sound
  sure (sycophancy), so **do not read 0.9 as 90 % likely correct**.
- The ranking is still informative: the study found lower confidence went with larger errors. Use confidence to
  order answers, not as a probability. The trust score below turns it into a probability that you can check.
- Confidence is ordinal, not a measurement: 0.6 to 0.8 is not the same step as 0.8 to 1.0.

## The trust score (combine effort and confidence)

`evaluation/trust.py` combines `passes`, decision mode and confidence into one number: the probability that the LLM's
answer is within 1 point of what your human coders would say.

1. Code a **small sample** of papers yourself (the learning curve in the report shows how few are enough; **pending**).
2. Fit the score on that sample: `T.fit_trust(df, "near")` and save it to JSON.
3. Score every other answer. Group answers into **high / medium / low** trust (`T.tiers`).
4. Send the low-trust answers to human coders and accept the high-trust ones (`T.review_plan`). The report shows how
   good the auto-accepted answers are for each review budget (10 %, 25 %, 50 %).

What this does **not** give you: a guarantee. Re-check a random slice of the auto-accepted answers.

## Checks to run before trusting any number

- Compare models with the human baseline using **weighted kappa and accuracy together**; accuracy ignores how far
  off a wrong answer is, kappa corrects for chance. Always read accuracy next to the chance rate (0.20 for five options).
- Report confidence intervals that resample **papers**, not rows: every paper contributes many correlated answers
  (`evaluation/robust.py`). A p-value from the raw rows is far too optimistic.
- Keep one results folder per model and per prompt version; never mix them.
- If a model's agreement with another model is high but with humans is low, the models may share a viewpoint that
  differs from yours; that is a finding, not noise (in this study models agreed with each other at kappa 0.47-0.67).

## Limits

- The framework is subjective. Where an LLM and a human differ, the LLM's reasoning is sometimes sound; read the
  reasoning and the cited text before deciding who is right.
- The human baseline is itself the consensus of a small team; it is a reference, not the truth.
- Everything here was tested on one corpus (accessibility and games papers). Re-fit the trust score on your own data.
