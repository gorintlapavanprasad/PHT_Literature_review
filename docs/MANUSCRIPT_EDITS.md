# Draft corrections for the manuscript

**For the authors to review and adapt.** Nothing here has been changed in the manuscript itself. Numbers come from the
manuscript's own tables (see `paper_number_checks.md`) or from the stored results; they must be regenerated from
the real human-comparison run once the consensus file is available (issues #5, #7), because Tables 4 and 5 are
themselves still unverified (issue #8: three models score below the 0.20 chance rate).

## 1. Table 1 (answer options repeated, "Solo vs. Solo")

Replace the whole table with `docs/table1.tex` (or `table1.md`). It is generated from `questions.py`, the text the models
were actually asked: eight questions, five distinct options each, and the fourth row is "Solo vs. Social".

## 2. Section 4.1 numbers that do not match the tables

| Where | Text now | Should say (from Tables 4 and 5) |
|---|---|---|
| Average kappa of GPT-5 | 0.1795 | 0.247 |
| Lowest kappa | "Llama3, -0.077, framework 7" | Mistral, -0.105, framework 7 (Llama3's lowest is -0.077 on F7, but Mistral's is lower) |
| Worst framework by average kappa | F2 (0.016) | F7 (about 0.000); F2 is second-lowest at 0.016 |
| Average accuracy of Gemini | 0.136 | 0.155 |
| Std. Dev. columns | (not explained) | add: "population standard deviation over the eight frameworks" |
| Gemini retrieval-mode answers | 844 | 851 (stored results) |

Everything else the section states about means and best frameworks reproduces from the tables.

## 3. Sec. 3.1 and Table 2: the models that actually produced the data

The manuscript describes GPT-5, Gemini Pro, Llama-3 (405B) and Mistral Large (675B) and says no model under 100B was used.
The stored results were produced by the models below (`docs/table2_models_used.md`). Either describe these, or re-run
with the models the text describes (issues #14 to #17).

| Provider | Model id in the results | Papers | Records |
|---|---|---|---|
| OpenAI | gpt-5.1 | 214 | 1,712 |
| Google | gemini-2.5-flash (a Flash model, not Pro) | 214 | 1,712 |
| Meta | meta-llama/Llama-3.3-70B-Instruct (70B) | 214 | 1,712 |
| Mistral AI | mistralai/Mistral-Small-3.1-24B-Instruct-2503 (24B) | 214 | 1,712 |

Papers and records above are from the results shipped in the TACCESS-submitted repo. The working copy of this repo has a few
more files (217 Gemini, 215 Mistral papers), including two test runs; use the submitted counts for the paper.

Draft replacement for the selection paragraph (check each claim against what the team actually decided):

> We evaluated four models: two proprietary (gpt-5.1, gemini-2.5-flash) and two open-weight (Llama-3.3-70B-Instruct,
> Mistral-Small-3.1-24B-Instruct), selected for availability through hosted APIs and recency at the time of the study.
> Parameter counts for the proprietary models are not public.

If the "100 billion parameters" criterion stays, it contradicts the 70B and 24B models and must be dropped or the models replaced.
Parameter sizes and release dates in Table 2 should only be filled in from the providers' own documentation.

## 4. Sec. 3.3.2: "expand K by X up to X times"

The submitted (v1) pipeline retrieves the top K = 15 chunks; if the answer is still unknown it doubles K once (15 to 30), and if
that fails it forces a 1 to 5 answer. Suggested text:

> If no answer is found, K is doubled once (from 15 to 30 chunks); if the answer is still unknown, the model is forced to choose.

(The newer code allows up to two expansions and, by default, keeps "Unknown" instead of forcing a choice; say so only if
those runs are reported.)

## 5. Smaller corrections

- Introduction: "a set of five Likert scale questions" is eight everywhere else.
- Counts: the text says 215 papers, "1,720 annotated responses" and "6,880 annotations". The submitted results contain **214 papers
  and 1,712 records per model (6,848 in total)**. The missing paper is `gotfrid16`: its PDF is in `papers/` but none of the four models has a results file for it (it was
  probably skipped by a failed run). Either re-run it or correct the counts to 214.
  State per-model and total counts once, using the real numbers.
- Cross-references: the text cites Table 10 and Table 11, and "Table ??" for the decision-mode distribution; the paper has only
  Tables 1 to 9. The Kruskal-Wallis value H = 5322.5264 is given for two different comparisons and H = 504.33 for a third:
  recompute with `evaluation/robust.py` and cite each once.
- Rank-biserial: define the sign convention (positive means the first mode has the larger error).
- Remove template leftovers: "Do Not Use This Code" CCS concepts and keywords, the conference name, and all `ToDo: [? ]` citations.

## 6. Sections still empty (drafts to edit)

**Limitations (draft):**

> Our results come from one corpus (TACCESS papers on games and accessibility) and one human baseline: the consensus of a small
> research team that does not represent all disability communities. The framework is subjective, so some model answers that
> differ from the baseline were defensible. Models are a snapshot: newer versions may behave differently. Self-reported
> confidence is not calibrated and models rarely reported values below 0.6. Observations are clustered by paper, so
> significance values from row-level tests overstate certainty; we report intervals that resample papers.

**Use of AI (draft; the authors must confirm what applies):**

> LLMs were used as the object of study to annotate the literature. [State here whether AI tools were also used for writing,
> coding or analysis, as the venue's policy requires.]
