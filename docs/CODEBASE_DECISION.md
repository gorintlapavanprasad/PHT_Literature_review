# Which code base is the working copy? (issue #1)

Needs a team decision. Facts first, then a recommendation.

## The two versions

| | TACCESS-submitted repo ([shelbyhagemann/TACCESS-Accountability](https://github.com/shelbyhagemann/TACCESS-Accountability)) | This repo (`PHT_Literature_review`) |
|---|---|---|
| Prompt | v1, with the forced-choice guess | v2, honest `Unknown` allowed |
| Retrieval | single retry (x2) | `max_expansions` retries, stopword filter, backfill |
| Llama / Mistral | via Hugging Face | Groq, Mistral API, any OpenAI-compatible host |
| Newer models (Claude, GPT-5.6, Gemini 3.x, Gemma) | no | opt-in, one results folder per model |
| Web UI | no | yes (`pht_ui/`) |
| Tests | two live-API scripts | ~250 automated tests, runs without keys |
| Evaluation / statistics | none | accuracy, kappa, cluster bootstrap, trust score, reports |
| Papers and results | 215 PDFs, results for 4 models | results for 4 models, no PDFs |
| Matches the paper's numbers | **yes** (it produced them) | only approximately, with `FORCED_CHOICE=1`; the prompt text differs |

My evaluation code reads the TACCESS repo's results files without any changes.

## Options

1. **Freeze the TACCESS repo as the paper snapshot; develop here.** Tag the TACCESS commit that was submitted, keep it
   unchanged, and do all new work in this repo. Cheapest, no merge conflicts, and the paper stays reproducible.
2. **Port the new pieces onto the TACCESS repo as pull requests.** One code base, but the two have diverged in
   `config.py`, `prompts.py`, `main.py` and `annotate.py`, so it needs a careful, reviewed merge.
3. **Merge both into a single repo history.** Most work; only worth it if everyone wants one place.

## Recommendation

Option 1 now, option 2 later if the team wants it. The paper must keep pointing at the exact code that produced its
numbers, and the new work (newer models, statistics, trust score) is easier to review in its own repo.
To reproduce the paper's pipeline from this repo use `FORCED_CHOICE=1` and the v1 results.

## What I can't decide for you

Whether the lab wants a single repository, and who owns it. Add the decision as a comment on issue #1.
