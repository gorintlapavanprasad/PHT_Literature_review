# Paper review: "Applying Accountability and Transparency to an LLM-Powered Synthesis of Game-Based Research in ASSETS"

_Hagemann et al. (manuscript). Read in full and compared with this repo on 2026-11-10._

## 1. What the paper claims

- RQ1: can LLMs be prompted to support FAccT (confidence, evidence location, retrieval effort)?
- RQ2: how well do 4 LLMs reproduce HCI-researcher consensus on 8 ordinal (1-5) PHT questions over 215 TACCESS papers?
- Method: BGE embeddings + chunk retrieval; one question per prompt; decision modes (full context / retrieval / expanded retry / forced choice); self-reported confidence; metrics = accuracy and quadratic-weighted Cohen's kappa; Kruskal-Wallis + Dunn + rank-biserial.
- Result: poor agreement with humans (best: GPT-5, mean kappa ~0.18-0.25, accuracy 0.45; others ~0.13 accuracy, kappa ~0). Decision mode and confidence correlate with correctness.

## 2. Where the repo did NOT match the paper (and what was done)

| Paper needs | Repo state before | Now |
|---|---|---|
| Quadratic-weighted kappa, accuracy per model x question (Tables 4, 5) | `metrics.py` stubs raising `NotImplementedError` | `evaluation/analysis.py` (+ `report.py` CLI) |
| Kruskal-Wallis, Dunn, rank-biserial (Tables 6-9) | absent | implemented and tested |
| Human baseline | CSV not in repo | loader supports Covidence export **and** a tidy `paper_id,q,human` CSV |
| "Continuously expand K up to X times" | single 2x retry | `max_expansions` (default 2, env `MAX_EXPANSIONS`), tested |
| Forced-choice mode | removed in v2 prompt (honest `Unknown`) | still off by default; paper's data is mostly forced-choice, so v1 vs v2 should be reported as an ablation |
| Confidence "scaled down" (future work, Sec. 5/5.2) | not done | `calibrate_confidence` (isotonic, per provider) + ECE |
| Reproducibility of results tables | impossible | `python -m evaluation.report --human <csv>` regenerates everything |

## 3. Problems inside the paper itself (fix before submission)

1. **Table 1 is wrong.** Fun-vs-Utility options (4) and (5) and Play-vs-Game options (4) and (5) are copies of option (3). The correct text is in `paper_annotation/questions.py`. Also "Solo vs. Solo" should be "Solo vs. Social".
2. **Mean-kappa inconsistency.** Table 4 gives GPT-5 F1-F8 = 0.144, 0.070, 0.545, 0.538, 0.130, 0.211, 0.230, 0.106 -> mean **0.247**, but the text says **0.1795**. (Llama3 -0.009, Gemini 0.042, Mistral 0.008 do match their rows.)
3. **Lowest kappa misattributed.** Text: lowest is Llama3 -0.077 on F7. Table 4: Mistral F7 = **-0.105** is lower.
4. **Gemini retrieval count.** Text says 844 retrieval-mode answers; the stored results give 851.
5. **"Five Likert questions"** in the Introduction vs eight everywhere else.
6. **Annotation counts.** "1,720 annotations" (Fig. 2, Sec. 3.2) vs "6,880" (Sec. 3.3.1): 1,720 is per model (215 x 8); 6,880 is all four models. Say so explicitly.
7. **Model description vs code.** Paper: GPT-5, Gemini Pro, Llama-3 405B, Mistral Large, "no model under 100B". Repo/results: `gpt-5.1`, `gemini-2.5-flash`, `Llama-3.3-70B`, `Mistral-Small-3.1-24B` (24B). Table 2 parameter counts/dates are placeholders ("X" citations) and several are not public. The text must describe the models that actually produced the data.
8. **Unresolved placeholders:** ~40 `ToDo: [? ]` citations, `Table ??`, "expand K by X up to X times", Limitations section empty, template CCS concepts/keywords, wrong conference name.
9. **Cross-references and statistics:** Section 4.2 refers to "Table 11" (the paper has only Tables 1-9) and "Table ??" for the decision-mode distribution; Table 10 is cited but does not exist. The Kruskal-Wallis statistic H=5322.5264 is quoted twice - once for decision mode vs correctness and again for decision mode vs confidence - while a third, different H (504.33) is later given for decision mode vs correctness. These cannot all be right; recompute with `evaluation.analysis.kruskal_by_mode`.
10. **Rank-biserial sign convention** is never defined, and the same pair appears with opposite signs between Tables 7 and 9 (fine, different variables) - state the convention (`analysis.rank_biserial`: r>0 means the first group is larger).

## 4. Something worth auditing in the *results* (not asserted as a bug)

Table 5 reports accuracy 0.126 / 0.136 / 0.150 for Llama / Gemini / Mistral. With 5 options that is **below the 0.20 chance rate**, and many cells repeat exactly (0.154 appears six times across models; Llama and Gemini share F2 = 0.065). Yet on the stored predictions the models **agree exactly with GPT-5 on 29-55 % of cells** and agree with each other at quadratic kappa 0.47-0.67 (`reports/inter_model_qwk.csv`), while GPT-5 reaches 45 % accuracy against humans. Those facts are hard to reconcile unless the humans are systematically different from the three non-GPT models, so the human-label join (paper-id alignment, label mapping such as the typo'd Covidence strings) for those models deserves a re-check. Running the new report on the real consensus file will settle it either way.

## 5. How to get better results (ordered by expected payoff)

1. **Stop using forced choice as the default.** The paper itself shows forced choice is the least accurate mode and the most common (40-65 % of records). Use the v2 pipeline (`Unknown` allowed), widen retrieval with `max_expansions`, and report **selective prediction** (accuracy vs coverage) - `selective_curve`.
2. **Learn from the human labels.** `training/stacker.py` fits a meta-classifier on all providers' answers/confidences/decision modes with paper-grouped cross-validation; `training/export_finetune.py` creates train/val chat JSONL for supervised fine-tuning (LoRA or a provider API). Use the held-out papers for the headline numbers.
3. **Ensemble** (`ensemble`: median / mode / confidence-weighted) - cheap and usually beats every single model.
4. **Calibrate confidence** so it is interpretable (paper Sec. 5.2); report ECE.
5. **Ordinal-aware reporting:** always report QWK and MAE next to accuracy, plus chance accuracy.
6. **Fix the human-LLM subjectivity gap with few-shot examples drawn from the consensus set** (leave-one-paper-out) rather than the single Miller 2007 example.
7. **Retrieval:** BM25 + normalised-cosine dense retrieval + cross-encoder rerank; evaluate with `retrieval_hit_rate`.
8. **Data hygiene:** never mix prompt versions / models in one results folder (`load_results(prompt_version=..., model_names=...)` filters).

## 6. What cannot be done without you

- The **human consensus file** (Covidence export). Without it nothing can be scored against humans; all human-agreement code is verified only on synthetic labels with known ground truth.
- **API keys** (OpenAI/Gemini/Groq/Mistral) to re-run the pipeline with v2 and to run new papers.
- A **GPU or provider fine-tuning account** to execute the fine-tune; the dataset exporter and the CPU stacking trainer are ready.
