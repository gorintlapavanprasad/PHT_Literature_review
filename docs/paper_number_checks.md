# Paper number checks

Every statement the manuscript makes about Tables 4-5, recomputed from the tables themselves.
Mismatches are marked **NO**.

| Claim | Stated | Computed | Match |
|---|---|---|---|
| mean kappa, GPT-5 | 0.1795 | 0.2468 | **NO** |
| mean kappa, Llama3 | -0.009 | -0.009 | yes |
| mean kappa, Gemini Pro | 0.0424 | 0.0424 | yes |
| mean kappa, Mistral | 0.0081 | 0.0081 | yes |
| highest kappa value (GPT-5, F3 = 0.545) | 0.545 | 0.545 | yes |
|   ...belongs to GPT-5 / F3 | GPT-5 F3 | GPT-5 F3 | yes |
| lowest kappa value (text: Llama3, -0.077, F7) | Llama3 F7 | Mistral F7 | **NO** |
|   ...its value | -0.077 | -0.105 | **NO** |
| average kappa for F4 (best framework) | 0.183 | 0.183 | yes |
| average kappa for F2 (worst framework) | 0.016 | 0.016 | yes |
| best framework by mean kappa is F4 | F4 | F4 | yes |
| worst framework by mean kappa is F2 | F2 | F7 | **NO** |
| printed Std. Dev. of kappa, GPT-5 (matches population sd, ddof=0) | 0.177 | 0.177 | yes |
| printed Std. Dev. of kappa, Llama3 (matches population sd, ddof=0) | 0.064 | 0.064 | yes |
| printed Std. Dev. of kappa, Gemini Pro (matches population sd, ddof=0) | 0.076 | 0.076 | yes |
| printed Std. Dev. of kappa, Mistral (matches population sd, ddof=0) | 0.058 | 0.058 | yes |
| mean accuracy, GPT-5 | 0.449 | 0.449 | yes |
| mean accuracy, Llama3 | 0.126 | 0.126 | yes |
| mean accuracy, Gemini Pro | 0.136 | 0.155 | **NO** |
| mean accuracy, Mistral | 0.15 | 0.149 | yes |
| average accuracy for F1 (best framework) | 0.251 | 0.251 | yes |
| average accuracy for F5 (worst framework) | 0.18 | 0.18 | yes |
| GPT-5 highest accuracy, F2 (text: 61%) | 0.61 | 0.612 | yes |
| Mistral lowest accuracy, F2 (text: 5.6%) | 0.056 | 0.056 | yes |
| printed Std. Dev. of accuracy, GPT-5 (matches population sd, ddof=0) | 0.125 | 0.125 | yes |
| printed Std. Dev. of accuracy, Llama3 (matches population sd, ddof=0) | 0.046 | 0.046 | yes |
| printed Std. Dev. of accuracy, Gemini Pro (matches population sd, ddof=0) | 0.07 | 0.07 | yes |
| printed Std. Dev. of accuracy, Mistral (matches population sd, ddof=0) | 0.056 | 0.056 | yes |

23 of 28 checks match; **5 do not.**

The printed Std. Dev. columns use the population formula (divide by 8), not the sample formula; say which in the paper.

Chance level (5 options = 0.20 for random guessing), for context:

- GPT-5: mean accuracy 0.449 is above the 0.20 chance level
- Llama3: mean accuracy 0.126 is BELOW the 0.20 chance level
- Gemini Pro: mean accuracy 0.155 is BELOW the 0.20 chance level
- Mistral: mean accuracy 0.149 is BELOW the 0.20 chance level
