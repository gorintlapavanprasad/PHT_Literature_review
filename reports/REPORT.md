# PHT LLM evaluation report

Records: 6866 | papers: 217 | providers: gemini, llama, mistral, openai
Prompt versions present: {'v1': np.int64(6865), 'v2-cot-fewshot': np.int64(1)}

## Decision modes and self-reported confidence
|                               |        n |   median_conf |
|:------------------------------|---------:|--------------:|
| ('gemini', 'expanded_retry')  |   45.000 |         0.900 |
| ('gemini', 'forced_choice')   |  138.000 |         0.600 |
| ('gemini', 'full_context')    |  687.000 |         0.900 |
| ('gemini', 'not_applicable')  |    1.000 |         0.950 |
| ('gemini', 'retrieval')       |  851.000 |         0.900 |
| ('llama', 'expanded_retry')   |   51.000 |         0.800 |
| ('llama', 'forced_choice')    |  904.000 |         0.600 |
| ('llama', 'full_context')     |  370.000 |         0.800 |
| ('llama', 'retrieval')        |  387.000 |         0.800 |
| ('mistral', 'expanded_retry') |   87.000 |         0.900 |
| ('mistral', 'forced_choice')  | 1114.000 |         0.600 |
| ('mistral', 'full_context')   |  263.000 |         0.900 |
| ('mistral', 'retrieval')      |  256.000 |         0.900 |
| ('openai', 'expanded_retry')  |  105.000 |         0.700 |
| ('openai', 'forced_choice')   |  688.000 |         0.600 |
| ('openai', 'full_context')    |  481.000 |         0.860 |
| ('openai', 'retrieval')       |  438.000 |         0.780 |

## Inter-model agreement (quadratic-weighted kappa, no human labels needed)
|         |   gemini |   llama |   mistral |   openai |
|:--------|---------:|--------:|----------:|---------:|
| gemini  |    1.000 |   0.549 |     0.466 |    0.668 |
| llama   |    0.549 |   1.000 |     0.542 |    0.528 |
| mistral |    0.466 |   0.542 |     1.000 |    0.516 |
| openai  |    0.668 |   0.528 |     0.516 |    1.000 |

> No human consensus supplied: accuracy / kappa vs humans, significance tests, ensembles and
> calibration were skipped. Re-run with `--human <file>`.
