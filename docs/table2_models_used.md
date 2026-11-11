# Models that actually produced the results

| Results folder | Model id | Papers | Records | Prompt version | Size (from id) | Release date |
|---|---|---|---|---|---|---|
| gemini | gemini-2.5-flash | 217 | 1722 | v1 (1721), v2-cot-fewshot (1) | (look up) | (look up) |
| groq | llama-3.3-70b-versatile | 2 | 16 | v2-cot-fewshot (16) | 70B | (look up) |
| llama | meta-llama/Llama-3.3-70B-Instruct | 214 | 1712 | v1 (1712) | 70B | (look up) |
| mistral | mistralai/Mistral-Small-3.1-24B-Instruct-2503 | 215 | 1720 | v1 (1720) | 24B | (look up) |
| openai | gpt-5.1 | 214 | 1712 | v1 (1712) | (look up) | (look up) |

**Folders that contain more than one model** (should not be mixed):
- `mistral`: also llama-3.1-8b-instant

Decision modes per folder:

- gemini: expanded_retry 45, forced_choice 138, full_context 687, not_applicable 1, retrieval 851
- groq: not_applicable 8, retrieval 7, unresolved 1
- llama: expanded_retry 51, forced_choice 904, full_context 370, retrieval 387
- mistral: expanded_retry 87, forced_choice 1114, full_context 263, retrieval 256
- openai: expanded_retry 105, forced_choice 688, full_context 481, retrieval 438
