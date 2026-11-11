# API keys: where to get them and where to put them

The keys used for the original study belonged to a previous developer's personal accounts and are **not available**.
Create new keys on **lab- or project-owned accounts** (not personal ones) so the work survives people leaving.

## Rules (read first)

1. Put keys in **one place only**: `paper_annotation/.env` (this file is git-ignored; `scripts/check_env.py` verifies that).
2. **Never paste a key into chat, an issue, a pull request, a commit or a document.** If a key leaks, revoke it at the provider immediately.
3. Set a **monthly spend limit** on every account before the first run.
4. Check your setup any time with `python scripts/check_env.py` (prints READY/MISSING, never the keys) and
   `python scripts/check_env.py --scan-repo` (looks for keys committed by mistake).

## One-time setup

```bash
cp pht_ui/.env.example paper_annotation/.env      # Windows PowerShell: Copy-Item pht_ui\.env.example paper_annotation\.env
# open paper_annotation/.env in an editor, uncomment the lines you need, paste the value after the "="
python scripts/check_env.py
```

Format: `NAME=value`, one per line, no quotes and no spaces around `=`.

## Which key for which model

| Provider (models) | Variable in `.env` | Where to create it (UI paths can change) | Notes |
|---|---|---|---|
| **Anthropic** (all `claude-*` versions: Fable, Opus, Sonnet, Haiku) | `ANTHROPIC_API_KEY` | console.anthropic.com -> API Keys -> Create Key; add credits under Billing and set a spend limit | Fable models need the organisation's data retention set to 30 days (they error under zero-retention). Pricing: platform.claude.com/docs/en/about-claude/pricing |
| **OpenAI** (`gpt-5.1`, newer GPT-5.x) | `OPENAI_API_KEY` | platform.openai.com/api-keys; add billing and a usage limit | verify current model IDs at developers.openai.com/api/docs/models |
| **Google Gemini** (`gemini-*`) | `GEMINI_API_KEY` | aistudio.google.com/apikey | verify IDs at ai.google.dev/gemini-api/docs/models |
| **Groq** (hosted Llama) | `GROQ_API_KEY` | console.groq.com/keys | free tier has tight rate limits (`PROVIDER_CONCURRENCY=1`) |
| **Mistral** | `MISTRAL_API_KEY` | console.mistral.ai/api-keys | verify IDs at docs.mistral.ai/models |
| **Hugging Face** (legacy runs of the paper) | `HF_TOKEN` | huggingface.co/settings/tokens (read access is enough) | only needed to re-create the original Llama/Mistral setup |
| **Any OpenAI-compatible host** (newer Llama, Mistral, Gemma 4, ...) | `COMPAT_API_KEY`, `COMPAT_BASE_URL`, `COMPAT_MODELS` | e.g. OpenRouter: openrouter.ai/keys, base URL `https://openrouter.ai/api/v1`; or Together, Fireworks, a local Ollama/vLLM | model IDs are host-specific: copy them from the host's model page |

Example `paper_annotation/.env` (values are placeholders):

```
ANTHROPIC_API_KEY=paste-your-key-here
OPENAI_API_KEY=paste-your-key-here
GEMINI_API_KEY=paste-your-key-here
COMPAT_BASE_URL=https://openrouter.ai/api/v1
COMPAT_API_KEY=paste-your-key-here
COMPAT_MODELS=model-id-1,model-id-2
```

## Safe order for the first run

1. `python scripts/check_env.py` -> the providers you set show READY.
2. `python scripts/smoke_models.py --dry-run --models claude-all openai gemini` -> shows which models are ready (no API calls).
3. `python scripts/smoke_models.py --models claude-haiku-5-5` -> one tiny request on the cheapest model (cents).
4. `python scripts/smoke_models.py --models claude-all` -> checks every Claude version answers (also tells you wrong/retired IDs).
5. `python scripts/estimate_cost.py --papers-dir ../TACCESS-Accountability/papers --sample 20 --total-papers 217` -> price list, per model.
6. Pilot on 20 papers with the cheapest model, compare with the provider's usage page, then scale up.

## Troubleshooting

| Message | Meaning |
|---|---|
| `SKIP ... set ANTHROPIC_API_KEY` | the variable is empty or missing in `.env` |
| `401` / `authentication_error` | wrong or revoked key |
| `404` / `not_found_error` on a model | the model ID is wrong or retired: check the provider's model list |
| `429` / `rate_limit` | too many parallel calls: lower `PROVIDER_CONCURRENCY` |
| `credit balance is too low` | add credits / raise the spend limit |
