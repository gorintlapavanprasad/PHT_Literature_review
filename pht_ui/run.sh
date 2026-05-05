#!/usr/bin/env bash
# Start the PHT ScholarLens Coding Console.
#
# Assumes:
#   - your paper_annotation/.env has OPENAI_API_KEY, GEMINI_API_KEY, HF_TOKEN set
#   - the repo structure is:
#       <root>/literature-review-worker-model-main/paper_annotation/...
#       <root>/pht_ui/backend/server.py
#       <root>/pht_ui/frontend/index.html
#
# Override PIPELINE_ROOT env var if your layout differs.

set -euo pipefail
cd "$(dirname "$0")"

exec uvicorn backend.server:app \
  --host 127.0.0.1 \
  --port 8765 \
  --reload
