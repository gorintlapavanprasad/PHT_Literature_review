# PHT Literature Review - web UI + pipeline.  Build:  docker build -t pht-review .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=8765
WORKDIR /app

# CPU-only torch first (the default wheel drags in multi-GB CUDA libraries).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
COPY paper_annotation/requirements.txt paper_annotation/requirements.txt
COPY pht_ui/backend/requirements.txt pht_ui/backend/requirements.txt
RUN pip install -r paper_annotation/requirements.txt -r pht_ui/backend/requirements.txt

COPY paper_annotation paper_annotation
COPY pht_ui pht_ui

# Persistent state (papers, indexes, results, embedding cache) lives under /data when mounted.
ENV PIPELINE_ROOT=/app/paper_annotation
VOLUME ["/app/papers", "/app/paper_annotation/data"]

EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request,os;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8765')+'/api/health')"
WORKDIR /app/pht_ui
CMD ["sh", "-c", "uvicorn backend.server:app --host 0.0.0.0 --port ${PORT}"]
