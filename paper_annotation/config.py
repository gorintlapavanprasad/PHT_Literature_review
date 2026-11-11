"""Central configuration for the paper annotation pipeline."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from llm.registry import claude_ids


PIPELINE_CONFIG = {
    "chunk_size": 900,
    "chunk_overlap": 150,
    "temperature": 0.1,
    "embedding_model_name": "BAAI/bge-large-en-v1.5",
    "top_k": 15,
    "lexical_top_k": 50,
    "max_expansions": 2,
    "provider_concurrency": 3,
    "max_concurrency": 1,
    "ingest_concurrency": 2,
}

MODELS = {
    "openai": [
        "gpt-5.1",
    ],
    # Groq: OpenAI-compatible, very fast inference, free tier.
    "groq": [
        "llama-3.3-70b-versatile",
    ],
    # Mistral AI (la Plateforme): OpenAI-compatible, free experiment tier.
    "mistral": [
        "mistral-small-latest",
    ],
    # Legacy HF — kept empty but retained for backwards compat of provider resolution.
    "huggingface": [
    ],
    "gemini": [
        "gemini-2.5-flash",
    ],
}

# Fresh output directories — each provider gets its own. Old HF-produced
# results live under openai/, llama/, mistral/, gemini/ from the original
# pipeline; we deliberately use new directory names here to avoid conflating
# Groq-produced with HF-produced results.
MODEL_OUTPUT_DIRS = {
    "gpt-5.1": "openai",
    "llama-3.3-70b-versatile": "groq",
    "mistral-small-latest": "mistral",
    "gemini-2.5-flash": "gemini",
    # Keep legacy mappings so any leftover HF results are still readable if
    # someone points the UI at the old folders.
    "meta-llama/Llama-3.3-70B-Instruct": "llama",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": "mistral_hf",
}


# Opt-in models: NOT run by default (they cost money / need extra keys). Select them explicitly with
#   python main.py --model claude-sonnet-5-5 --model claude-opus-5-5 ...   or aliases: claude, compat
EXTRA_MODELS = {
    "anthropic": claude_ids(),   # every registered Claude version (llm/registry.py)
    # OpenAI-compatible hosts (Groq/Together/OpenRouter/Ollama...) for newer Llama / Mistral / Gemma:
    #   COMPAT_MODELS="google/gemma-3-27b-it,meta-llama/llama-4-maverick"
    "compat": [m.strip() for m in os.getenv("COMPAT_MODELS", "").split(",") if m.strip()],
}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def all_models() -> list[str]:
    """Return all configured model names in a stable order."""
    return [
        *MODELS["openai"],
        *MODELS["gemini"],
        *MODELS["groq"],
        *MODELS["mistral"],
        *MODELS["huggingface"],
    ]


def selectable_models() -> list[str]:
    """Default models plus the opt-in ones (what --model may name)."""
    return [*all_models(), *EXTRA_MODELS["anthropic"], *EXTRA_MODELS["compat"]]


def model_output_dir(model_name: str) -> str:
    """Map a model name to its results directory (one folder per model - runs never mix)."""
    if model_name in MODEL_OUTPUT_DIRS:
        return MODEL_OUTPUT_DIRS[model_name]
    if model_name in EXTRA_MODELS["anthropic"]:
        return f"anthropic_{_slug(model_name)}"
    if model_name in EXTRA_MODELS["compat"]:
        return f"compat_{_slug(model_name)}"
    raise ValueError(f"No results directory configured for model: {model_name}")


def model_provider(model_name: str) -> str:
    """Return provider key for a configured model."""
    if model_name in MODELS["openai"]:
        return "openai"
    if model_name in MODELS["groq"]:
        return "groq"
    if model_name in MODELS["mistral"]:
        return "mistral"
    if model_name in MODELS["huggingface"]:
        return "huggingface"
    if model_name in MODELS["gemini"]:
        return "gemini"
    if model_name in EXTRA_MODELS["anthropic"]:
        return "anthropic"
    if model_name in EXTRA_MODELS["compat"]:
        return "compat"
    raise ValueError(f"Unsupported model: {model_name}")


@dataclass(frozen=True)
class Config:
    """Configuration values for the pipeline."""

    data_dir: Path
    pdf_dir: Path
    index_dir: Path
    results_dir: Path
    chunk_size: int
    chunk_overlap: int
    top_k: int
    lexical_top_k: int
    max_expansions: int
    provider_concurrency: int
    max_concurrency: int
    ingest_concurrency: int
    model_name: str
    temperature: float
    embedding_model_name: str
    embedding_cache_path: Path
    openai_api_key: str
    gemini_api_key: str
    groq_api_key: str
    mistral_api_key: str
    anthropic_api_key: str
    compat_api_key: str
    compat_base_url: str
    huggingface_token: str
    huggingface_request_timeout: int
    prompt_version: str


def load_config() -> Config:
    """Load configuration from environment variables with defaults."""
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"

    def _clean_secret(value: str) -> str:
        return value.strip().strip('"').strip("'")

    openai_api_key = _clean_secret(os.getenv("OPENAI_API_KEY", ""))
    gemini_api_key = _clean_secret(os.getenv("GEMINI_API_KEY", ""))
    groq_api_key = _clean_secret(os.getenv("GROQ_API_KEY", ""))
    mistral_api_key = _clean_secret(os.getenv("MISTRAL_API_KEY", ""))
    huggingface_token = _clean_secret(os.getenv("HF_TOKEN", os.getenv("HUGGINGFACE_API_TOKEN", "")))

    return Config(
        data_dir=data_dir,
        pdf_dir=base_dir.parent / "papers",
        index_dir=data_dir / "indexes",
        results_dir=data_dir / "results",
        chunk_size=int(os.getenv("CHUNK_SIZE", str(PIPELINE_CONFIG["chunk_size"]))),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", str(PIPELINE_CONFIG["chunk_overlap"]))),
        top_k=int(os.getenv("TOP_K", str(PIPELINE_CONFIG.get("top_k")))),
        lexical_top_k=int(os.getenv("LEXICAL_TOP_K", str(PIPELINE_CONFIG.get("lexical_top_k")))),
        provider_concurrency=int(os.getenv("PROVIDER_CONCURRENCY", str(PIPELINE_CONFIG["provider_concurrency"]))),
        max_expansions=int(os.getenv("MAX_EXPANSIONS", str(PIPELINE_CONFIG["max_expansions"]))),
        max_concurrency=int(os.getenv("MAX_CONCURRENCY", str(PIPELINE_CONFIG.get("max_concurrency")))),
        ingest_concurrency=int(os.getenv("INGEST_CONCURRENCY", str(PIPELINE_CONFIG.get("ingest_concurrency")))),
        model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        temperature=float(os.getenv("TEMPERATURE", str(PIPELINE_CONFIG["temperature"]))),
        embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", PIPELINE_CONFIG["embedding_model_name"]),
        embedding_cache_path=data_dir / "embeddings_cache.sqlite",
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key,
        groq_api_key=groq_api_key,
        mistral_api_key=mistral_api_key,
        anthropic_api_key=_clean_secret(os.getenv("ANTHROPIC_API_KEY", "")),
        compat_api_key=_clean_secret(os.getenv("COMPAT_API_KEY", "")),
        compat_base_url=os.getenv("COMPAT_BASE_URL", "").strip(),
        huggingface_token=huggingface_token,
        huggingface_request_timeout=int(os.getenv("HUGGINGFACE_REQUEST_TIMEOUT", "120")),
        prompt_version=os.getenv("PROMPT_VERSION", "v2-cot-fewshot"),
    )


if __name__ == "__main__":
    config = load_config()
    print(config)
