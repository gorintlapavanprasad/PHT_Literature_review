"""List the model IDs a provider really offers, so you can copy exact, working IDs (never prints keys).

    python scripts/list_models.py groq
    python scripts/list_models.py mistral --contains small
    python scripts/list_models.py openrouter --free --contains gemma
    python scripts/list_models.py gemini --contains flash

Providers: groq, mistral, openrouter, gemini, openai. Keys are read from paper_annotation/.env;
OpenRouter's list is public and needs no key.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ENDPOINTS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", None),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
}


def filter_ids(ids: list[str], contains: str = "", free: bool = False) -> list[str]:
    out = [i for i in ids if (not contains or contains.lower() in i.lower()) and (not free or i.endswith(":free"))]
    return sorted(set(out))


def fetch_ids(provider: str, environ=os.environ, getter=None) -> list[str]:
    """IDs from the provider's models endpoint. `getter(url, headers)` is injectable for tests."""
    if provider == "gemini":
        key = environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            raise SystemExit("GEMINI_API_KEY is not set in paper_annotation/.env")
        from google import genai
        client = genai.Client(api_key=key)
        return [m.name.removeprefix("models/") for m in client.models.list()]
    if provider not in ENDPOINTS:
        raise SystemExit(f"Unknown provider '{provider}'. Choose from: {', '.join([*ENDPOINTS, 'gemini'])}")
    base, var = ENDPOINTS[provider]
    headers = {}
    if var:
        key = environ.get(var, "").strip()
        if not key:
            raise SystemExit(f"{var} is not set in paper_annotation/.env")
        headers["Authorization"] = f"Bearer {key}"
    if getter is None:
        import requests
        def getter(url, h):
            r = requests.get(url, headers=h, timeout=60)
            r.raise_for_status()
            return r.json()
    data = getter(f"{base}/models", headers)
    return [m["id"] for m in data.get("data", [])]


def main(argv=None) -> int:
    from dotenv import load_dotenv

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("provider")
    ap.add_argument("--contains", default="", help="only IDs containing this text")
    ap.add_argument("--free", action="store_true", help="only free models (OpenRouter ':free' IDs)")
    a = ap.parse_args(argv)
    load_dotenv(ROOT / "paper_annotation" / ".env")
    try:
        ids = filter_ids(fetch_ids(a.provider), a.contains, a.free)
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, SystemExit):
            raise
        print(f"Could not list models: {type(exc).__name__}: {str(exc)[:200]}", file=sys.stderr)
        return 1
    print("\n".join(ids) if ids else "(no models matched)")
    print(f"\n{len(ids)} models", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
