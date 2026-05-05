"""Gemini API connectivity smoke test."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    
    api_key = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
    model_name = os.getenv("GEMINI_TEST_MODEL", "gemini-2.5-flash").strip()
    
    if not api_key:
        print("❌ GEMINI_API_KEY not set")
        raise SystemExit(1)
    
    print(f"✓ API key found")
    print(f"Testing model: {model_name}")
    
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents="Say OK",
        )
        content = response.text
        print(f"✓ Model responded: {content[:100]}")
        print("✅ Gemini smoke test PASSED")
    except Exception as e:
        print(f"❌ Request failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
