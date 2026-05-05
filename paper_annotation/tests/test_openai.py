"""OpenAI API connectivity smoke test."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    
    api_key = os.getenv("OPENAI_API_KEY", "").strip().strip('"').strip("'")
    model_name = os.getenv("OPENAI_TEST_MODEL", "gpt-4.1").strip()
    
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        raise SystemExit(1)
    
    print(f"✓ API key found")
    print(f"Testing model: {model_name}")
    
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=10,
        )
        content = response.choices[0].message.content
        print(f"✓ Model responded: {content}")
        print("✅ OpenAI smoke test PASSED")
    except Exception as e:
        print(f"❌ Request failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
