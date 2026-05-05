"""Text cleaning utilities."""
from __future__ import annotations

import re


def clean_text(text: str) -> str:
    """Normalize whitespace and remove hyphenation artifacts."""
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


if __name__ == "__main__":
    sample = "This is hyphen-\nated text.\n\nNew   line."
    print(clean_text(sample))
