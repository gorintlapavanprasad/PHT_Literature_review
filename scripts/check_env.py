"""Report which API keys are configured WITHOUT ever printing a key (issue #13).

    python scripts/check_env.py              # which providers are ready / missing
    python scripts/check_env.py --scan-repo  # also look for keys accidentally committed to the repo

Keys belong ONLY in paper_annotation/.env (git-ignored). See docs/API_KEYS.md.
Exit code: 0 ok, 1 if --scan-repo found something that looks like a committed key.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper_annotation"))

# Shapes of real keys (prefix + long body). Placeholders like "sk-..." or "your-key" do not match.
KEY_PATTERNS = {
    "Anthropic": re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    "OpenAI": re.compile(r"sk-(?:proj-)?[A-Za-z0-9_\-]{30,}"),
    "Google": re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
    "Groq": re.compile(r"gsk_[A-Za-z0-9]{30,}"),
    "HuggingFace": re.compile(r"hf_[A-Za-z0-9]{30,}"),
}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".env", ".example", ".yml", ".yaml", ".json", ".toml", ".cfg", ".ini", ".sh", ".ipynb"}


def provider_status(environ) -> list[dict]:
    """One row per provider: which required variables are set (booleans only - never values)."""
    import config as C

    rows = []
    for provider, names in C.PROVIDER_KEYS.items():
        have = {n: bool((environ.get(n) or "").strip()) for n in names}
        rows.append({"provider": provider, "ready": all(have.values()), "vars": have})
    return rows


def scan_text(text: str) -> list[str]:
    """Names of key types found in `text` (not the keys themselves)."""
    return [name for name, pat in KEY_PATTERNS.items() if pat.search(text)]


def scan_repo(root: Path) -> list[tuple[str, list[str]]]:
    """Check every git-tracked text file. Returns [(relative path, [key types])]; never returns key text."""
    try:
        files = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True).stdout.splitlines()
    except Exception:
        files = [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and ".git" not in p.parts and ".venv" not in p.parts]
    hits = []
    for rel in files:
        p = root / rel
        if p.suffix.lower() not in TEXT_SUFFIXES and p.name not in {".env", ".env.local"}:
            continue
        try:
            if p.stat().st_size > 2_000_000:
                continue
            found = scan_text(p.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
        if found:
            hits.append((rel, found))
    return hits


def env_file_is_ignored(root: Path) -> bool:
    r = subprocess.run(["git", "check-ignore", "-q", "paper_annotation/.env"], cwd=root)
    return r.returncode == 0


def main(argv=None) -> int:
    from dotenv import load_dotenv

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan-repo", action="store_true")
    a = ap.parse_args(argv)

    env_path = ROOT / "paper_annotation" / ".env"
    print(f".env file: {'found' if env_path.exists() else 'NOT FOUND'} ({env_path})")
    print(f".env is git-ignored: {'yes' if env_file_is_ignored(ROOT) else 'NO - fix .gitignore before adding keys'}")
    load_dotenv(env_path)
    print()
    for row in provider_status(os.environ):
        state = "READY  " if row["ready"] else "MISSING"
        detail = ", ".join(f"{n}={'set' if ok else 'NOT SET'}" for n, ok in row["vars"].items())
        print(f"{state} {row['provider']:<12} {detail}")
    rc = 0
    if a.scan_repo:
        hits = scan_repo(ROOT)
        print()
        if hits:
            rc = 1
            print("POSSIBLE KEYS COMMITTED (revoke them at the provider, then remove from history):")
            for rel, kinds in hits:
                print(f"  {rel}: {', '.join(kinds)}")
        else:
            print("Repo scan: no key-shaped strings found in tracked text files.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
