"""Smoke-test every configured model with ONE tiny request (issue #11).

Catches wrong / retired model IDs and bad keys before you spend money on a full run.
Providers whose key is missing are SKIPPED (with the variable to set), never failed.

    python scripts/smoke_models.py --dry-run                       # list models + which keys are set (no API calls)
    python scripts/smoke_models.py --models claude-haiku-5-5       # one model
    python scripts/smoke_models.py --models claude-all             # every Claude version
    python scripts/smoke_models.py                                  # the default paper models

Exit code 0 = no FAIL (skips are fine), 1 = at least one FAIL.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper_annotation"))

PROMPT = ('Return exactly this JSON object and nothing else: '
          '{"answer": 3, "confidence": 0.9, "evidence": [], "reasoning": "smoke test"}')


def _sanitize(msg: str) -> str:
    """Never leak a key into logs: mask anything that looks like one."""
    msg = re.sub(r"(sk-[A-Za-z0-9_\-]{6})[A-Za-z0-9_\-]+", r"\1***", msg)
    msg = re.sub(r"(AIza[0-9A-Za-z_\-]{4})[0-9A-Za-z_\-]+", r"\1***", msg)
    msg = re.sub(r"(gsk_[A-Za-z0-9]{4})[A-Za-z0-9]+", r"\1***", msg)
    msg = re.sub(r"(hf_[A-Za-z0-9]{4})[A-Za-z0-9]+", r"\1***", msg)
    return msg[:300]


def missing_env(provider: str, environ) -> list[str]:
    import config as C
    return [k for k in C.PROVIDER_KEYS.get(provider, []) if not (environ.get(k) or "").strip()]


def run_smoke(models: list[str], environ, build_llm, provider_of, cfg=None) -> list[dict]:
    """Returns one result dict per model: status in {OK, FAIL, SKIP}."""
    results = []
    for name in models:
        provider = provider_of(name)
        miss = missing_env(provider, environ)
        if miss:
            results.append({"model": name, "provider": provider, "status": "SKIP", "detail": f"set {', '.join(miss)}"})
            continue
        t0 = time.perf_counter()
        try:
            llm = build_llm(name, cfg)
            out = json.loads(llm.generate(PROMPT))
            ok = out.get("answer") == 3
            results.append({"model": name, "provider": provider, "status": "OK" if ok else "FAIL",
                            "seconds": round(time.perf_counter() - t0, 2),
                            "detail": "" if ok else f"unexpected answer: {str(out)[:150]}"})
        except Exception as exc:  # noqa: BLE001 - report every failure kind
            results.append({"model": name, "provider": provider, "status": "FAIL",
                            "seconds": round(time.perf_counter() - t0, 2),
                            "detail": _sanitize(f"{type(exc).__name__}: {exc}")})
    return results


def format_table(results: list[dict]) -> str:
    rows = [f"{'STATUS':<6} {'MODEL':<42} {'SEC':>6}  DETAIL"]
    for r in results:
        rows.append(f"{r['status']:<6} {r['model']:<42} {r.get('seconds', ''):>6}  {r.get('detail', '')}")
    n = {s: sum(r["status"] == s for r in results) for s in ("OK", "FAIL", "SKIP")}
    rows.append(f"\n{n['OK']} ok, {n['FAIL']} failed, {n['SKIP']} skipped")
    return "\n".join(rows)


def main(argv=None) -> int:
    import os

    from dotenv import load_dotenv

    load_dotenv(ROOT / "paper_annotation" / ".env")
    import config as C
    import main as pipeline_main

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=None, help="model ids or aliases (claude, claude-all, openai, gemini, ...)")
    ap.add_argument("--dry-run", action="store_true", help="only show which models/keys are available")
    ap.add_argument("--json", type=Path, default=None, help="also write results to this JSON file")
    a = ap.parse_args(argv)

    models = pipeline_main._resolve_models(a.models, C.all_models(), C.selectable_models())
    if a.dry_run:
        for m in models:
            prov = C.model_provider(m)
            miss = missing_env(prov, os.environ)
            print(f"{'READY' if not miss else 'NEEDS ' + ','.join(miss):<28} {m}")
        return 0
    results = run_smoke(models, os.environ, pipeline_main._build_llm, C.model_provider, C.load_config())
    print(format_table(results))
    if a.json:
        a.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 1 if any(r["status"] == "FAIL" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
