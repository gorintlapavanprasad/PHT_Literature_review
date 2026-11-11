"""Plan (and optionally run) a small pilot before a full run (issue #19).

Picks N papers, copies them into an isolated pilot folder so the real results are never touched, prints the cost
estimate and the exact commands. It spends money ONLY with --execute.

    python scripts/pilot.py --papers-dir ../TACCESS-Accountability/papers --n 20 --models claude-haiku-5-5
    python scripts/pilot.py --n 20 --models claude-haiku-5-5 gemini-3.6-flash --execute
    python scripts/pilot.py --n 20 --models claude-haiku-5-5 --execute --forced-choice      # v1-style ablation

Results go to <out>/results/<model folder>/ ; afterwards:
    cd paper_annotation && python -m evaluation.report --results <out>/results [--human consensus.csv]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper_annotation"))


def choose_papers(papers_dir: Path, n: int, seed: str = "pht-pilot") -> list[Path]:
    """Deterministic spread over the corpus (not just the alphabetical first N)."""
    pdfs = sorted(Path(papers_dir).glob("*.pdf"))
    pdfs.sort(key=lambda p: hashlib.sha1(f"{seed}|{p.stem}".encode()).hexdigest())
    return sorted(pdfs[:n])


def prepare(out: Path, pdfs: list[Path]) -> dict:
    """Create the isolated pilot workspace and copy the chosen PDFs into it."""
    dirs = {"papers": out / "papers", "indexes": out / "indexes", "results": out / "results"}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    for pdf in pdfs:
        shutil.copy2(pdf, dirs["papers"] / pdf.name)
    return dirs


def pilot_env(dirs: dict, forced_choice: bool = False, base=None) -> dict:
    env = dict(base if base is not None else os.environ)
    env.update(PDF_DIR=str(dirs["papers"]), INDEX_DIR=str(dirs["indexes"]), RESULTS_DIR=str(dirs["results"]))
    if forced_choice:
        env["FORCED_CHOICE"] = "1"
    return env


def commands(models: list[str]) -> list[list[str]]:
    return [[sys.executable, "main.py", "--stage", "all", "--model", m] for m in models]


def main(argv=None, runner=subprocess.run) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_dir = next((p for p in (ROOT / "papers", ROOT.parent / "TACCESS-Accountability" / "papers") if p.exists()), ROOT / "papers")
    ap.add_argument("--papers-dir", type=Path, default=default_dir)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--models", nargs="+", required=True, help="model ids or aliases")
    ap.add_argument("--out", type=Path, default=ROOT / "pilot")
    ap.add_argument("--forced-choice", action="store_true", help="enable the v1-style guess step (ablation)")
    ap.add_argument("--execute", action="store_true", help="actually call the models (costs money)")
    a = ap.parse_args(argv)

    pdfs = choose_papers(a.papers_dir, a.n)
    if not pdfs:
        print(f"No PDFs in {a.papers_dir}", file=sys.stderr)
        return 2

    import config as C
    import main as pipeline_main

    models = pipeline_main._resolve_models(a.models, C.all_models(), C.selectable_models())
    dirs = prepare(a.out, pdfs)
    print(f"Pilot: {len(pdfs)} papers copied to {dirs['papers']}")
    print("Models: " + ", ".join(models) + ("   [forced-choice ON]" if a.forced_choice else ""))
    print("\nCost estimate (list prices; see scripts/estimate_cost.py):")
    runner([sys.executable, str(ROOT / "scripts" / "estimate_cost.py"), "--papers-dir", str(dirs["papers"]),
            "--sample", str(a.n), "--total-papers", str(len(pdfs))], check=False)

    env = pilot_env(dirs, a.forced_choice)
    print("\nCommands (run from paper_annotation/):")
    for cmd in commands(models):
        print("  " + " ".join(cmd[1:]))
    if not a.execute:
        print("\nDry run only. Add --execute to spend money.")
        return 0
    rc = 0
    for cmd in commands(models):
        r = runner(cmd, cwd=ROOT / "paper_annotation", env=env, check=False)
        rc = rc or getattr(r, "returncode", 0)
    print(f"\nDone. Report: cd paper_annotation && python -m evaluation.report --results {dirs['results']}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
