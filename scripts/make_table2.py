"""Build the "models actually used" table from the results on disk (issue #27).

The manuscript's Table 2 lists GPT-5, Gemini Pro, Llama-3 405B and Mistral Large, but the stored results were
produced by other models. This reads every results folder and reports what really ran: model id, number of papers
and records, prompt version(s), and how often each decision mode was used. Parameter size and release date are only
filled in when the model id states them (e.g. "70B"); otherwise they are left blank for you to look up - they are
never guessed.

    python scripts/make_table2.py --results paper_annotation/data/results
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIZE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])")


def summarise(results_dir: Path) -> list[dict]:
    rows = []
    for folder in sorted(p for p in Path(results_dir).iterdir() if p.is_dir()):
        models, versions, modes, papers, n = Counter(), Counter(), Counter(), set(), 0
        for f in folder.glob("*.jsonl"):
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n += 1
                papers.add(r.get("paper_id") or f.stem)
                models[r.get("model_name") or "?"] += 1
                versions[r.get("prompt_version") or "?"] += 1
                modes[r.get("decision_mode") or "?"] += 1
        if not n:
            continue
        top = models.most_common()
        size = SIZE.search(top[0][0])
        rows.append({
            "folder": folder.name, "model": top[0][0], "other_models": [m for m, _ in top[1:]],
            "papers": len(papers), "records": n,
            "prompt_versions": dict(versions), "decision_modes": dict(modes),
            "size_from_name": f"{size.group(1)}B" if size else "",
        })
    return rows


def to_markdown(rows: list[dict]) -> str:
    lines = ["# Models that actually produced the results", "",
             "| Results folder | Model id | Papers | Records | Prompt version | Size (from id) | Release date |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        v = ", ".join(f"{k} ({c})" for k, c in r["prompt_versions"].items())
        lines.append(f"| {r['folder']} | {r['model']} | {r['papers']} | {r['records']} | {v} | {r['size_from_name'] or '(look up)'} | (look up) |")
    mixed = [r for r in rows if r["other_models"]]
    if mixed:
        lines += ["", "**Folders that contain more than one model** (should not be mixed):"]
        lines += [f"- `{r['folder']}`: also {', '.join(r['other_models'])}" for r in mixed]
    lines += ["", "Decision modes per folder:", ""]
    for r in rows:
        lines.append(f"- {r['folder']}: " + ", ".join(f"{k} {c}" for k, c in sorted(r["decision_modes"].items())))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=ROOT / "paper_annotation" / "data" / "results")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "table2_models_used.md")
    a = ap.parse_args(argv)
    rows = summarise(a.results)
    if not rows:
        print(f"No results found in {a.results}", file=sys.stderr)
        return 2
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(to_markdown(rows), encoding="utf-8")
    print(to_markdown(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
