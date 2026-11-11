"""Generate the corrected paper Table 1 (PHT framework questions) from questions.py.

Table 1 in the manuscript repeats answer options (Fun-vs-Utility 4/5 and Play-vs-Game 4/5 are copies of
option 3) and has a "Solo vs. Solo" typo. The single source of truth is paper_annotation/questions.py,
which is also what the LLMs were actually asked - so the table is generated from it.

    python scripts/make_table1.py            # writes docs/table1.md and docs/table1.tex
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper_annotation"))

from questions import PHT_FRAMEWORK_QUESTIONS  # noqa: E402

TITLES = [
    "Fun vs. Utility", "Play vs. Game", "Skill vs. Chance", "Solo vs. Social",
    "Sequential vs. Simultaneous", "Synchronous vs. Asynchronous", "Competitive vs. Collaborative",
    "Symmetrical vs. Asymmetrical",
]
OPTION = re.compile(r"\((\d)\)\s*(.*?)\s*(?:,\s*)?(?=\(\d\)|$)", re.DOTALL)


def parse_options(question: str) -> list[tuple[int, str, str]]:
    """[(number, label, example)] from one question string."""
    body = question.split("apply to the design in the study:", 1)[1]
    out = []
    for num, text in OPTION.findall(body):
        text = " ".join(text.split()).rstrip(",")
        m = re.match(r"(.*?)\s*\(i\.e\.,?\s*(.*)\)\s*$", text)
        label, example = (m.group(1), m.group(2)) if m else (text, "")
        out.append((int(num), label.strip(), example.strip()))
    return out


def build() -> list[tuple[str, list[tuple[int, str, str]]]]:
    rows = []
    for title, q in zip(TITLES, PHT_FRAMEWORK_QUESTIONS):
        opts = parse_options(q)
        if [o[0] for o in opts] != [1, 2, 3, 4, 5]:
            raise ValueError(f"{title}: expected options 1-5, got {[o[0] for o in opts]}")
        if len({o[1] for o in opts}) != 5:
            raise ValueError(f"{title}: duplicated option text")
        rows.append((title, opts))
    return rows


def to_markdown(rows) -> str:
    lines = ["| Framework question | Answer options (with example) |", "|---|---|"]
    for title, opts in rows:
        cell = "<br>".join(f"({n}) {label}" + (f" (i.e., {ex})" if ex else "") for n, label, ex in opts)
        lines.append(f"| **{title}**<br>Which of the following best apply to the design in the study: | {cell} |")
    return "\n".join(lines) + "\n"


def to_latex(rows) -> str:
    esc = lambda t: t.replace("&", r"\&").replace("%", r"\%")
    out = [r"\begin{table*}[t]", r"\caption{The playful health technology (PHT) framework questions, with examples for each answer option.}",
           r"\label{tab:pht}", r"\small", r"\begin{tabular}{p{0.2\textwidth}p{0.72\textwidth}}", r"\toprule",
           r"Framework question & Answer options \\", r"\midrule"]
    for title, opts in rows:
        cell = r" \newline ".join(f"({n}) {esc(label)}" + (f" (i.e., {esc(ex)})" if ex else "") for n, label, ex in opts)
        out.append(f"{esc(title)} & {cell} \\\\ \\addlinespace")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(out) + "\n"


def main() -> None:
    rows = build()
    out = ROOT / "docs"
    out.mkdir(exist_ok=True)
    (out / "table1.md").write_text(to_markdown(rows), encoding="utf-8")
    (out / "table1.tex").write_text(to_latex(rows), encoding="utf-8")
    print("wrote", out / "table1.md", "and", out / "table1.tex")


if __name__ == "__main__":
    main()
