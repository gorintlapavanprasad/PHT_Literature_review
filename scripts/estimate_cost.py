"""Estimate tokens and dollars before a full run (issue #12).

Uses REAL prompt sizes: chunks N sample PDFs exactly like the pipeline does and builds the production prompt
(llm.prompts.build_prompt) for all 8 questions. Output size and extra passes are assumptions you can change.
It is an ESTIMATE (+-2x): thinking tokens, retries and tokenizer differences are not known in advance. Run a
20-paper pilot and compare with the provider's usage page before the big run.

    python scripts/estimate_cost.py --papers-dir ../TACCESS-Accountability/papers --sample 20 --total-papers 217
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper_annotation"))

CHARS_PER_TOKEN = 3.6          # English prose + PDF text; conservative (slightly over-counts tokens)
FULL_CONTEXT_CHUNKS = 30       # same rule as pipeline.annotate
TOP_K = 15


def sample_prompt_chars(pdfs: list[Path], chunk_size: int = 900, overlap: int = 150) -> dict:
    """Average production-prompt length (chars) over the sample, plus the share of full-context papers."""
    from ingest.chunker import chunk_text
    from ingest.pdf_loader import PageText, load_pdf_text
    from ingest.text_cleaner import clean_text
    from llm.prompts import build_prompt
    from questions import PHT_FRAMEWORK_QUESTIONS as QS

    sizes, full = [], 0
    for pdf in pdfs:
        pages = [PageText(p.page_number, clean_text(p.text)) for p in load_pdf_text(pdf)]
        chunks = chunk_text(pdf.stem, pages, chunk_size, overlap)
        if not chunks:
            continue
        use = chunks if len(chunks) <= FULL_CONTEXT_CHUNKS else chunks[:TOP_K]
        full += len(chunks) <= FULL_CONTEXT_CHUNKS
        sizes.extend(len(build_prompt(q, use, question_index=i)) for i, q in enumerate(QS))
    if not sizes:
        raise SystemExit("No extractable text in the sampled PDFs.")
    return {"avg_prompt_chars": sum(sizes) / len(sizes), "papers": len(pdfs), "full_context_share": full / max(len(pdfs), 1)}


def estimate(avg_prompt_chars: float, total_papers: int, in_price: float, out_price: float,
             output_tokens: int = 800, passes: float = 1.2, questions: int = 8) -> dict:
    calls = total_papers * questions * passes
    in_tok = calls * avg_prompt_chars / CHARS_PER_TOKEN
    out_tok = calls * output_tokens
    return {"calls": calls, "input_tokens": in_tok, "output_tokens": out_tok,
            "usd": in_tok / 1e6 * in_price + out_tok / 1e6 * out_price}


def table(avg_chars: float, total_papers: int, models, output_tokens: int, passes: float) -> str:
    rows = [f"{'MODEL':<22} {'$/Min':>6} {'$/Mout':>7} {'CALLS':>7} {'IN TOK (M)':>11} {'OUT TOK (M)':>12} {'EST. USD':>10}"]
    for m in models:
        e = estimate(avg_chars, total_papers, m.in_price, m.out_price, output_tokens, passes)
        rows.append(f"{m.id:<22} {m.in_price:>6.2f} {m.out_price:>7.2f} {e['calls']:>7.0f} "
                    f"{e['input_tokens'] / 1e6:>11.2f} {e['output_tokens'] / 1e6:>12.2f} {e['usd']:>10.2f}")
    return "\n".join(rows)


def main(argv=None) -> int:
    from llm.registry import CLAUDE_MODELS

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_dir = next((p for p in (ROOT / "papers", ROOT.parent / "TACCESS-Accountability" / "papers") if p.exists()), ROOT / "papers")
    ap.add_argument("--papers-dir", type=Path, default=default_dir)
    ap.add_argument("--sample", type=int, default=20, help="how many PDFs to measure")
    ap.add_argument("--total-papers", type=int, default=217, help="papers in the run you are pricing")
    ap.add_argument("--output-tokens", type=int, default=800, help="assumed output tokens/call incl. thinking")
    ap.add_argument("--passes", type=float, default=1.2, help="avg LLM calls per question (1 + retries)")
    a = ap.parse_args(argv)

    pdfs = sorted(a.papers_dir.glob("*.pdf"))[: a.sample]
    if not pdfs:
        print(f"No PDFs in {a.papers_dir}. Pass --papers-dir.", file=sys.stderr)
        return 2
    m = sample_prompt_chars(pdfs)
    print(f"Measured {m['papers']} PDFs: avg prompt {m['avg_prompt_chars']:.0f} chars "
          f"(~{m['avg_prompt_chars'] / CHARS_PER_TOKEN:.0f} tokens), {m['full_context_share']:.0%} fit in full context.")
    print(f"Assuming {a.output_tokens} output tokens/call and {a.passes} calls/question; pricing {a.total_papers} papers.\n")
    print(table(m["avg_prompt_chars"], a.total_papers, CLAUDE_MODELS, a.output_tokens, a.passes))
    print("\nESTIMATE only (+-2x). List prices cached 2026-11-03 - confirm on the provider's pricing page. "
          "Pilot 20 papers first.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
