"""Issue #12: cost estimator."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import estimate_cost as E  # noqa: E402
from llm.registry import CLAUDE_MODELS, spec  # noqa: E402


def test_estimate_arithmetic_by_hand():
    e = E.estimate(avg_prompt_chars=3600, total_papers=10, in_price=2.0, out_price=10.0, output_tokens=1000, passes=1.0)
    assert e["calls"] == 80
    assert e["input_tokens"] == pytest.approx(80 * 1000)            # 3600 chars / 3.6 = 1000 tokens
    assert e["usd"] == pytest.approx(80_000 / 1e6 * 2 + 80_000 / 1e6 * 10)


def test_cost_scales_linearly_and_orders_by_tier():
    a = E.estimate(5000, 20, 1, 5)["usd"]
    assert E.estimate(5000, 40, 1, 5)["usd"] == pytest.approx(2 * a)
    h, s, o = (spec(i) for i in ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"))
    cost = {m.id: E.estimate(20000, 217, m.in_price, m.out_price)["usd"] for m in (h, s, o)}
    assert cost["claude-haiku-5-5"] < cost["claude-sonnet-5-5"] < cost["claude-opus-5-5"]


def test_table_lists_every_registered_model():
    t = E.table(15000, 217, CLAUDE_MODELS, 800, 1.2)
    assert all(m.id in t for m in CLAUDE_MODELS) and "EST. USD" in t


def test_sample_prompt_chars_uses_real_pdfs(make_pdf):
    short = make_pdf("short", ["cooperative game for blind players " * 30])
    long_ = make_pdf("long", ["cooperative game for blind players " * 80] * 40)
    m = E.sample_prompt_chars([short, long_])
    assert m["papers"] == 2 and 0 < m["full_context_share"] <= 1
    assert m["avg_prompt_chars"] > 3000                              # framework context + question + chunks


def test_main_runs_end_to_end_and_errors_without_pdfs(make_pdf, tmp_path, capsys):
    d = tmp_path / "pdfs"
    d.mkdir()
    pdf = make_pdf("a", ["game text " * 200] * 3)
    (d / "a.pdf").write_bytes(pdf.read_bytes())
    assert E.main(["--papers-dir", str(d), "--sample", "5", "--total-papers", "10"]) == 0
    out = capsys.readouterr().out
    assert "claude-opus-5-5" in out and "ESTIMATE" in out
    assert E.main(["--papers-dir", str(tmp_path / "empty")]) == 2
