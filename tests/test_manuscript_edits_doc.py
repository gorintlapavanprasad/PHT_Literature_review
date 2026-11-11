"""The numbers quoted in docs/MANUSCRIPT_EDITS.md must agree with what the checker computes."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_paper_numbers as P  # noqa: E402

DOC = (ROOT / "docs" / "MANUSCRIPT_EDITS.md").read_text(encoding="utf-8")


def test_quoted_corrections_match_the_tables():
    assert "0.247" in DOC and round(float(np.mean(P.KAPPA["GPT-5"])), 3) == 0.247
    assert "0.155" in DOC and round(float(np.mean(P.ACC["Gemini Pro"])), 3) == 0.155
    assert "-0.105" in DOC and min(v for m in P.MODELS for v in P.KAPPA[m]) == -0.105
    f7 = np.mean([P.KAPPA[m][6] for m in P.MODELS])
    f2 = np.mean([P.KAPPA[m][1] for m in P.MODELS])
    assert f7 < f2 and abs(f7) < 0.001 and round(float(f2), 3) == 0.016


def test_model_table_matches_the_stored_results_if_present():
    import make_table2 as M
    res = ROOT / "paper_annotation" / "data" / "results"
    if not res.exists():
        return
    rows = {r["folder"]: r for r in M.summarise(res)}
    assert rows["openai"]["model"] == "gpt-5.1" and rows["gemini"]["model"] == "gemini-2.5-flash"
    assert rows["llama"]["model"] == "meta-llama/Llama-3.3-70B-Instruct" and rows["mistral"]["size_from_name"] == "24B"
    for folder in ("openai", "gemini", "llama", "mistral"):
        assert rows[folder]["model"] in DOC
