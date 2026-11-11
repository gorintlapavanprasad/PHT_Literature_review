import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_paper_numbers as P  # noqa: E402


def _by_claim():
    return {r["claim"]: r for r in P.checks()}


def test_known_mismatches_are_detected():
    bad = {c for c, r in _by_claim().items() if not r["ok"]}
    assert "mean kappa, GPT-5" in bad                                  # text 0.1795, table gives 0.247
    assert "mean accuracy, Gemini Pro" in bad                          # text 0.136, table gives 0.155
    assert any(c.startswith("lowest kappa value") for c in bad)        # Mistral F7 (-0.105) is lower than Llama3
    assert "worst framework by mean kappa is F2" in bad                # F7 is lower on average


def test_statements_that_are_right_pass():
    ok = {c for c, r in _by_claim().items() if r["ok"]}
    for c in ("mean kappa, Llama3", "mean kappa, Gemini Pro", "mean kappa, Mistral", "mean accuracy, GPT-5",
              "mean accuracy, Llama3", "mean accuracy, Mistral", "average kappa for F4 (best framework)",
              "average accuracy for F1 (best framework)", "average accuracy for F5 (worst framework)"):
        assert c in ok, c
    assert all(r["ok"] for c, r in _by_claim().items() if c.startswith("printed Std. Dev."))


def test_gpt_mean_kappa_value():
    r = _by_claim()["mean kappa, GPT-5"]
    assert r["computed"] == 0.2468 and r["stated"] == 0.1795


def test_chance_notes_flag_below_chance_models():
    notes = " ".join(P.chance_notes())
    assert "Llama3: mean accuracy 0.126 is BELOW" in notes and "GPT-5: mean accuracy 0.449 is above" in notes
