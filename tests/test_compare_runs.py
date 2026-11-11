"""Issue #24: comparing v1 (forced choice) with v2 (may abstain)."""
import json

import numpy as np
import pandas as pd
import pytest

from evaluation import compare as C
from questions import PHT_FRAMEWORK_QUESTIONS as QS


def frame(rows, provider):
    out = []
    for pid, q, ans, mode in rows:
        out.append({"paper_id": pid, "provider": provider, "model_name": provider, "q": q, "answer": ans,
                    "answer_num": float(ans) if isinstance(ans, int) else np.nan, "confidence": 0.7,
                    "decision_mode": mode, "prompt_version": "t", "n_evidence": 1, "passes": 1})
    return pd.DataFrame(out)


def scenario():
    """20 cells. v1 always answers; v2 abstains on the 8 cells where v1 had to force a guess (and those guesses were poor)."""
    v1, v2, human = [], [], []
    for i in range(20):
        pid, q = f"p{i}", i % 8
        h = (i % 5) + 1
        human.append({"paper_id": pid, "q": q, "human": h})
        if i < 12:                                    # grounded answers: both runs agree and are mostly right
            ans = h if i % 4 else min(5, h + 1)
            v1.append((pid, q, ans, "retrieval"))
            v2.append((pid, q, ans, "retrieval"))
        else:                                         # hard cells: v1 guessed wrongly, v2 said Unknown
            v1.append((pid, q, (h % 5) + 1, "forced_choice"))
            v2.append((pid, q, "Unknown", "unresolved"))
    return frame(v1, "v1"), frame(v2, "v2"), pd.DataFrame(human)


def test_without_human_labels_reports_abstention_and_agreement():
    a, b, _ = scenario()
    r = C.compare_runs(a, b)
    assert r["shared_cells"] == 20 and r["answer_rate_a"] == 1.0 and r["answer_rate_b"] == pytest.approx(12 / 20)
    assert r["cells_b_abstained_a_answered"] == 8
    assert r["exact_agreement_where_both_answered"] == 1.0
    assert r["mode_counts_a"]["forced_choice"] == 8 and r["mode_counts_b"]["unresolved"] == 8
    assert "human_a" not in r


def test_with_human_labels_shows_the_avoided_guesses_were_wrong():
    a, b, h = scenario()
    r = C.compare_runs(a, b, h)
    assert r["avoided_guesses_n"] == 8 and r["avoided_guesses_accuracy"] == 0.0     # forced guesses were all wrong
    assert r["human_b"]["answer_rate"] < r["human_a"]["answer_rate"]
    assert r["human_b"]["accuracy"] > r["human_a"]["accuracy"]                      # abstaining raises accuracy on answered cells
    assert r["both_answered_accuracy_a"] == r["both_answered_accuracy_b"]            # identical on the cells both answered


def test_only_shared_cells_are_compared():
    a, b, _ = scenario()
    r = C.compare_runs(a, b.iloc[:10])
    assert r["shared_cells"] == 10
    with pytest.raises(ValueError):
        C.compare_runs(a, frame([("zzz", 0, 3, "retrieval")], "x"))


def test_report_text_mentions_the_key_numbers():
    a, b, h = scenario()
    txt = C.format_report(C.compare_runs(a, b, h), "v1", "v2")
    assert "8 guesses v2 avoided" in txt and "0.0%" in txt and "Answer rate: v1 100.0%, v2 60.0%" in txt


def _write(folder, rows):
    folder.mkdir(parents=True, exist_ok=True)
    by_paper = {}
    for pid, q, ans, mode in rows:
        by_paper.setdefault(pid, []).append({"paper_id": pid, "question": QS[q], "model_name": "m", "prompt_version": "t",
                                             "answer": ans, "confidence": 0.7, "evidence": ["chunk_1"], "decision_mode": mode})
    for pid, recs in by_paper.items():
        (folder / f"{pid}.jsonl").write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")


def test_cli_end_to_end_from_folders(tmp_path, capsys):
    a, b, h = scenario()
    _write(tmp_path / "res" / "run_v1", [(r.paper_id, r.q, int(r.answer_num), r.decision_mode) for r in a.itertuples()])
    _write(tmp_path / "res" / "run_v2", [(r.paper_id, r.q, r.answer, r.decision_mode) for r in b.itertuples()])
    h.to_csv(tmp_path / "h.csv", index=False)
    rc = C.main(["--a", str(tmp_path / "res" / "run_v1"), "--b", str(tmp_path / "res" / "run_v2"),
                 "--human", str(tmp_path / "h.csv"), "--out", str(tmp_path / "cmp.md")])
    assert rc == 0 and "guesses run_v2 avoided" in (tmp_path / "cmp.md").read_text(encoding="utf-8")


def test_load_run_errors_on_empty_folder(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit):
        C.load_run(tmp_path / "empty")


def test_cli_explains_when_runs_share_no_papers(tmp_path, capsys):
    _write(tmp_path / "res" / "a", [("p1", 0, 3, "retrieval")])
    _write(tmp_path / "res" / "b", [("p2", 0, 3, "retrieval")])
    assert C.main(["--a", str(tmp_path / "res" / "a"), "--b", str(tmp_path / "res" / "b")]) == 2
    assert "same papers" in capsys.readouterr().err
