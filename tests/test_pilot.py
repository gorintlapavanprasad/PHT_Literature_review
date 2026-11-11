"""Issue #19: pilot planning is deterministic, isolated from real results, and never spends without --execute."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pilot as P  # noqa: E402

import config  # noqa: E402


@pytest.fixture
def pdf_dir(tmp_path, make_pdf):
    d = tmp_path / "corpus"
    d.mkdir()
    src = make_pdf("src", ["cooperative game text " * 60] * 2)
    for i in range(30):
        (d / f"paper{i:02d}.pdf").write_bytes(src.read_bytes())
    return d


def test_choice_is_deterministic_sized_and_not_just_alphabetical(pdf_dir):
    a, b = P.choose_papers(pdf_dir, 10), P.choose_papers(pdf_dir, 10)
    assert a == b and len(a) == 10 and len({p.name for p in a}) == 10
    assert [p.name for p in a] != [f"paper{i:02d}.pdf" for i in range(10)]
    assert len(P.choose_papers(pdf_dir, 500)) == 30


def test_prepare_copies_into_isolated_folders(pdf_dir, tmp_path):
    dirs = P.prepare(tmp_path / "pilot", P.choose_papers(pdf_dir, 4))
    assert len(list(dirs["papers"].glob("*.pdf"))) == 4
    assert dirs["results"].exists() and dirs["indexes"].exists()
    assert str(dirs["results"]).startswith(str(tmp_path / "pilot"))


def test_env_points_the_pipeline_at_the_pilot_folders_only(tmp_path):
    dirs = {"papers": tmp_path / "p", "indexes": tmp_path / "i", "results": tmp_path / "r"}
    env = P.pilot_env(dirs, forced_choice=True, base={"KEEP": "1"})
    assert env["KEEP"] == "1" and env["FORCED_CHOICE"] == "1"
    assert env["RESULTS_DIR"] == str(tmp_path / "r") and "FORCED_CHOICE" not in P.pilot_env(dirs, base={})


def test_config_honours_the_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("PDF_DIR", str(tmp_path / "p"))
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "r"))
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "i"))
    c = config.load_config()
    assert (c.pdf_dir, c.results_dir, c.index_dir) == (tmp_path / "p", tmp_path / "r", tmp_path / "i")


def test_dry_run_never_runs_the_pipeline(pdf_dir, tmp_path, capsys):
    calls = []
    rc = P.main(["--papers-dir", str(pdf_dir), "--n", "5", "--models", "claude-haiku-5-5", "--out", str(tmp_path / "o")],
                runner=lambda cmd, **kw: calls.append(cmd) or SimpleNamespace(returncode=0))
    out = capsys.readouterr().out
    assert rc == 0 and "Dry run only" in out and "claude-haiku-5-5" in out
    assert all("main.py" not in " ".join(map(str, c)) for c in calls)         # only the cost estimate ran


def test_execute_runs_one_command_per_model_with_pilot_env(pdf_dir, tmp_path):
    seen = []
    rc = P.main(["--papers-dir", str(pdf_dir), "--n", "3", "--models", "claude-haiku-5-5", "claude-sonnet-5-5",
                 "--out", str(tmp_path / "o"), "--execute", "--forced-choice"],
                runner=lambda cmd, **kw: seen.append((cmd, kw)) or SimpleNamespace(returncode=0))
    runs = [(c, kw) for c, kw in seen if "main.py" in c]
    assert rc == 0 and [c[-1] for c, _ in runs] == ["claude-haiku-5-5", "claude-sonnet-5-5"]
    assert all(kw["env"]["RESULTS_DIR"].startswith(str(tmp_path / "o")) and kw["env"]["FORCED_CHOICE"] == "1" for _, kw in runs)


def test_no_pdfs_is_an_error(tmp_path):
    assert P.main(["--papers-dir", str(tmp_path / "none"), "--models", "claude-haiku-5-5"]) == 2


def test_failure_of_a_model_run_is_reported(pdf_dir, tmp_path):
    rc = P.main(["--papers-dir", str(pdf_dir), "--n", "2", "--models", "claude-haiku-5-5", "--out", str(tmp_path / "o"),
                 "--execute"], runner=lambda cmd, **kw: SimpleNamespace(returncode=1 if "main.py" in cmd else 0))
    assert rc == 1
