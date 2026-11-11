import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_table2 as M  # noqa: E402


def _write(folder, paper, model, n=3, version="v1", mode="retrieval"):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"{paper}.jsonl").open("w", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(json.dumps({"paper_id": paper, "model_name": model, "prompt_version": version,
                                 "decision_mode": mode, "question": f"q{i}"}) + "\n")


def test_reports_the_models_that_really_ran(tmp_path):
    _write(tmp_path / "llama", "a", "meta-llama/Llama-3.3-70B-Instruct")
    _write(tmp_path / "llama", "b", "meta-llama/Llama-3.3-70B-Instruct")
    _write(tmp_path / "openai", "a", "gpt-5.1", version="v2", mode="forced_choice")
    rows = {r["folder"]: r for r in M.summarise(tmp_path)}
    assert rows["llama"]["papers"] == 2 and rows["llama"]["records"] == 6
    assert rows["llama"]["size_from_name"] == "70B"             # read from the id, not guessed
    assert rows["openai"]["size_from_name"] == "" and rows["openai"]["prompt_versions"] == {"v2": 3}


def test_flags_folders_that_mix_models(tmp_path):
    _write(tmp_path / "mistral", "a", "mistralai/Mistral-Small-3.1-24B-Instruct-2503", n=5)
    _write(tmp_path / "mistral", "b", "llama-3.1-8b-instant", n=2)
    md = M.to_markdown(M.summarise(tmp_path))
    assert "more than one model" in md and "llama-3.1-8b-instant" in md and "24B" in md


def test_unknown_size_and_date_are_left_for_a_human(tmp_path):
    _write(tmp_path / "x", "a", "gemini-2.5-flash")
    md = M.to_markdown(M.summarise(tmp_path))
    assert md.count("(look up)") == 2


def test_main_errors_on_empty_dir(tmp_path):
    assert M.main(["--results", str(tmp_path), "--out", str(tmp_path / "o.md")]) == 2
