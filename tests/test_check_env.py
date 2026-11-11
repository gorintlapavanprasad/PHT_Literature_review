"""Issue #13: check_env never prints keys, detects committed keys, and the repo itself is clean."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_env as E  # noqa: E402

FAKE = {
    "Anthropic": "sk-ant-api03-" + "A1b2C3d4E5" * 4,
    "OpenAI": "sk-proj-" + "Zy9Xw8Vu7T" * 4,
    "Google": "AIza" + "Sy1234567890abcdef" * 2,
    "Groq": "gsk_" + "abcdEFGH12" * 4,
    "HuggingFace": "hf_" + "qwertyUIOP" * 4,
}


def test_status_reports_booleans_only(monkeypatch, capsys):
    env = {"ANTHROPIC_API_KEY": FAKE["Anthropic"], "OPENAI_API_KEY": " ", "COMPAT_API_KEY": "x"}
    rows = {r["provider"]: r for r in E.provider_status(env)}
    assert rows["anthropic"]["ready"] and not rows["openai"]["ready"]
    assert not rows["compat"]["ready"] and rows["compat"]["vars"] == {"COMPAT_API_KEY": True, "COMPAT_BASE_URL": False}
    assert all(isinstance(v, bool) for r in rows.values() for v in r["vars"].values())   # no key text anywhere


def test_scan_text_recognises_each_key_shape_but_not_placeholders():
    for name, key in FAKE.items():
        assert name in E.scan_text(f"KEY={key}")
    for placeholder in ("OPENAI_API_KEY=sk-...", "GEMINI_API_KEY=...", "HF_TOKEN=hf_...", "ANTHROPIC_API_KEY=sk-ant-..."):
        assert E.scan_text(placeholder) == []


def test_scan_repo_finds_planted_key_and_never_returns_it(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "config.py").write_text(f'KEY = "{FAKE["OpenAI"]}"\n', encoding="utf-8")
    (tmp_path / "ok.md").write_text("no secrets here\n", encoding="utf-8")
    (tmp_path / "data.bin").write_bytes(b"\x00" + FAKE["Groq"].encode())            # non-text suffix: skipped
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    hits = E.scan_repo(tmp_path)
    assert hits == [("config.py", ["OpenAI"])]
    assert FAKE["OpenAI"] not in str(hits)


def test_this_repo_has_no_committed_keys():
    """Guards the real repo (old keys belonged to a previous developer's accounts)."""
    assert E.scan_repo(Path(__file__).resolve().parents[1]) == []


def test_dotenv_is_git_ignored():
    assert E.env_file_is_ignored(Path(__file__).resolve().parents[1])


def test_main_runs_and_prints_no_secrets(monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE["Anthropic"])
    assert E.main([]) == 0
    out = capsys.readouterr().out
    assert "READY   anthropic" in out and FAKE["Anthropic"] not in out and "A1b2C3d4E5" not in out
