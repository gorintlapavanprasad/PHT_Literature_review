"""Issue #29: Docker has never been built here (not installed), so check statically what can go wrong without it."""
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()


def _instructions():
    out, cur = [], ""
    for line in DOCKERFILE:
        if line.strip().startswith("#") or not line.strip():
            continue
        cur += line.rstrip("\\").strip() + " "
        if not line.rstrip().endswith("\\"):
            out.append(cur.strip())
            cur = ""
    return out


def test_every_copy_source_exists():
    for ins in _instructions():
        if ins.startswith("COPY "):
            src = ins.split()[1]
            assert (ROOT / src).exists(), f"Dockerfile copies {src} but it does not exist"


def test_requirements_files_used_in_pip_install_exist():
    for ins in _instructions():
        for req in re.findall(r"-r\s+(\S+)", ins):
            assert (ROOT / req).exists(), req


def test_starts_the_web_app_on_the_configured_port_and_has_a_healthcheck():
    text = "\n".join(DOCKERFILE)
    assert "backend.server:app" in text and "--host 0.0.0.0" in text and "${PORT}" in text
    assert "HEALTHCHECK" in text and "/api/health" in text
    assert (ROOT / "pht_ui" / "backend" / "server.py").exists()


def test_image_is_python_312_and_installs_cpu_torch_first():
    text = "\n".join(DOCKERFILE)
    assert re.search(r"FROM python:3\.12", text) and "whl/cpu" in text


def test_secrets_are_not_baked_into_the_image():
    ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert ".env" in ignore and ".git" in ignore and ".venv" in ignore
    assert not any(i.startswith("ENV") and re.search(r"KEY|TOKEN|SECRET", i) for i in _instructions())


def test_compose_file_is_valid_and_mounts_existing_host_paths():
    c = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    svc = c["services"]["pht"]
    assert svc["build"] == "." and "8765:8765" in svc["ports"]
    assert svc["env_file"] == ["paper_annotation/.env"]
    for vol in svc["volumes"]:
        host = vol.split(":")[0]
        assert host.startswith("./")


def test_server_entrypoint_imports_without_extra_files():
    """The container runs `uvicorn backend.server:app` from pht_ui/ with PIPELINE_ROOT=/app/paper_annotation."""
    import sys
    sys.path.insert(0, str(ROOT / "pht_ui"))
    from backend import server
    assert hasattr(server, "app") and (server.PIPELINE_ROOT / "main.py").exists()
