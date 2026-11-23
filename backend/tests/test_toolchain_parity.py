from __future__ import annotations

import json
import tomllib
from pathlib import Path

import yaml

from services.simple_forecast import SUPPORTED_TICKERS

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_backend_manifest_copy_matches_root_and_preserves_286_universe() -> None:
    root_manifest = json.loads(
        (REPO_ROOT / "data" / "tri_exchange" / "manifest_broad_300.json").read_text(
            encoding="utf-8"
        )
    )
    backend_manifest_path = (
        REPO_ROOT / "backend" / "data" / "tri_exchange" / "manifest_broad_300.json"
    )
    assert backend_manifest_path.is_file()
    backend_manifest = json.loads(backend_manifest_path.read_text(encoding="utf-8"))

    assert backend_manifest["valid_tickers"] == root_manifest["valid_tickers"]
    assert len(SUPPORTED_TICKERS) == 286


def test_pyproject_prunes_unused_runtime_deps_and_keeps_pyarrow_in_dev() -> None:
    pyproject = tomllib.loads(
        (REPO_ROOT / "backend" / "pyproject.toml").read_text(encoding="utf-8")
    )
    prod_deps = [
        dep.split(">")[0].split("=")[0].split("[")[0].strip().lower()
        for dep in pyproject["project"]["dependencies"]
    ]
    dev_deps = [
        dep.split(">")[0].split("=")[0].split("[")[0].strip().lower()
        for dep in pyproject["dependency-groups"]["dev"]
    ]

    for removed in ("boto3", "cryptography", "filelock", "pyarrow"):
        assert removed not in prod_deps
    assert "pyarrow" in dev_deps


def test_uv_locked_and_dependabot_alignment() -> None:
    render_yaml = (REPO_ROOT / "render.yaml").read_text(encoding="utf-8")
    dockerfile = (REPO_ROOT / "backend" / "Dockerfile").read_text(encoding="utf-8")
    dependabot = (REPO_ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")

    assert "--locked" in render_yaml
    assert "--frozen" not in render_yaml
    assert "--locked" in dockerfile
    assert "--frozen" not in dockerfile
    assert 'package-ecosystem: "uv"' in dependabot


def test_ci_runs_on_stacked_prs_and_operations_workflow_uses_uv_locked() -> None:
    ci_workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )
    # PyYAML parses bare `on:` as boolean True.
    triggers = ci_workflow.get("on", ci_workflow.get(True, {}))
    assert "pull_request" in triggers
    pr_trigger = triggers["pull_request"]
    assert pr_trigger is None or "branches" not in pr_trigger

    ops_text = (REPO_ROOT / ".github" / "workflows" / "live-forecast-operations.yml").read_text(
        encoding="utf-8"
    )
    assert "uv sync --project backend --locked" in ops_text
    assert "backend/requirements.txt" not in ops_text
