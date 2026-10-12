"""The CI workflow and the audit script are build config that nothing else exercises.

A YAML slip in the workflow only shows up as a run that "failed in 0s" after pushing, so
parse it here, where `make check` runs it.
"""

import os
import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).parent.parent
GATE_PREFIXES = ("ruff format", "mypy", "pytest", "./scripts/pip_audit.sh")


def _steps():
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
    return workflow["jobs"]["build-and-test"]["steps"]


def test_workflow_parses_and_runs_the_four_gates_in_order():
    commands = [step["run"].split("\n")[0] for step in _steps() if "run" in step]

    gates = [c for c in commands if c.startswith(GATE_PREFIXES)]

    assert [g.split()[0] for g in gates] == [
        "ruff",
        "mypy",
        "pytest",
        "./scripts/pip_audit.sh",
    ]


def test_audit_script_is_executable():
    # CI runs it as ./scripts/pip_audit.sh, which needs the mode kept in git.
    assert os.access(ROOT / "scripts" / "pip_audit.sh", os.X_OK)


def test_ignore_file_lists_each_advisory_once_in_a_known_id_format():
    lines = (ROOT / ".pip-audit-ignore").read_text().splitlines()
    ids = [
        line.split()[0]
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]

    # Empty is the goal; the file only has to stay well formed when an entry is added.
    assert all(re.fullmatch(r"(PYSEC|GHSA|CVE)-[\w-]+", i) for i in ids)
    assert len(ids) == len(set(ids))


def test_ci_installs_from_the_lockfile_so_the_lock_is_what_gets_tested():
    install = "\n".join(step["run"] for step in _steps() if "run" in step)

    assert "uv sync --locked" in install
    assert "--extra dev" in install and "--extra engines" in install
    # A plain `pip install -e` resolves transitive dependencies fresh on every run.
    assert 'pip install -e ".[dev,engines]"' not in install


def test_dependabot_watches_the_lockfile_and_the_workflow_actions():
    config = yaml.safe_load((ROOT / ".github" / "dependabot.yml").read_text())

    ecosystems = {update["package-ecosystem"] for update in config["updates"]}

    assert config["version"] == 2
    assert ecosystems == {"uv", "github-actions"}
    assert all(update["directory"] == "/" for update in config["updates"])


def test_dependabot_only_refreshes_the_lockfile_and_leaves_declared_ranges_alone():
    # Its default raises the minimums in pyproject.toml to whatever is locked, which drops
    # support for versions that were verified on purpose (mcp 2.2.x) and upgrades nothing.
    config = yaml.safe_load((ROOT / ".github" / "dependabot.yml").read_text())

    uv_update = next(u for u in config["updates"] if u["package-ecosystem"] == "uv")

    assert uv_update["versioning-strategy"] == "lockfile-only"


def test_dependabot_skips_the_packages_it_cannot_resolve_or_must_not_bump():
    config = yaml.safe_load((ROOT / ".github" / "dependabot.yml").read_text())
    uv_update = next(u for u in config["updates"] if u["package-ecosystem"] == "uv")

    ignored = {entry["dependency-name"] for entry in uv_update["ignore"]}

    assert ignored == {"mlx", "mlx-lm", "ruff", "kev", "torch"}
