"""scripts/pip_audit.sh, run for real against a stand-in pip-audit.

The stand-in prints a canned JSON report when called with `-f json` (the stale-exemption
pass) and records its arguments otherwise (the gate run), so no network is needed.
"""

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "pip_audit.sh"

STUB = """#!/usr/bin/env bash
for a in "$@"; do
  if [[ $a == json ]]; then cat "$STUB_REPORT"; exit 1; fi
done
echo "$@" > "$STUB_GATE_ARGS"
exit 0
"""


def _report(*vulns: tuple[str, list[str]]) -> str:
    return json.dumps(
        {
            "dependencies": [
                {
                    "name": "pkg",
                    "version": "1.0",
                    "vulns": [{"id": i, "aliases": a} for i, a in vulns],
                }
            ],
            "fixes": [],
        }
    )


def _run(tmp_path: Path, ignore_lines: list[str], report: str):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "pip-audit"
    stub.write_text(STUB)
    stub.chmod(0o755)
    ignore_file = tmp_path / "ignore"
    ignore_file.write_text("\n".join(ignore_lines) + "\n")
    report_file = tmp_path / "report.json"
    report_file.write_text(report)
    gate_args = tmp_path / "gate_args"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "PIP_AUDIT_IGNORE_FILE": str(ignore_file),
        "STUB_REPORT": str(report_file),
        "STUB_GATE_ARGS": str(gate_args),
    }
    result = subprocess.run(
        [str(SCRIPT)], env=env, capture_output=True, text=True, timeout=30
    )
    gate = gate_args.read_text() if gate_args.exists() else None
    return result, gate, ignore_file


def test_fails_naming_each_exemption_that_is_no_longer_reported(tmp_path):
    result, gate, ignore_file = _run(
        tmp_path,
        ["# comment", "CVE-2026-1 still here", "CVE-2026-2 fixed upstream"],
        _report(("CVE-2026-1", [])),
    )

    assert result.returncode != 0
    assert (
        f"CVE-2026-2 is no longer reported; remove it from {ignore_file}"
        in result.stderr
    )
    assert "CVE-2026-1 is no longer reported" not in result.stderr
    assert gate is None


def test_passes_when_every_exemption_is_still_reported_by_id_or_alias(tmp_path):
    result, gate, _ = _run(
        tmp_path,
        ["CVE-2026-1", "CVE-2026-2"],
        _report(("CVE-2026-1", []), ("GHSA-aaaa-bbbb-cccc", ["CVE-2026-2"])),
    )

    assert result.returncode == 0, result.stderr
    assert "--ignore-vuln CVE-2026-1 --ignore-vuln CVE-2026-2" in gate


def test_fails_when_the_json_pass_returns_no_report(tmp_path):
    # A network error must not read as "every exemption is stale" or as a pass.
    result, gate, _ = _run(tmp_path, ["CVE-2026-1"], "")

    assert result.returncode != 0
    assert "did not return a JSON report" in result.stderr
    assert gate is None


def test_skips_the_stale_check_when_nothing_is_exempted(tmp_path):
    # An empty report would fail the JSON pass, so a pass here shows it never ran.
    result, gate, _ = _run(tmp_path, ["# only comments"], "")

    assert result.returncode == 0, result.stderr
    assert gate is not None and "--ignore-vuln" not in gate
