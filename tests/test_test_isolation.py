"""A test file must pass on its own, not only inside the full suite.

`jev_mcp.app` builds the shared provider when it is first imported. If that first import
happens inside a test that has already replaced `RoutingProvider` with a mock, the shared
provider becomes the mock and `compact` keeps every chunk. conftest imports `jev_mcp.app`
up front so the order no longer matters; this runs the file that used to fail, alone.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_the_triage_mcp_tests_pass_when_run_on_their_own():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_email_triage_mcp.py",
            "-q",
            "--no-cov",
            "-p",
            "no:cacheprovider",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout[-1500:]
