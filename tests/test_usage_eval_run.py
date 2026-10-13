"""The wrapper that runs the mcp-builder runner with this repo's corrected agent loop."""

import json
import sys

from tests.evals.usage import run_eval

FAKE_RUNNER = """
import json, sys
EVALUATION_PROMPT = "the rules"

def agent_loop(*args, **kwargs):
    raise AssertionError("the runner's own loop was used")

async def main():
    with open(OUT, "w") as handle:
        json.dump(
            {
                "argv": sys.argv[1:],
                "loop": agent_loop.func.__name__,
                "system": agent_loop.keywords["system"],
            },
            handle,
        )
"""


def test_the_runner_is_started_with_the_corrected_loop_and_its_own_arguments(
    tmp_path, monkeypatch
):
    # Arrange
    out = tmp_path / "seen.json"
    runner = tmp_path / "evaluation.py"
    runner.write_text(f"OUT = {str(out)!r}\n" + FAKE_RUNNER)
    monkeypatch.setattr(sys, "argv", ["x"])
    monkeypatch.setattr(sys, "path", list(sys.path))

    # Act
    run_eval.main([str(runner), "-t", "stdio", "questions.xml"])

    # Assert
    seen = json.loads(out.read_text())
    assert seen == {
        "argv": ["-t", "stdio", "questions.xml"],
        "loop": "agent_loop",
        "system": "the rules",
    }
    assert str(tmp_path) in sys.path  # the runner imports its sibling modules
