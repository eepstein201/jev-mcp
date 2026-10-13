"""Run the mcp-builder evaluation runner with this repo's corrected agent loop.

    python tests/evals/usage/run_eval.py <path to the skill's scripts/evaluation.py> <runner args...>

Needs a Python with `anthropic` and an `mcp` 1.x client (the runner is written for 1.x); see README.md.
"""

import asyncio
import functools
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tests.evals.usage.loop import agent_loop  # noqa: E402


def main(argv: list[str]) -> None:
    runner = Path(argv[0]).resolve()
    sys.path.insert(
        0, str(runner.parent)
    )  # the runner imports its sibling `connections`
    spec = importlib.util.spec_from_file_location("mcp_builder_evaluation", runner)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.agent_loop = functools.partial(agent_loop, system=module.EVALUATION_PROMPT)
    sys.argv = [str(runner), *argv[1:]]
    asyncio.run(module.main())


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
