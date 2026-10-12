"""Run the browser-agent cases several times each and report how often each one finishes.

    python -m tests.evals.browser_eval [--repeats N] [--cases D4 T3 ...]

It drives a real browser through `run_browser_agent` on the local engines, so both daemons must
be running. It measures; it does not gate, so it always exits 0. A run that could not be carried
out (engines down, no browser) counts as a failed run, is reported on stderr, and is counted in
a line after the table so that it is not mistaken for the agent failing.
"""

import argparse
import sys
import threading
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from jev_mcp.browser_agent import run_browser_agent
from tests.evals.rates import format_table, run_repeated

HOST = "127.0.0.1"
PORT = 8791
BASE = f"http://{HOST}:{PORT}"
PAGES_DIR = Path(__file__).resolve().parent / "pages"
DEFAULT_REPEATS = 4
MAX_STEPS = 20

# fmt: off
CASES = [
    ("D1", BASE + "/dense.html?case=d1",
     "Click the Select button for the 25mm ceramic bearing. The page shows MISSION_COMPLETE only when you click the right one."),
    ("D2", BASE + "/dense.html?case=d2",
     "Click the Details link for the only part that is both In Stock and costs less than $15. The page shows MISSION_COMPLETE only when you click the right one."),
    ("D3", BASE + "/dense.html?case=d3",
     "Click the Select button for the M6 left bracket. The page shows MISSION_COMPLETE only when you click the right one."),
    ("D4", BASE + "/dense.html?case=d4",
     "Click the Select button for the 49mm timing pulley. It may be below the visible area, so scroll down if needed. The page shows MISSION_COMPLETE only when you click the right one."),
    ("T1", BASE + "/typing.html?case=t1",
     "Type exactly blue orchid 42 into the Search field, then click Submit. The page shows MISSION_COMPLETE only for the exact value."),
    ("T2", BASE + "/typing.html?case=t2",
     "Type exactly MR-8837 into the Reference code field, then click Submit. The page shows MISSION_COMPLETE only for the exact value."),
    ("T3", BASE + "/typing.html?case=t3",
     "Type exactly sunset ferry 9 into the Search field and exactly QX-2210 into the Reference code field, then click Submit. The page shows MISSION_COMPLETE only for both exact values."),
    ("S1", BASE + "/select.html",
     "Select Lisboa (LIS-04) in the Destination city dropdown — exactly that option, not another Lisboa variant. The page shows MISSION_COMPLETE only for the right choice."),
]
# fmt: on


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        """Keep the per-request access log out of the eval's output."""


@contextmanager
def serve_pages() -> Iterator[None]:
    """Serve the eval pages on 127.0.0.1:8791 for the duration of the block."""
    handler = partial(_QuietHandler, directory=str(PAGES_DIR))
    server = ThreadingHTTPServer((HOST, PORT), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def run_case(name: str, url: str, goal: str, errored: list[str]) -> bool:
    """One run; True when the agent finished. A run that errored is reported and noted in `errored`."""
    try:
        outcome = run_browser_agent(url, goal, max_steps=MAX_STEPS)
    except Exception as e:
        message = f"{type(e).__name__}: {e}"
    else:
        if outcome.get("status") != "ERROR":
            return outcome.get("result") == "done"
        message = str(outcome.get("message", "no message"))
    print(f"{name}: run errored: {message}", file=sys.stderr)
    errored.append(name)
    return False


def main(argv: Sequence[str] | None = None) -> int:
    names = [name for name, _, _ in CASES]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--cases", nargs="+", choices=names, default=names)
    args = parser.parse_args(argv)

    errored: list[str] = []
    with serve_pages():
        rates = {
            name: run_repeated(
                partial(run_case, name, url, goal, errored), args.repeats
            )
            for name, url, goal in CASES
            if name in args.cases
        }
    print(format_table(rates))
    for name, count in Counter(errored).items():
        print(f"{name}: {count} of {rates[name].runs} runs errored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
