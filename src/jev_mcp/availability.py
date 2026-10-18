"""Engine availability sampler: probe both engines on an interval, log, summarize.

    python -m jev_mcp.availability sample [--interval S] [--log PATH] [--once]
    python -m jev_mcp.availability summary [--log PATH] [--since ISO|24h] [--json]

Each sample appends one JSON line per engine to an append-only log (see
docs/perf/availability.md). Probes are the doctor's (`doctor.check_engine`)."""

import argparse
import json
import logging
import math
import os
import re
import signal
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from jev_mcp import doctor

logger = logging.getLogger("jev_mcp.availability")

ENGINE_NAMES = ("fast", "smart")
DEFAULT_INTERVAL_SECONDS = 60.0
# The log is rotated to <path>.1 (one backup kept) once it exceeds this size.
MAX_LOG_BYTES = 4 * 1024 * 1024
MAX_ERROR_CHARS = 200
_DURATION_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
_DURATION = re.compile(r"^(\d+)([smhd])$")

CheckFn = Callable[..., Mapping[str, Any]]


def default_log_path() -> str:
    return os.path.join(os.path.expanduser("~"), ".jev", "availability.jsonl")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _short(text: str) -> str:
    return text[:MAX_ERROR_CHARS]


def _sample_engine(
    name: str, check: CheckFn, now: Callable[[], datetime], clock: Callable[[], float]
) -> dict[str, Any]:
    ts = now().isoformat()
    started = clock()
    try:
        result = check(name)
    except Exception as e:
        return _record(ts, name, False, False, None, f"{type(e).__name__}: {e}")
    elapsed_ms = (clock() - started) * 1000.0
    health_ok = result.get("status") == "up"
    probe = result.get("probe")
    if probe is None:
        return _record(
            ts, name, health_ok, False, None, f"health: {result.get('status')}"
        )
    probe_ok = bool(probe.get("ok"))
    error = None if probe_ok else str(probe.get("detail") or "probe failed")
    return _record(ts, name, health_ok, probe_ok, elapsed_ms, error)


def _record(
    ts: str,
    engine: str,
    health_ok: bool,
    probe_ok: bool,
    latency_ms: float | None,
    error: str | None,
) -> dict[str, Any]:
    return {
        "ts": ts,
        "engine": engine,
        "health_ok": health_ok,
        "probe_ok": probe_ok,
        "latency_ms": latency_ms,
        "error": None if error is None else _short(error),
    }


def sample_once(
    check: CheckFn,
    now: Callable[[], datetime] = _utc_now,
    clock: Callable[[], float] = time.monotonic,
) -> list[dict[str, Any]]:
    """One record per engine; an engine error becomes a failed record, never a raise."""
    return [_sample_engine(name, check, now, clock) for name in ENGINE_NAMES]


def append_rows(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    """Append rows as JSON lines, rotating to <path>.1 first if over the size cap."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        oversized = os.path.getsize(path) > MAX_LOG_BYTES
    except OSError:
        oversized = False
    if oversized:
        os.replace(path, path + ".1")
    with open(path, "a", encoding="utf-8") as f:
        f.write("".join(json.dumps(row) + "\n" for row in rows))


def install_stop_handlers(stop: threading.Event) -> Callable[[], None]:
    """Make SIGINT/SIGTERM set `stop`; returns a function restoring the old handlers."""
    signals = (signal.SIGINT, signal.SIGTERM)
    previous = {sig: signal.getsignal(sig) for sig in signals}

    def handler(signum: int, frame: Any) -> None:
        stop.set()

    for sig in signals:
        signal.signal(sig, handler)

    def restore() -> None:
        for sig, old in previous.items():
            signal.signal(sig, old)

    return restore


def run(
    path: str,
    interval: float,
    once: bool = False,
    stop: threading.Event | None = None,
    check: CheckFn | None = None,
    now: Callable[[], datetime] = _utc_now,
    wait: Callable[[float], bool] | None = None,
) -> int:
    """Sample until `stop` is set (or once); returns the number of samples taken."""
    stop = stop or threading.Event()
    wait = wait or stop.wait
    probe = check or doctor.check_engine
    taken = 0
    while not stop.is_set():
        rows = sample_once(probe, now)
        try:
            append_rows(path, rows)
        except OSError as e:
            logger.error("could not write %s: %s", path, e)
        taken += 1
        if once or wait(interval):
            break
    return taken


def parse_since(raw: str, now: datetime) -> datetime:
    """An ISO-8601 timestamp (naive means UTC) or a duration like 30m, 24h, 7d."""
    match = _DURATION.match(raw)
    if match:
        return now - timedelta(seconds=int(match[1]) * _DURATION_UNITS[match[2]])
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _parse_row(line: str) -> dict[str, Any] | None:
    try:
        row = json.loads(line)
        if not isinstance(row, dict) or row.get("engine") not in ENGINE_NAMES:
            return None
        when = datetime.fromisoformat(str(row["ts"]).replace("Z", "+00:00"))
    except (ValueError, KeyError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return {**row, "_when": when}


def read_rows(path: str) -> list[dict[str, Any]]:
    """Rows from the backup then the log; corrupt or partial lines are skipped."""
    rows: list[dict[str, Any]] = []
    for source in (path + ".1", path):
        try:
            with open(source, encoding="utf-8", errors="replace") as f:
                lines = f.read().splitlines()
        except OSError:
            continue
        rows.extend(r for r in map(_parse_row, lines) if r is not None)
    return rows


def percentile(values: Sequence[float], pct: float) -> float | None:
    """Nearest-rank percentile; None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100.0 * len(ordered)))
    return ordered[rank - 1]


def _is_failed(row: Mapping[str, Any]) -> bool:
    return not (row.get("health_ok") and row.get("probe_ok"))


def _outages(rows: Sequence[Mapping[str, Any]]) -> tuple[int, float]:
    """(count, longest seconds). A run of failed samples lasts from its first sample to
    the next good sample, or to its last failed sample when still open."""
    count = 0
    longest = 0.0
    start: datetime | None = None
    last: datetime | None = None
    for row in rows:
        when = row["_when"]
        if _is_failed(row):
            if start is None:
                start, count = when, count + 1
            last = when
            continue
        if start is not None:
            longest = max(longest, (when - start).total_seconds())
            start = None
    if start is not None and last is not None:
        longest = max(longest, (last - start).total_seconds())
    return count, longest


def _pct(part: int, whole: int) -> float:
    return round(part / whole * 100.0, 2)


def summarize(
    rows: Sequence[Mapping[str, Any]], since: datetime | None
) -> dict[str, dict[str, Any]]:
    """Per-engine stats over the window; engines with no samples are omitted."""
    result: dict[str, dict[str, Any]] = {}
    for name in ENGINE_NAMES:
        mine = sorted(
            (
                r
                for r in rows
                if r["engine"] == name and (since is None or r["_when"] >= since)
            ),
            key=lambda r: r["_when"],
        )
        if not mine:
            continue
        latencies = [
            float(r["latency_ms"])
            for r in mine
            if isinstance(r.get("latency_ms"), (int, float))
        ]
        outages, longest = _outages(mine)
        result[name] = {
            "samples": len(mine),
            "uptime_pct": _pct(sum(bool(r.get("health_ok")) for r in mine), len(mine)),
            "probe_success_pct": _pct(
                sum(bool(r.get("probe_ok")) for r in mine), len(mine)
            ),
            "latency_p50_ms": percentile(latencies, 50),
            "latency_p95_ms": percentile(latencies, 95),
            "outages": outages,
            "longest_outage_s": longest,
        }
    return result


def _fmt_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f} ms"


def render(summary: Mapping[str, Mapping[str, Any]]) -> str:
    if not summary:
        return "no samples in the window"
    lines = []
    for name, s in summary.items():
        lines.append(
            f"{name}: {s['samples']} samples, uptime {s['uptime_pct']:.2f}%, "
            f"probe success {s['probe_success_pct']:.2f}%, "
            f"p50 {_fmt_ms(s['latency_p50_ms'])}, p95 {_fmt_ms(s['latency_p95_ms'])}, "
            f"outages {s['outages']}, longest outage {s['longest_outage_s']:.0f} s"
        )
    return "\n".join(lines)


def _positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jev_mcp.availability", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sample = sub.add_parser("sample", help="probe both engines on an interval")
    sample.add_argument(
        "--interval", type=_positive_float, default=DEFAULT_INTERVAL_SECONDS
    )
    sample.add_argument("--log", default=None)
    sample.add_argument("--once", action="store_true", help="take one sample and exit")
    summary = sub.add_parser("summary", help="summarize the log")
    summary.add_argument("--log", default=None)
    summary.add_argument(
        "--since", default=None, help="ISO timestamp or duration (24h)"
    )
    summary.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str]) -> int:
    """The CLI entry point; the only function in this module that prints."""
    args = _build_parser().parse_args(argv)
    path = args.log or default_log_path()
    if args.command == "sample":
        stop = threading.Event()
        restore = install_stop_handlers(stop)
        try:
            run(path, args.interval, once=args.once, stop=stop)
        finally:
            restore()
        return 0
    since = None
    if args.since:
        try:
            since = parse_since(args.since, _utc_now())
        except ValueError:
            print(f"invalid --since: {args.since!r}", file=__import__("sys").stderr)
            return 2
    result = summarize(read_rows(path), since)
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else render(result))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
