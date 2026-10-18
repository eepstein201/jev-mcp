"""Latency harness for the email-triage webhook (POST /api/v1/triage/email).

    python -m tests.evals.webhook_latency --url http://127.0.0.1:8097 --dataset PATH \
        [--context-id ID] [--repeats N] [--attempt-header none|script] [--json]

The caller starts the webhook; the API key is read from JEV_MCP_API_KEY and is only
ever sent as `Authorization: Bearer <key>`, never printed or stored. See docs/perf/latency.md.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Optional

ENDPOINT_PATH = "/api/v1/triage/email"
ATTEMPT_HEADER = "X-Triage-Attempt"
# Mirror examples/gas_triage.js (MAX_DEGRADED_ATTEMPTS, RETRY_PAUSE_MS / 1000).
MAX_DEGRADED_ATTEMPTS = 3
RETRY_PAUSE_S = 5.0
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_CONTEXT_ID = "latency-eval"
API_KEY_ENV = "JEV_MCP_API_KEY"

FAST_VS_ESCALATED_NOTE = (
    "The webhook response does not expose whether a request stayed on the fast path or was "
    "escalated to the smart engine (no escalated flag survives triage_email_content). Only a "
    "failed escalation is visible, as degraded=true; a clean response may be either. The split "
    "below is therefore degraded (escalation attempted and failed) vs not degraded (fast-only "
    "or escalated successfully, indistinguishable)."
)

Record = dict[str, Any]


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """The q-th percentile (0-100) by linear interpolation between closest ranks.

    Method: sort ascending, rank = q/100 * (n - 1) (0-based), interpolate between the two
    neighbouring values (the "inclusive" method, numpy's default). None for no values.
    """
    if not values:
        return None
    ordered = sorted(values)
    rank = (q / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def summarize(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "max": max(values),
    }


def group_latencies(records: Sequence[Record], key: str) -> dict[str, list[float]]:
    """Latencies grouped by record[key]; records whose key is None are left out."""
    groups: dict[str, list[float]] = {}
    for rec in records:
        value = rec.get(key)
        if value is None:
            continue
        groups.setdefault(str(value), []).append(rec["latency_s"])
    return groups


def _share(records: Sequence[Record], predicate: Callable[[Record], bool]) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if predicate(r)) / len(records)


def build_report(records: Sequence[Record]) -> dict[str, Any]:
    latencies = [r["latency_s"] for r in records]
    degraded = [r["latency_s"] for r in records if r.get("degraded")]
    clean = [r["latency_s"] for r in records if not r.get("degraded")]
    return {
        "requests": len(records),
        "overall": summarize(latencies),
        "first_request_s": latencies[0] if latencies else None,
        "rest": summarize(latencies[1:]),
        "by_routing": {
            k: summarize(v)
            for k, v in sorted(group_latencies(records, "suggested_routing").items())
        },
        "fast_vs_escalated": {
            "note": FAST_VS_ESCALATED_NOTE,
            "degraded": summarize(degraded),
            "not_degraded": summarize(clean),
        },
        "degraded_share": _share(records, lambda r: bool(r.get("degraded"))),
        "http_503_share": _share(records, lambda r: r.get("http_status") == 503),
        "http_503_responses": sum(r.get("http_503_count", 0) for r in records),
        "error_share": _share(records, lambda r: r.get("error") is not None),
        "errors": dict(Counter(r["error"] for r in records if r.get("error"))),
    }


def load_dataset(path: Path) -> list[dict[str, str]]:
    """JSON-lines rows reduced to subject/sender/body; extra keys are ignored."""
    rows = []
    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: line {number} is not valid JSON: {exc}") from exc
        rows.append({k: str(row.get(k) or "") for k in ("subject", "sender", "body")})
    return rows


def _post(
    url: str,
    api_key: str,
    payload: dict[str, str],
    attempt: Optional[int],
    timeout: float,
    urlopen: Callable[..., Any],
) -> tuple[Optional[int], Optional[dict[str, Any]], Optional[str]]:
    """One request: (http_status, parsed JSON body or None, error text or None)."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if attempt is not None:
        headers[ATTEMPT_HEADER] = str(attempt)
    request = urllib.request.Request(
        url.rstrip("/") + ENDPOINT_PATH,
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as exc:
        status, raw = exc.code, exc.read()
    except (urllib.error.URLError, OSError) as exc:
        return None, None, f"{type(exc).__name__}: {exc}"
    try:
        body = json.loads(raw)
    except (ValueError, TypeError):
        return status, None, "non-JSON response body"
    return status, body if isinstance(body, dict) else None, None


def send_message(
    url: str,
    api_key: str,
    context_id: str,
    row: dict[str, str],
    attempt_mode: str = "none",
    timeout: float = DEFAULT_TIMEOUT_S,
    urlopen: Optional[Callable[..., Any]] = None,
    clock: Optional[Callable[[], float]] = None,
    sleep: Optional[Callable[[float], None]] = None,
) -> Record:
    """Send one message and time it. In "script" mode, mimic examples/gas_triage.js:
    attempts 1..3 with X-Triage-Attempt, a pause after each 503, pauses counted in the total."""
    urlopen = urlopen or urllib.request.urlopen
    clock = clock or time.perf_counter
    sleep = sleep or time.sleep
    payload = {"context_id": context_id, **row}
    max_attempts = MAX_DEGRADED_ATTEMPTS if attempt_mode == "script" else 1
    http_503_count = 0
    started = clock()
    attempts = 0
    status: Optional[int] = None
    body: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        header_attempt = attempt if attempt_mode == "script" else None
        status, body, error = _post(
            url, api_key, payload, header_attempt, timeout, urlopen
        )
        if status == 503:
            http_503_count += 1
        if status != 503 or attempt == max_attempts:
            break
        sleep(RETRY_PAUSE_S)
    latency = clock() - started
    body = body or {}
    return {
        "latency_s": latency,
        "http_status": status,
        "status": body.get("status"),
        "suggested_routing": body.get("suggested_routing"),
        "degraded": bool(body.get("degraded")),
        "error": error,
        "attempts": attempts,
        "http_503_count": http_503_count,
    }


def run_all(
    url: str,
    api_key: str,
    context_id: str,
    rows: Sequence[dict[str, str]],
    repeats: int = 1,
    attempt_mode: str = "none",
    timeout: float = DEFAULT_TIMEOUT_S,
    urlopen: Optional[Callable[..., Any]] = None,
    clock: Optional[Callable[[], float]] = None,
    sleep: Optional[Callable[[float], None]] = None,
) -> list[Record]:
    return [
        send_message(
            url,
            api_key,
            context_id,
            row,
            attempt_mode=attempt_mode,
            timeout=timeout,
            urlopen=urlopen,
            clock=clock,
            sleep=sleep,
        )
        for _ in range(repeats)
        for row in rows
    ]


def _fmt(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.3f}"


def _summary_line(label: str, s: dict[str, Any]) -> str:
    return (
        f"  {label:<28} n={s['count']:<4} mean={_fmt(s['mean'])} p50={_fmt(s['p50'])} "
        f"p90={_fmt(s['p90'])} p95={_fmt(s['p95'])} max={_fmt(s['max'])}"
    )


def format_text(report: dict[str, Any]) -> str:
    if report["requests"] == 0:
        return "0 requests measured."
    split = report["fast_vs_escalated"]
    lines = [
        f"{report['requests']} requests (seconds)",
        _summary_line("overall", report["overall"]),
        f"  first request (cold start): {_fmt(report['first_request_s'])}",
        _summary_line("rest (after first)", report["rest"]),
        "by suggested_routing",
        *[_summary_line(k, v) for k, v in report["by_routing"].items()],
        "fast vs escalated",
        f"  {split['note']}",
        _summary_line("degraded", split["degraded"]),
        _summary_line("not degraded", split["not_degraded"]),
        f"degraded share: {report['degraded_share']:.1%}",
        f"HTTP 503 share (final): {report['http_503_share']:.1%} "
        f"({report['http_503_responses']} 503 responses in total)",
        f"error share (connection/timeout/bad body): {report['error_share']:.1%} "
        f"{report['errors'] or ''}".rstrip(),
    ]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True, help="webhook base URL")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--context-id", default=DEFAULT_CONTEXT_ID)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--attempt-header", choices=("none", "script"), default="none")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    api_key = os.environ.get(API_KEY_ENV, "")
    if not api_key:
        print(
            f"warning: {API_KEY_ENV} is not set; requests go without Authorization.",
            file=sys.stderr,
        )
    records = run_all(
        args.url,
        api_key,
        args.context_id,
        load_dataset(args.dataset),
        repeats=args.repeats,
        attempt_mode=args.attempt_header,
        timeout=args.timeout,
    )
    report = build_report(records)
    print(json.dumps(report, indent=2) if args.as_json else format_text(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
