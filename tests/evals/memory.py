"""Record real memory use of the local engines, so model pairs can be compared on it.

    python -m tests.evals.memory [--json]            # daemons must be running

Three signals, none of which is `ps -o rss` (which misses Metal-mapped weights):

* startup buffers: `model buffer`, `KV buffer` and `compute buffer` sizes (MiB)
  from the LAST startup block of each llama-server log;
* resident footprint per daemon from macOS `footprint -p <pid>`;
* system pressure: `sysctl vm.swapusage` and `vm_stat` counters. These are a
  snapshot; callers take one before and one after a run and `delta()` them.

Every external command goes through `_run`, so tests mock `subprocess.check_output`.
"""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

JEV_DIR = Path.home() / ".jev"
DEFAULT_FAST_LOG = JEV_DIR / "mlx_server_fast.log"
DEFAULT_SMART_LOG = JEV_DIR / "mlx_server_smart.log"
DEFAULT_FAST_PORT = "8080"
DEFAULT_SMART_PORT = "8081"
COMMAND_TIMEOUT_SECONDS = 15

# A new server start begins a new block; only the last one describes the live process.
STARTUP_MARKERS = ("llama_server: initializing", "main: loading model")
BUFFER_PATTERN = re.compile(
    r"\b(?P<kind>model|KV|compute) buffer size\s*=\s*(?P<mib>[\d.]+)\s*MiB"
)
BUFFER_FIELDS = {"model": "model_mib", "KV": "kv_mib", "compute": "compute_mib"}

FOOTPRINT_PATTERN = re.compile(r"Footprint:\s*([\d.]+)\s*(B|KB|MB|GB|TB)\b")
PEAK_PATTERN = re.compile(r"phys_footprint_peak:\s*(\d+)\s*B\b")
UNIT_BYTES = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}

SWAP_PATTERN = re.compile(r"\b(total|used|free)\s*=\s*([\d.]+)M")
VM_PAGE_SIZE_PATTERN = re.compile(r"page size of (\d+) bytes")
VM_COUNTERS = {
    "Pageins": "pageins",
    "Pageouts": "pageouts",
    "Swapins": "swapins",
    "Swapouts": "swapouts",
}
VM_COUNTER_PATTERN = re.compile(
    r"^(Pageins|Pageouts|Swapins|Swapouts):\s+(\d+)\.", re.M
)

ROLES = ("fast", "smart")
MIB = 1024**2


def _run(cmd):
    """The single subprocess boundary. Returns stdout, or None if the command failed."""
    try:
        return subprocess.check_output(
            cmd,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except (subprocess.SubprocessError, OSError):
        return None


# --- startup log ------------------------------------------------------------


def _empty_buffers():
    return {field: None for field in BUFFER_FIELDS.values()}


def parse_startup_buffers(lines):
    """Sum each buffer kind (across devices) in the last startup block of `lines`."""
    totals = _empty_buffers()
    for line in lines:
        if any(marker in line for marker in STARTUP_MARKERS):
            totals = _empty_buffers()
            continue
        match = BUFFER_PATTERN.search(line)
        if match:
            field = BUFFER_FIELDS[match["kind"]]
            totals[field] = (totals[field] or 0.0) + float(match["mib"])
    return totals


def read_startup_buffers(path):
    path = Path(path)
    try:
        with open(path, "r", errors="replace") as f:
            buffers = parse_startup_buffers(f)
    except OSError:
        return {**_empty_buffers(), "log": str(path), "error": "log not found"}
    return {**buffers, "log": str(path)}


# --- resident footprint -----------------------------------------------------


def find_pids(port):
    """PIDs of llama-server processes listening on `port` (exact port, not a prefix)."""
    out = _run(["pgrep", "-f", rf"llama-server.*--port {port}( |$)"])
    return [int(tok) for tok in (out or "").split() if tok.isdigit()]


def parse_footprint(text):
    match = FOOTPRINT_PATTERN.search(text or "")
    if not match:
        return None
    peak = PEAK_PATTERN.search(text)
    return {
        "footprint_bytes": int(float(match[1]) * UNIT_BYTES[match[2]]),
        "peak_bytes": int(peak[1]) if peak else None,
    }


def process_footprint(pid):
    return parse_footprint(
        _run(["footprint", "-p", str(pid), "-f", "bytes", "--noCategories"])
    )


def resident_footprint(port):
    pids = find_pids(port)
    found = [fp for fp in map(process_footprint, pids) if fp is not None]
    return {
        "port": int(port),
        "pids": pids,
        "footprint_bytes": sum(fp["footprint_bytes"] for fp in found) or None,
        "peak_bytes": sum(fp["peak_bytes"] or 0 for fp in found) or None,
    }


# --- system pressure --------------------------------------------------------


def parse_swapusage(text):
    values = {key: float(num) for key, num in SWAP_PATTERN.findall(text or "")}
    if set(values) != {"total", "used", "free"}:
        return None
    return {f"{key}_mib": values[key] for key in ("total", "used", "free")}


def parse_vm_stat(text):
    text = text or ""
    page_size = VM_PAGE_SIZE_PATTERN.search(text)
    counters = {
        VM_COUNTERS[name]: int(count)
        for name, count in VM_COUNTER_PATTERN.findall(text)
    }
    if not page_size or set(counters) != set(VM_COUNTERS.values()):
        return None
    return {"page_size": int(page_size[1]), **counters}


def system_pressure():
    return {
        "swap": parse_swapusage(_run(["sysctl", "vm.swapusage"])),
        "vm": parse_vm_stat(_run(["vm_stat"])),
    }


# --- snapshot and delta -----------------------------------------------------


def snapshot(fast_log, smart_log, fast_port, smart_port):
    logs = {"fast": fast_log, "smart": smart_log}
    ports = {"fast": fast_port, "smart": smart_port}
    return {
        "startup": {
            role: read_startup_buffers(logs[role]) if logs[role] else _empty_buffers()
            for role in ROLES
        },
        "resident": {role: resident_footprint(ports[role]) for role in ROLES},
        "system": system_pressure(),
    }


def _diff(before, after):
    if before is None or after is None:
        return None
    return after - before


def _get(snap, *path):
    for key in path:
        if not isinstance(snap, dict) or snap.get(key) is None:
            return None
        snap = snap[key]
    return snap


def delta(before, after):
    """What changed between two snapshots (after - before); None where a side is missing."""
    out = {
        "swap_used_mib": _diff(
            _get(before, "system", "swap", "used_mib"),
            _get(after, "system", "swap", "used_mib"),
        )
    }
    for counter in VM_COUNTERS.values():
        out[counter] = _diff(
            _get(before, "system", "vm", counter), _get(after, "system", "vm", counter)
        )
    out["footprint_bytes"] = {
        role: _diff(
            _get(before, "resident", role, "footprint_bytes"),
            _get(after, "resident", role, "footprint_bytes"),
        )
        for role in ROLES
    }
    return out


# --- CLI --------------------------------------------------------------------


def _mib(value, divisor=1):
    return "n/a" if value is None else f"{value / divisor:,.1f} MiB"


def format_snapshot(snap):
    lines = []
    for role in ROLES:
        buf = snap["startup"][role]
        res = snap["resident"][role]
        lines.append(
            f"{role} (port {res.get('port', 'n/a')}, pids {res['pids'] or 'n/a'})"
        )
        lines.append(
            f"  startup buffers: model {_mib(buf['model_mib'])}, "
            f"KV {_mib(buf['kv_mib'])}, compute {_mib(buf['compute_mib'])}"
        )
        lines.append(
            f"  resident footprint: {_mib(res['footprint_bytes'], MIB)}"
            f" (peak {_mib(res['peak_bytes'], MIB)})"
        )
    swap = snap["system"]["swap"]
    vm = snap["system"]["vm"]
    lines.append(
        f"system: swap used {_mib(swap and swap['used_mib'])}"
        f" of {_mib(swap and swap['total_mib'])}"
    )
    lines.append(
        "  pageouts {}, swapouts {}".format(
            "n/a" if vm is None else f"{vm['pageouts']:,}",
            "n/a" if vm is None else f"{vm['swapouts']:,}",
        )
    )
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Record real memory use of the engines."
    )
    parser.add_argument("--json", action="store_true", help="print one JSON snapshot")
    parser.add_argument("--fast-log", default=str(DEFAULT_FAST_LOG))
    parser.add_argument("--smart-log", default=str(DEFAULT_SMART_LOG))
    parser.add_argument(
        "--fast-port", default=os.getenv("JEV_FAST_PORT", DEFAULT_FAST_PORT)
    )
    parser.add_argument(
        "--smart-port", default=os.getenv("JEV_SMART_PORT", DEFAULT_SMART_PORT)
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    snap = snapshot(args.fast_log, args.smart_log, args.fast_port, args.smart_port)
    print(json.dumps(snap, indent=2) if args.json else format_snapshot(snap))


if __name__ == "__main__":
    main()
