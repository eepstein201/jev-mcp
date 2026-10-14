"""Record real memory use of the local engines, so model pairs can be compared on it.

    python -m tests.evals.memory [--json] [--warm-request]   # daemons must be running
    python -m tests.evals.memory --delta before.json after.json

What each signal is, and is not:

* startup buffers: `model buffer`, `KV buffer` and `compute buffer` sizes (MiB) from
  the LAST startup block of each llama-server log. Null (with a reason) when the
  build/verbosity does not print them.
* `dirty_footprint_bytes`: macOS `footprint -p <pid>` Footprint, i.e. dirty plus
  swapped/compressed pages. It does NOT include clean file-backed pages, so mmap'd
  GGUF weights and Metal buffers over them can be missing or only partly counted
  (observed: ~1.9 GiB for a 4.4 GiB GGUF). Read it with the `categories` rows
  (`mapped file`, `IOAccelerator`, `IOSurface`; dirty/clean bytes) and with
  `gguf_bytes`, the model file's size on disk, as the upper bound of the weights.
* system: `sysctl vm.swapusage`, `vm_stat` counters and wired memory, and the
  `ioreg` GPU "In use system memory" figure. Counters are a snapshot; take one
  before and one after a run and `delta()` them (`--delta` on two saved `--json`
  snapshots).

A default run reads IDLE footprints: the daemons have not necessarily allocated
their KV/compute buffers for a large request. `--warm-request` first sends one
~12k-token chat request to each daemon port so the footprint is read after a
max-size request; it uses the daemons (and takes as long as the prompt eval).

Every external command goes through `_run`, so tests mock `subprocess.check_output`;
every signal that cannot be read carries a reason instead of a bare null.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

JEV_DIR = Path.home() / ".jev"
DEFAULT_FAST_LOG = JEV_DIR / "mlx_server_fast.log"
DEFAULT_SMART_LOG = JEV_DIR / "mlx_server_smart.log"
DEFAULT_FAST_PORT = "8080"
DEFAULT_SMART_PORT = "8081"
COMMAND_TIMEOUT_SECONDS = 15
PROPS_TIMEOUT_SECONDS = 3
WARM_REQUEST_TOKENS = 12000
WARM_REQUEST_TIMEOUT_SECONDS = 600
LOOPBACK = "127.0.0.1"

# A new server start begins a new block; only the last one describes the live process.
STARTUP_MARKERS = ("llama_server: initializing", "main: loading model")
BUFFER_PATTERN = re.compile(
    r"\b(?P<kind>model|KV|compute) buffer size\s*=\s*(?P<mib>[\d.]+)\s*MiB"
)
BUFFER_FIELDS = {"model": "model_mib", "KV": "kv_mib", "compute": "compute_mib"}

FOOTPRINT_PATTERN = re.compile(r"Footprint:\s*([\d.]+)\s*(B|KB|MB|GB|TB)\b")
PEAK_PATTERN = re.compile(r"phys_footprint_peak:\s*(\d+)\s*B\b")
CATEGORY_ROW_PATTERN = re.compile(
    r"^\s*(\d+) B\s+(\d+) B\s+(\d+) B\s+(\d+)\s+(\S.*?)\s*$", re.M
)
# The rows where Metal/mmap'd weights show up when they are resident at all.
CATEGORY_OF_INTEREST = re.compile(r"mapped file|IOAccelerator|IOSurface", re.I)
UNIT_BYTES = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}

SWAP_PATTERN = re.compile(r"\b(total|used|free)\s*=\s*([\d.]+)M")
VM_PAGE_SIZE_PATTERN = re.compile(r"page size of (\d+) bytes")
VM_WIRED_PATTERN = re.compile(r"^Pages wired down:\s+(\d+)\.", re.M)
VM_COUNTERS = {
    "Pageins": "pageins",
    "Pageouts": "pageouts",
    "Swapins": "swapins",
    "Swapouts": "swapouts",
}
VM_COUNTER_PATTERN = re.compile(
    r"^(Pageins|Pageouts|Swapins|Swapouts):\s+(\d+)\.", re.M
)
GPU_IN_USE_PATTERN = re.compile(r'"In use system memory"\s*=\s*(\d+)')

ROLES = ("fast", "smart")
MIB = 1024**2
REASON_SNIPPET_CHARS = 80


def _run(cmd):
    """The single subprocess boundary: (stdout, None) or (None, reason, returncode)."""
    name = cmd[0]
    try:
        out = subprocess.check_output(
            cmd,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
        return out, None, 0
    except FileNotFoundError:
        return None, f"{name} not found", None
    except subprocess.TimeoutExpired:
        return None, f"{name} timed out after {COMMAND_TIMEOUT_SECONDS}s", None
    except subprocess.CalledProcessError as exc:
        return None, f"{name} exited {exc.returncode}", exc.returncode
    except (subprocess.SubprocessError, OSError) as exc:
        return None, f"{name}: {exc}", None


def _unparseable(name, text):
    snippet = " ".join((text or "").split())[:REASON_SNIPPET_CHARS]
    return f"{name} output unparseable: {snippet!r}"


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
    except FileNotFoundError:
        return {**_empty_buffers(), "log": str(path), "error": "log not found"}
    except OSError as exc:
        return {
            **_empty_buffers(),
            "log": str(path),
            "error": f"{type(exc).__name__}: {exc}",
        }
    out = {**buffers, "log": str(path)}
    if all(v is None for v in buffers.values()):
        out["error"] = (
            "no model/KV/compute buffer lines in the last startup block "
            "(this llama-server build or verbosity does not print them)"
        )
    return out


# --- resident footprint -----------------------------------------------------


def find_pids(port):
    """(PIDs, reason) of llama-server processes on `port` (exact port, not a prefix)."""
    out, reason, code = _run(["pgrep", "-f", rf"llama-server.*--port {port}( |$)"])
    if out is None:
        if code == 1:
            return [], f"no llama-server process matches --port {port}"
        return [], reason
    pids = [int(tok) for tok in out.split() if tok.isdigit()]
    if not pids:
        return [], f"no llama-server process matches --port {port}"
    return pids, None


def parse_footprint(text):
    match = FOOTPRINT_PATTERN.search(text or "")
    if not match:
        return None
    peak = PEAK_PATTERN.search(text)
    categories = {}
    for dirty, clean, _reclaimable, _regions, name in CATEGORY_ROW_PATTERN.findall(
        text
    ):
        if CATEGORY_OF_INTEREST.search(name):
            categories[name] = {"dirty_bytes": int(dirty), "clean_bytes": int(clean)}
    return {
        "dirty_footprint_bytes": int(float(match[1]) * UNIT_BYTES[match[2]]),
        "peak_bytes": int(peak[1]) if peak else None,
        "categories": categories,
    }


def process_footprint(pid):
    """(parsed footprint | None, reason). Categories are kept (no --noCategories)."""
    out, reason, _ = _run(["footprint", "-p", str(pid), "-f", "bytes"])
    if out is None:
        return None, reason
    parsed = parse_footprint(out)
    if parsed is None:
        return None, _unparseable("footprint", out)
    return parsed, None


def model_path_from_args(args_text):
    """The `-m`/`--model` value of a llama-server command line, or None."""
    try:
        tokens = shlex.split(args_text or "")
    except ValueError:
        return None
    for i, tok in enumerate(tokens):
        if tok in ("-m", "--model") and i + 1 < len(tokens):
            return tokens[i + 1]
        if tok.startswith("--model="):
            return tok.split("=", 1)[1]
    return None


def _props_model_path(port):
    url = f"http://{LOOPBACK}:{port}/props"
    try:
        with urllib.request.urlopen(url, timeout=PROPS_TIMEOUT_SECONDS) as resp:
            return json.loads(resp.read().decode()).get("model_path")
    except (OSError, ValueError, AttributeError):
        return None


def gguf_info(pid, port):
    """{"model_path", "gguf_bytes"} plus "gguf_reason" when the size is unknown."""
    out, _, _ = _run(["ps", "-p", str(pid), "-o", "args="])
    path = model_path_from_args(out)
    if path is None:
        path = _props_model_path(port)
    if path is None:
        return {
            "model_path": None,
            "gguf_bytes": None,
            "gguf_reason": "no -m/--model in the process arguments and /props gave none",
        }
    try:
        return {"model_path": path, "gguf_bytes": os.stat(path).st_size}
    except OSError as exc:
        return {"model_path": path, "gguf_bytes": None, "gguf_reason": str(exc)}


def _merge_categories(per_process):
    merged = {}
    for categories in per_process:
        for name, rows in categories.items():
            have = merged.setdefault(name, {"dirty_bytes": 0, "clean_bytes": 0})
            have["dirty_bytes"] += rows["dirty_bytes"]
            have["clean_bytes"] += rows["clean_bytes"]
    return merged


def resident_footprint(port):
    pids, reason = find_pids(port)
    out = {
        "port": int(port),
        "pids": pids,
        "dirty_footprint_bytes": None,
        "peak_bytes": None,
        "categories": {},
        "model_path": None,
        "gguf_bytes": None,
    }
    if not pids:
        return {**out, "reason": reason}
    parsed, reasons = [], []
    for pid in pids:
        fp, why = process_footprint(pid)
        if fp is None:
            reasons.append(f"pid {pid}: {why}")
        else:
            parsed.append(fp)
    out.update(gguf_info(pids[0], port))
    if not parsed:
        return {**out, "reason": "; ".join(reasons)}
    out["dirty_footprint_bytes"] = sum(fp["dirty_footprint_bytes"] for fp in parsed)
    out["peak_bytes"] = sum(fp["peak_bytes"] or 0 for fp in parsed) or None
    out["categories"] = _merge_categories(fp["categories"] for fp in parsed)
    if reasons:
        out["reason"] = "; ".join(reasons)
    return out


def send_warm_request(port, tokens=WARM_REQUEST_TOKENS):
    """Send one ~`tokens`-token chat request so buffers are sized for a large prompt."""
    body = {
        "messages": [
            {
                "role": "user",
                "content": " ".join(["data"] * tokens) + "\n\nReply with one word.",
            }
        ],
        "max_tokens": 1,
        "temperature": 0,
    }
    req = urllib.request.Request(
        f"http://{LOOPBACK}:{port}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=WARM_REQUEST_TIMEOUT_SECONDS) as resp:
            resp.read()
    except (OSError, ValueError) as exc:
        return {"ok": False, "reason": str(exc)}
    return {"ok": True, "seconds": round(time.perf_counter() - t0, 1)}


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
    wired = VM_WIRED_PATTERN.search(text)
    size = int(page_size[1])
    return {
        "page_size": size,
        **counters,
        "wired_bytes": int(wired[1]) * size if wired else None,
    }


def parse_gpu_in_use(text):
    """The largest `"In use system memory"` value in ioreg IOAccelerator output."""
    values = [int(v) for v in GPU_IN_USE_PATTERN.findall(text or "")]
    return max(values) if values else None


def system_pressure():
    reasons = {}
    swap_text, why, _ = _run(["sysctl", "vm.swapusage"])
    swap = parse_swapusage(swap_text)
    if swap is None:
        reasons["swap"] = why or _unparseable("sysctl vm.swapusage", swap_text)
    vm_text, why, _ = _run(["vm_stat"])
    vm = parse_vm_stat(vm_text)
    if vm is None:
        reasons["vm"] = why or _unparseable("vm_stat", vm_text)
    gpu_text, why, _ = _run(
        ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"]
    )
    gpu = parse_gpu_in_use(gpu_text)
    if gpu is None:
        reasons["gpu_in_use"] = (
            why or "no 'In use system memory' in ioreg IOAccelerator output"
        )
    return {"swap": swap, "vm": vm, "gpu_in_use_bytes": gpu, "reasons": reasons}


# --- snapshot and delta -----------------------------------------------------


def snapshot(fast_log, smart_log, fast_port, smart_port, warm_request=False):
    logs = {"fast": fast_log, "smart": smart_log}
    ports = {"fast": fast_port, "smart": smart_port}
    out = {}
    if warm_request:
        out["warm_request"] = {role: send_warm_request(ports[role]) for role in ROLES}
    out["startup"] = {
        role: read_startup_buffers(logs[role]) if logs[role] else _empty_buffers()
        for role in ROLES
    }
    out["resident"] = {role: resident_footprint(ports[role]) for role in ROLES}
    out["system"] = system_pressure()
    return out


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
    for counter in (*VM_COUNTERS.values(), "wired_bytes"):
        out[counter] = _diff(
            _get(before, "system", "vm", counter), _get(after, "system", "vm", counter)
        )
    out["gpu_in_use_bytes"] = _diff(
        _get(before, "system", "gpu_in_use_bytes"),
        _get(after, "system", "gpu_in_use_bytes"),
    )
    out["dirty_footprint_bytes"] = {
        role: _diff(
            _get(before, "resident", role, "dirty_footprint_bytes"),
            _get(after, "resident", role, "dirty_footprint_bytes"),
        )
        for role in ROLES
    }
    return out


# --- CLI --------------------------------------------------------------------


def _mib(value, divisor=1, reason=None):
    if value is None:
        return f"n/a ({reason})" if reason else "n/a"
    return f"{value / divisor:,.1f} MiB"


def format_snapshot(snap):
    lines = []
    for role in ROLES:
        buf = snap["startup"][role]
        res = snap["resident"][role]
        pids = res["pids"] or "n/a"
        lines.append(f"{role} (port {res.get('port', 'n/a')}, pids {pids})")
        lines.append(
            f"  startup buffers: model {_mib(buf['model_mib'])}, "
            f"KV {_mib(buf['kv_mib'])}, compute {_mib(buf['compute_mib'])}"
        )
        if buf.get("error"):
            lines.append(f"    ({buf['error']})")
        lines.append(
            f"  dirty footprint: "
            f"{_mib(res['dirty_footprint_bytes'], MIB, res.get('reason'))}"
            f" (peak {_mib(res['peak_bytes'], MIB)})"
        )
        for name, rows in res.get("categories", {}).items():
            lines.append(
                f"    {name}: dirty {_mib(rows['dirty_bytes'], MIB)}, "
                f"clean {_mib(rows['clean_bytes'], MIB)}"
            )
        lines.append(
            f"  GGUF on disk: {_mib(res.get('gguf_bytes'), MIB, res.get('gguf_reason'))}"
        )
    system = snap["system"]
    swap, vm = system["swap"], system["vm"]
    reasons = system.get("reasons", {})
    lines.append(
        f"system: swap used {_mib(swap and swap['used_mib'], 1, reasons.get('swap'))}"
        f" of {_mib(swap and swap['total_mib'])}"
    )
    lines.append(
        "  pageouts {}, swapouts {}".format(
            "n/a" if vm is None else f"{vm['pageouts']:,}",
            "n/a" if vm is None else f"{vm['swapouts']:,}",
        )
        + (f" ({reasons['vm']})" if "vm" in reasons else "")
    )
    lines.append(
        "  wired {}, GPU in use {}".format(
            _mib(vm and vm["wired_bytes"], MIB),
            _mib(system["gpu_in_use_bytes"], MIB, reasons.get("gpu_in_use")),
        )
    )
    for role, result in snap.get("warm_request", {}).items():
        lines.append(f"warm request {role}: {result}")
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
    parser.add_argument(
        "--warm-request",
        action="store_true",
        help=f"send one ~{WARM_REQUEST_TOKENS}-token request to each daemon first, "
        "so the footprint is read after a max-size request (default: idle)",
    )
    parser.add_argument(
        "--delta",
        nargs=2,
        metavar=("BEFORE_JSON", "AFTER_JSON"),
        help="print after - before for two saved --json snapshots (no sampling)",
    )
    return parser


def _load_snapshot(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read snapshot {path}: {exc}") from None


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.delta:
        before, after = (_load_snapshot(p) for p in args.delta)
        print(json.dumps(delta(before, after), indent=2))
        return
    snap = snapshot(
        args.fast_log,
        args.smart_log,
        args.fast_port,
        args.smart_port,
        warm_request=args.warm_request,
    )
    print(json.dumps(snap, indent=2) if args.json else format_snapshot(snap))


if __name__ == "__main__":
    main()
