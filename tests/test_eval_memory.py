"""Memory recorder: log, footprint, swap and vm_stat parsing, all at the subprocess boundary."""

import json
import subprocess
import urllib.error
import urllib.request

import pytest

from tests.evals import memory
from tests.evals.memory import (
    delta,
    find_pids,
    model_path_from_args,
    parse_footprint,
    parse_gpu_in_use,
    parse_startup_buffers,
    parse_swapusage,
    parse_vm_stat,
    read_startup_buffers,
    snapshot,
)

# Shape of a llama-server startup block (device names and sizes are representative).
STARTUP_BLOCK = """\
0.00.239.415 I srv  llama_server: initializing ...
0.21.747.416 I srv    load_model: loading model '/models/m.gguf'
0.26.376.312 W load: control-looking token: 128247 '</s>' was not control-type
1.01.100.000 I load_tensors:   CPU_Mapped model buffer size =   292.36 MiB
1.01.100.001 I load_tensors: MTL0_Mapped model buffer size =  4168.09 MiB
1.20.000.000 I llama_kv_cache:       MTL0 KV buffer size =   448.00 MiB
1.20.000.001 I llama_kv_cache: size =  448.00 MiB (16384 cells, 28 layers)
1.22.000.000 I sched_reserve:       MTL0 compute buffer size =   304.00 MiB
1.22.000.001 I sched_reserve:        CPU compute buffer size =    39.01 MiB
1.24.912.821 I srv  llama_server: model loaded
"""

OLD_BLOCK = STARTUP_BLOCK.replace("1.01.100.001", "9.99.999.999").replace(
    "4168.09", "1111.00"
)

SERVE_NOISE = """\
93.27.751.004 W srv         alloc:  - making room for prompt cache entry (size = 15.316 MiB)
93.43.488.898 I slot      release: id  0 | task 7104 | stop processing: n_tokens = 153
"""

# `footprint -p <pid> -f bytes` as captured from a running llama-server (trimmed).
FOOTPRINT_BYTES = """\
======================================================================
llama-server [1604]: 64-bit    Footprint: 5249376256 B (16384 bytes per page)
======================================================================

      Dirty         Clean   Reclaimable    Regions    Category
        ---           ---           ---        ---    ---
1895628800 B           0 B           0 B         41    Untagged
 73580544 B           0 B           0 B         32    Malloc Small
   491520 B           0 B           0 B        108    IOAccelerator (graphics)
    65536 B           0 B           0 B          2    IOAccelerator
        0 B   4400000000 B           0 B        487    mapped file
        0 B           0 B           0 B          1    IOSurface
        ---           ---           ---        ---    ---
2032981608 B      131072 B           0 B       3853    TOTAL

Auxiliary data:
    phys_footprint: 5249376256 B
    phys_footprint_peak: 6012345344 B

"""

FOOTPRINT_FORMATTED = """\
======================================================================
llama-server [1604]: 64-bit    Footprint: 5006 MB (16384 bytes per page)
======================================================================
"""

SWAPUSAGE = (
    "vm.swapusage: total = 14336.00M  used = 13668.12M  free = 667.88M  (encrypted)\n"
)

VM_STAT = """\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                     4320.
Pages wired down:                             728315.
"Translation faults":                      262755675.
Pageins:                                    16486590.
Pageouts:                                     600948.
Swapins:                                     1896676.
Swapouts:                                    2894821.
"""

IOREG = """\
+-o AGXAcceleratorG14X  <class AGXAcceleratorG14X, id 0x1000003f1>
    {
      "PerformanceStatistics" = {"Alloc system memory"=12699762688,"In use system memory"=125894656,"Device Utilization %"=0}
    }
"""

PS_ARGS = (
    "/opt/homebrew/bin/llama-server -m /models/Qwen2.5-7B-Instruct-Q4_K_M.gguf "
    "--port 8081 --parallel 2 -c 32768\n"
)


@pytest.fixture(autouse=True)
def no_real_http(monkeypatch):
    """The recorder talks to loopback daemons; no test may reach a real one."""
    calls = []

    def refuse(req, timeout=None):
        calls.append(req)
        raise urllib.error.URLError("no network in tests")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    return calls


# --- startup log ------------------------------------------------------------


def test_buffers_are_summed_across_devices_in_mib():
    out = parse_startup_buffers(STARTUP_BLOCK.splitlines())
    assert out["model_mib"] == pytest.approx(292.36 + 4168.09)
    assert out["kv_mib"] == pytest.approx(448.0)
    assert out["compute_mib"] == pytest.approx(304.0 + 39.01)


def test_the_last_startup_block_wins_over_earlier_ones():
    lines = (OLD_BLOCK + SERVE_NOISE + STARTUP_BLOCK).splitlines()
    out = parse_startup_buffers(lines)
    assert out["model_mib"] == pytest.approx(292.36 + 4168.09)


def test_a_last_block_without_buffer_lines_reports_none_not_an_older_value():
    # A log whose newest start printed no buffer lines must not borrow the previous start's.
    quiet = "0.00.100.000 I srv  llama_server: initializing ...\n0.01 I srv llama_server: model loaded\n"
    out = parse_startup_buffers((STARTUP_BLOCK + SERVE_NOISE + quiet).splitlines())
    assert out == {"model_mib": None, "kv_mib": None, "compute_mib": None}


def test_a_log_with_no_startup_marker_is_read_whole():
    lines = ["I load_tensors: MTL0_Mapped model buffer size =  100.50 MiB"]
    assert parse_startup_buffers(lines)["model_mib"] == pytest.approx(100.5)


def test_unrelated_lines_yield_all_none():
    out = parse_startup_buffers(SERVE_NOISE.splitlines())
    assert out == {"model_mib": None, "kv_mib": None, "compute_mib": None}


def test_reading_a_log_file(tmp_path):
    log = tmp_path / "fast.log"
    log.write_text(OLD_BLOCK + SERVE_NOISE + STARTUP_BLOCK)
    out = read_startup_buffers(log)
    assert out["kv_mib"] == pytest.approx(448.0)
    assert out["log"] == str(log)
    assert "error" not in out


def test_a_log_without_buffer_lines_says_why_the_values_are_null(tmp_path):
    log = tmp_path / "smart.log"
    log.write_text(SERVE_NOISE)
    out = read_startup_buffers(log)
    assert out["model_mib"] is None
    assert "no model/KV/compute buffer lines" in out["error"]


def test_a_missing_log_is_reported_not_raised(tmp_path):
    out = read_startup_buffers(tmp_path / "absent.log")
    assert out["model_mib"] is None
    assert "not found" in out["error"]


def test_an_unreadable_log_is_not_called_not_found(tmp_path):
    out = read_startup_buffers(tmp_path)  # a directory: OSError, not FileNotFoundError
    assert "not found" not in out["error"]
    assert "IsADirectoryError" in out["error"]


# --- footprint --------------------------------------------------------------


def test_footprint_is_reported_as_dirty_footprint_with_its_peak():
    out = parse_footprint(FOOTPRINT_BYTES)
    assert out["dirty_footprint_bytes"] == 5249376256
    assert out["peak_bytes"] == 6012345344
    assert "footprint_bytes" not in out


def test_graphics_and_mapped_file_categories_are_captured():
    cats = parse_footprint(FOOTPRINT_BYTES)["categories"]
    assert cats["mapped file"] == {"dirty_bytes": 0, "clean_bytes": 4400000000}
    assert cats["IOAccelerator"] == {"dirty_bytes": 65536, "clean_bytes": 0}
    assert cats["IOAccelerator (graphics)"]["dirty_bytes"] == 491520
    assert "IOSurface" in cats
    assert "Malloc Small" not in cats and "TOTAL" not in cats


def test_footprint_formatted_units_are_converted():
    out = parse_footprint(FOOTPRINT_FORMATTED)
    assert out["dirty_footprint_bytes"] == 5006 * 1024**2
    assert out["peak_bytes"] is None
    assert out["categories"] == {}


def test_footprint_of_unparseable_text_is_none():
    assert parse_footprint("no such process") is None


def test_the_model_path_is_taken_from_the_dash_m_argument():
    assert model_path_from_args(PS_ARGS) == "/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    assert (
        model_path_from_args("llama-server --model /a/b.gguf --port 1") == "/a/b.gguf"
    )
    assert model_path_from_args("llama-server --port 1") is None
    assert model_path_from_args("") is None


# --- swap, vm_stat, gpu -----------------------------------------------------


def test_swapusage_is_parsed_in_mib():
    assert parse_swapusage(SWAPUSAGE) == {
        "total_mib": 14336.0,
        "used_mib": 13668.12,
        "free_mib": 667.88,
    }


def test_swapusage_of_garbage_is_none():
    assert parse_swapusage("nope") is None


def test_vm_stat_counters_page_size_and_wired_memory():
    assert parse_vm_stat(VM_STAT) == {
        "page_size": 16384,
        "pageins": 16486590,
        "pageouts": 600948,
        "swapins": 1896676,
        "swapouts": 2894821,
        "wired_bytes": 728315 * 16384,
    }


def test_vm_stat_without_a_wired_line_still_parses_the_counters():
    text = VM_STAT.replace(
        "Pages wired down:                             728315.\n", ""
    )
    assert parse_vm_stat(text)["wired_bytes"] is None


def test_vm_stat_of_garbage_is_none():
    assert parse_vm_stat("nope") is None


def test_gpu_in_use_system_memory_is_read_from_ioreg():
    assert parse_gpu_in_use(IOREG) == 125894656


def test_gpu_in_use_of_unrelated_ioreg_output_is_none():
    assert parse_gpu_in_use("+-o Root\n") is None


# --- subprocess boundary ----------------------------------------------------


def fake_check_output(
    pids=None,
    footprints=None,
    swap=SWAPUSAGE,
    vm=VM_STAT,
    ioreg=IOREG,
    args=None,
):
    """Dispatch on argv[0]; every command goes through subprocess.check_output."""
    pids = pids or {}
    footprints = footprints or {}
    args = args if args is not None else {}

    def run(cmd, **kwargs):
        if cmd[0] == "pgrep":
            for port, found in pids.items():
                if f"--port {port}" in cmd[-1]:
                    return "\n".join(str(p) for p in found) + "\n"
            raise subprocess.CalledProcessError(1, cmd)
        if cmd[0] == "footprint":
            pid = int(cmd[cmd.index("-p") + 1])
            return footprints[pid]
        if cmd[0] == "ps":
            pid = int(cmd[cmd.index("-p") + 1])
            if pid not in args:
                raise subprocess.CalledProcessError(1, cmd)
            return args[pid]
        if cmd[0] == "sysctl":
            return swap
        if cmd[0] == "vm_stat":
            return vm
        if cmd[0] == "ioreg":
            if ioreg is None:
                raise FileNotFoundError("ioreg")
            return ioreg
        raise AssertionError(f"unexpected command {cmd}")

    return run


def test_find_pids_matches_the_port_exactly(monkeypatch):
    seen = []

    def run(cmd, **kwargs):
        seen.append(cmd)
        return "1604\n"

    monkeypatch.setattr(subprocess, "check_output", run)
    assert find_pids(8081) == ([1604], None)
    assert seen[0][0] == "pgrep"
    assert seen[0][-1] == r"llama-server.*--port 8081( |$)"


def test_find_pids_with_no_match_says_so(monkeypatch):
    monkeypatch.setattr(subprocess, "check_output", fake_check_output())
    pids, reason = find_pids(8080)
    assert pids == []
    assert "no llama-server process" in reason and "8080" in reason


def test_a_missing_tool_is_reported_with_its_name(monkeypatch):
    def run(cmd, **kwargs):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(subprocess, "check_output", run)
    pids, reason = find_pids(8080)
    assert pids == [] and "pgrep" in reason and "not found" in reason


def test_a_timeout_and_a_permission_error_are_distinguished(monkeypatch):
    def timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 15)

    monkeypatch.setattr(subprocess, "check_output", timeout)
    assert "timed out" in find_pids(8080)[1]

    def denied(cmd, **kwargs):
        raise PermissionError("Operation not permitted")

    monkeypatch.setattr(subprocess, "check_output", denied)
    assert "Operation not permitted" in find_pids(8080)[1]


def test_unparseable_output_gets_its_own_reason(monkeypatch):
    monkeypatch.setattr(
        subprocess, "check_output", fake_check_output(swap="garbage\n", vm="junk\n")
    )
    system = snapshot(None, None, 8080, 8081)["system"]
    assert system["swap"] is None and system["vm"] is None
    assert "unparseable" in system["reasons"]["swap"]
    assert "unparseable" in system["reasons"]["vm"]


def test_every_tool_failing_yields_reasons_not_exceptions(monkeypatch):
    def run(cmd, **kwargs):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(subprocess, "check_output", run)
    out = snapshot(fast_log=None, smart_log=None, fast_port=8080, smart_port=8081)
    assert out["system"]["swap"] is None
    assert out["system"]["vm"] is None
    assert out["system"]["gpu_in_use_bytes"] is None
    assert set(out["system"]["reasons"]) == {"swap", "vm", "gpu_in_use"}
    assert out["resident"]["fast"]["dirty_footprint_bytes"] is None
    assert out["resident"]["fast"]["reason"]


@pytest.fixture
def two_daemons(monkeypatch, tmp_path):
    gguf = tmp_path / "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    gguf.write_bytes(b"x" * 1234)
    monkeypatch.setattr(
        subprocess,
        "check_output",
        fake_check_output(
            pids={8080: [1596], 8081: [1604]},
            footprints={1596: FOOTPRINT_FORMATTED, 1604: FOOTPRINT_BYTES},
            args={1604: f"llama-server -m {gguf} --port 8081\n"},
        ),
    )
    fast_log = tmp_path / "fast.log"
    smart_log = tmp_path / "smart.log"
    fast_log.write_text(STARTUP_BLOCK)
    smart_log.write_text(OLD_BLOCK)
    return fast_log, smart_log


def test_snapshot_joins_startup_resident_and_system(two_daemons):
    fast_log, smart_log = two_daemons
    snap = snapshot(fast_log, smart_log, fast_port=8080, smart_port=8081)

    assert snap["startup"]["fast"]["model_mib"] == pytest.approx(292.36 + 4168.09)
    assert snap["startup"]["smart"]["model_mib"] == pytest.approx(292.36 + 1111.0)
    smart = snap["resident"]["smart"]
    assert smart["port"] == 8081
    assert smart["pids"] == [1604]
    assert smart["dirty_footprint_bytes"] == 5249376256
    assert smart["peak_bytes"] == 6012345344
    assert smart["categories"]["mapped file"]["clean_bytes"] == 4400000000
    assert smart["gguf_bytes"] == 1234
    assert smart["model_path"].endswith("Qwen2.5-7B-Instruct-Q4_K_M.gguf")
    assert snap["resident"]["fast"]["dirty_footprint_bytes"] == 5006 * 1024**2
    assert snap["system"]["swap"]["used_mib"] == 13668.12
    assert snap["system"]["vm"]["pageouts"] == 600948
    assert snap["system"]["vm"]["wired_bytes"] == 728315 * 16384
    assert snap["system"]["gpu_in_use_bytes"] == 125894656
    assert "warm_request" not in snap


def test_gguf_size_comes_from_props_when_ps_has_no_model_argument(
    monkeypatch, tmp_path
):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x" * 99)
    monkeypatch.setattr(
        subprocess,
        "check_output",
        fake_check_output(
            pids={8081: [1604]},
            footprints={1604: FOOTPRINT_BYTES},
            args={1604: "llama-server --port 8081\n"},
        ),
    )
    urls = []

    def urlopen(url, timeout=None):
        urls.append(url)
        return FakeResponse({"model_path": str(gguf)})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    out = snapshot(None, None, 8080, 8081)["resident"]["smart"]
    assert urls == ["http://127.0.0.1:8081/props"]
    assert out["gguf_bytes"] == 99


def test_a_gguf_that_cannot_be_found_has_a_reason(monkeypatch, tmp_path):
    monkeypatch.setattr(
        subprocess,
        "check_output",
        fake_check_output(
            pids={8081: [1604]},
            footprints={1604: FOOTPRINT_BYTES},
            args={1604: f"llama-server -m {tmp_path}/gone.gguf --port 8081\n"},
        ),
    )
    out = snapshot(None, None, 8080, 8081)["resident"]["smart"]
    assert out["gguf_bytes"] is None
    assert "gone.gguf" in out["gguf_reason"]


def test_several_pids_on_one_port_are_summed(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "check_output",
        fake_check_output(
            pids={8080: [1, 2]}, footprints={1: FOOTPRINT_BYTES, 2: FOOTPRINT_BYTES}
        ),
    )
    out = snapshot(None, None, 8080, 8081)["resident"]["fast"]
    assert out["pids"] == [1, 2]
    assert out["dirty_footprint_bytes"] == 2 * 5249376256


def test_a_daemon_that_is_not_running_has_no_footprint_and_a_reason(monkeypatch):
    monkeypatch.setattr(subprocess, "check_output", fake_check_output())
    out = snapshot(None, None, 8080, 8081)["resident"]["smart"]
    assert out["pids"] == []
    assert out["dirty_footprint_bytes"] is None
    assert "no llama-server process" in out["reason"]


def test_a_pid_whose_footprint_is_unreadable_says_so(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "check_output",
        fake_check_output(pids={8080: [7]}, footprints={7: "not a footprint\n"}),
    )
    out = snapshot(None, None, 8080, 8081)["resident"]["fast"]
    assert out["dirty_footprint_bytes"] is None
    assert "unparseable" in out["reason"]


# --- warm request -----------------------------------------------------------


class FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_warm_request_posts_one_large_chat_request_to_each_port(
    two_daemons, monkeypatch
):
    fast_log, smart_log = two_daemons
    sent = []

    def urlopen(req, timeout=None):
        if isinstance(req, str):  # a /props lookup
            return FakeResponse({})
        sent.append((req.full_url, json.loads(req.data), timeout))
        return FakeResponse({"choices": []})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    snap = snapshot(fast_log, smart_log, 8080, 8081, warm_request=True)

    assert [u for u, _, _ in sent] == [
        "http://127.0.0.1:8080/v1/chat/completions",
        "http://127.0.0.1:8081/v1/chat/completions",
    ]
    body = sent[0][1]
    assert body["max_tokens"] == 1
    # about WARM_REQUEST_TOKENS tokens of padding (one short word per token)
    words = body["messages"][0]["content"].split()
    assert (
        0.9 * memory.WARM_REQUEST_TOKENS
        <= len(words)
        <= 1.1 * memory.WARM_REQUEST_TOKENS
    )
    assert sent[0][2] >= 60
    assert snap["warm_request"]["fast"]["ok"] is True
    assert snap["warm_request"]["smart"]["ok"] is True


def test_a_failed_warm_request_is_recorded_not_raised(two_daemons, monkeypatch):
    fast_log, smart_log = two_daemons

    def urlopen(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    snap = snapshot(fast_log, smart_log, 8080, 8081, warm_request=True)
    assert snap["warm_request"]["fast"]["ok"] is False
    assert "connection refused" in snap["warm_request"]["fast"]["reason"]


def test_warm_request_is_off_by_default(two_daemons, no_real_http):
    fast_log, smart_log = two_daemons
    snap = snapshot(fast_log, smart_log, 8080, 8081)
    assert "warm_request" not in snap
    # only /props lookups (plain URL strings), never a POST
    assert all(isinstance(call, str) for call in no_real_http)


# --- delta ------------------------------------------------------------------


def make_snap(smart_bytes, swap_used, pageouts, swapouts, wired=1000, gpu=50):
    return {
        "resident": {
            "fast": {"dirty_footprint_bytes": 100},
            "smart": {"dirty_footprint_bytes": smart_bytes},
        },
        "system": {
            "swap": {"used_mib": swap_used},
            "vm": {
                "pageouts": pageouts,
                "swapouts": swapouts,
                "pageins": 5,
                "swapins": 5,
                "wired_bytes": wired,
            },
            "gpu_in_use_bytes": gpu,
        },
    }


def test_delta_reports_what_grew_between_two_snapshots():
    before = make_snap(1000, 100.0, 10, 20, wired=1000, gpu=50)
    after = make_snap(1500, 160.5, 17, 26, wired=1800, gpu=90)
    assert delta(before, after) == {
        "swap_used_mib": 60.5,
        "pageouts": 7,
        "swapouts": 6,
        "pageins": 0,
        "swapins": 0,
        "wired_bytes": 800,
        "gpu_in_use_bytes": 40,
        "dirty_footprint_bytes": {"fast": 0, "smart": 500},
    }


def test_delta_is_none_where_either_side_is_missing():
    before = make_snap(None, 100.0, 10, 20)
    before["system"]["swap"] = None
    after = make_snap(1500, 160.0, 17, 26)
    out = delta(before, after)
    assert out["swap_used_mib"] is None
    assert out["dirty_footprint_bytes"]["smart"] is None
    assert out["pageouts"] == 7


# --- CLI --------------------------------------------------------------------


def test_main_json_prints_one_snapshot(two_daemons, capsys):
    fast_log, smart_log = two_daemons
    memory.main(
        [
            "--json",
            "--fast-log",
            str(fast_log),
            "--smart-log",
            str(smart_log),
            "--fast-port",
            "8080",
            "--smart-port",
            "8081",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"startup", "resident", "system"}
    assert payload["resident"]["smart"]["pids"] == [1604]


def test_main_text_is_a_readable_summary(two_daemons, capsys):
    fast_log, smart_log = two_daemons
    memory.main(["--fast-log", str(fast_log), "--smart-log", str(smart_log)])
    out = capsys.readouterr().out
    assert "fast" in out and "smart" in out
    assert "swap used" in out and "pageouts" in out and "wired" in out
    assert "dirty footprint" in out and "GGUF" in out
    assert "MiB" in out


def test_main_text_marks_missing_values_with_their_reason(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(subprocess, "check_output", fake_check_output(ioreg=None))
    memory.main(
        ["--fast-log", str(tmp_path / "a.log"), "--smart-log", str(tmp_path / "b.log")]
    )
    out = capsys.readouterr().out
    assert "n/a" in out
    assert "no llama-server process" in out
    assert "not found" in out


def test_default_ports_come_from_the_environment(monkeypatch, capsys):
    seen = []

    def run(cmd, **kwargs):
        if cmd[0] == "pgrep":
            seen.append(cmd[-1])
            raise subprocess.CalledProcessError(1, cmd)
        return {"sysctl": SWAPUSAGE, "vm_stat": VM_STAT, "ioreg": IOREG}[cmd[0]]

    monkeypatch.setattr(subprocess, "check_output", run)
    monkeypatch.setenv("JEV_FAST_PORT", "9001")
    monkeypatch.setenv("JEV_SMART_PORT", "9002")
    memory.main(
        ["--json", "--fast-log", "/nonexistent/a", "--smart-log", "/nonexistent/b"]
    )
    assert any("--port 9001" in s for s in seen)
    assert any("--port 9002" in s for s in seen)


def test_main_warm_request_flag_sends_requests(two_daemons, monkeypatch, capsys):
    fast_log, smart_log = two_daemons
    sent = []

    def urlopen(req, timeout=None):
        if not isinstance(req, str):
            sent.append(req.full_url)
        return FakeResponse({})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    memory.main(
        [
            "--json",
            "--warm-request",
            "--fast-log",
            str(fast_log),
            "--smart-log",
            str(smart_log),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert len(sent) == 2
    assert payload["warm_request"]["smart"]["ok"] is True


def test_main_delta_diffs_two_saved_snapshots(tmp_path, capsys):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(json.dumps(make_snap(1000, 100.0, 10, 20)))
    after.write_text(json.dumps(make_snap(1500, 160.5, 17, 26)))
    memory.main(["--delta", str(before), str(after)])
    out = json.loads(capsys.readouterr().out)
    assert out["pageouts"] == 7
    assert out["dirty_footprint_bytes"]["smart"] == 500


def test_main_delta_with_an_unreadable_file_exits_cleanly(tmp_path):
    good = tmp_path / "good.json"
    good.write_text("{}")
    bad = tmp_path / "bad.json"
    bad.write_text("{nope")
    with pytest.raises(SystemExit) as exc:
        memory.main(["--delta", str(good), str(bad)])
    assert "bad.json" in str(exc.value)
    with pytest.raises(SystemExit) as exc:
        memory.main(["--delta", str(good), str(tmp_path / "missing.json")])
    assert "missing.json" in str(exc.value)
