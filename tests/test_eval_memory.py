"""Memory recorder: log, footprint, swap and vm_stat parsing, all at the subprocess boundary."""

import json
import subprocess

import pytest

from tests.evals import memory
from tests.evals.memory import (
    delta,
    find_pids,
    parse_footprint,
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

FOOTPRINT_BYTES = """\
======================================================================
llama-server [1604]: 64-bit    Footprint: 5249376256 B (16384 bytes per page)
======================================================================

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


def test_a_missing_log_is_reported_not_raised(tmp_path):
    out = read_startup_buffers(tmp_path / "absent.log")
    assert out["model_mib"] is None
    assert "not found" in out["error"]


# --- footprint --------------------------------------------------------------


def test_footprint_in_bytes_and_peak():
    assert parse_footprint(FOOTPRINT_BYTES) == {
        "footprint_bytes": 5249376256,
        "peak_bytes": 6012345344,
    }


def test_footprint_formatted_units_are_converted():
    out = parse_footprint(FOOTPRINT_FORMATTED)
    assert out["footprint_bytes"] == 5006 * 1024**2
    assert out["peak_bytes"] is None


def test_footprint_of_unparseable_text_is_none():
    assert parse_footprint("no such process") is None


# --- swap and vm_stat -------------------------------------------------------


def test_swapusage_is_parsed_in_mib():
    assert parse_swapusage(SWAPUSAGE) == {
        "total_mib": 14336.0,
        "used_mib": 13668.12,
        "free_mib": 667.88,
    }


def test_swapusage_of_garbage_is_none():
    assert parse_swapusage("nope") is None


def test_vm_stat_counters_and_page_size():
    assert parse_vm_stat(VM_STAT) == {
        "page_size": 16384,
        "pageins": 16486590,
        "pageouts": 600948,
        "swapins": 1896676,
        "swapouts": 2894821,
    }


def test_vm_stat_of_garbage_is_none():
    assert parse_vm_stat("nope") is None


# --- subprocess boundary ----------------------------------------------------


def fake_check_output(pids=None, footprints=None, swap=SWAPUSAGE, vm=VM_STAT):
    """Dispatch on argv[0]; every command goes through subprocess.check_output."""
    pids = pids or {}
    footprints = footprints or {}

    def run(cmd, **kwargs):
        if cmd[0] == "pgrep":
            for port, found in pids.items():
                if f"--port {port}" in cmd[-1]:
                    return "\n".join(str(p) for p in found) + "\n"
            raise subprocess.CalledProcessError(1, cmd)
        if cmd[0] == "footprint":
            pid = int(cmd[cmd.index("-p") + 1])
            return footprints[pid]
        if cmd[0] == "sysctl":
            return swap
        if cmd[0] == "vm_stat":
            return vm
        raise AssertionError(f"unexpected command {cmd}")

    return run


def test_find_pids_matches_the_port_exactly(monkeypatch):
    seen = []

    def run(cmd, **kwargs):
        seen.append(cmd)
        return "1604\n"

    monkeypatch.setattr(subprocess, "check_output", run)
    assert find_pids(8081) == [1604]
    assert seen[0][0] == "pgrep"
    assert seen[0][-1] == r"llama-server.*--port 8081( |$)"


def test_find_pids_with_no_match_is_empty(monkeypatch):
    monkeypatch.setattr(subprocess, "check_output", fake_check_output())
    assert find_pids(8080) == []


def test_a_missing_tool_is_none_not_an_exception(monkeypatch):
    def run(cmd, **kwargs):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(subprocess, "check_output", run)
    assert find_pids(8080) == []
    out = snapshot(fast_log=None, smart_log=None, fast_port=8080, smart_port=8081)
    assert out["system"] == {"swap": None, "vm": None}
    assert out["resident"]["fast"]["footprint_bytes"] is None


@pytest.fixture
def two_daemons(monkeypatch, tmp_path):
    monkeypatch.setattr(
        subprocess,
        "check_output",
        fake_check_output(
            pids={8080: [1596], 8081: [1604]},
            footprints={1596: FOOTPRINT_FORMATTED, 1604: FOOTPRINT_BYTES},
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
    assert snap["resident"]["smart"] == {
        "port": 8081,
        "pids": [1604],
        "footprint_bytes": 5249376256,
        "peak_bytes": 6012345344,
    }
    assert snap["resident"]["fast"]["footprint_bytes"] == 5006 * 1024**2
    assert snap["system"]["swap"]["used_mib"] == 13668.12
    assert snap["system"]["vm"]["pageouts"] == 600948


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
    assert out["footprint_bytes"] == 2 * 5249376256


def test_a_daemon_that_is_not_running_has_no_footprint(monkeypatch):
    monkeypatch.setattr(subprocess, "check_output", fake_check_output())
    out = snapshot(None, None, 8080, 8081)["resident"]["smart"]
    assert out == {
        "port": 8081,
        "pids": [],
        "footprint_bytes": None,
        "peak_bytes": None,
    }


# --- delta ------------------------------------------------------------------


def make_snap(smart_bytes, swap_used, pageouts, swapouts):
    return {
        "resident": {
            "fast": {"footprint_bytes": 100},
            "smart": {"footprint_bytes": smart_bytes},
        },
        "system": {
            "swap": {"used_mib": swap_used},
            "vm": {
                "pageouts": pageouts,
                "swapouts": swapouts,
                "pageins": 5,
                "swapins": 5,
            },
        },
    }


def test_delta_reports_what_grew_between_two_snapshots():
    before = make_snap(1000, 100.0, 10, 20)
    after = make_snap(1500, 160.5, 17, 26)
    assert delta(before, after) == {
        "swap_used_mib": 60.5,
        "pageouts": 7,
        "swapouts": 6,
        "pageins": 0,
        "swapins": 0,
        "footprint_bytes": {"fast": 0, "smart": 500},
    }


def test_delta_is_none_where_either_side_is_missing():
    before = make_snap(None, 100.0, 10, 20)
    before["system"]["swap"] = None
    after = make_snap(1500, 160.0, 17, 26)
    out = delta(before, after)
    assert out["swap_used_mib"] is None
    assert out["footprint_bytes"]["smart"] is None
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
    assert "swap used" in out and "pageouts" in out
    assert "MiB" in out


def test_main_text_marks_missing_values(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(subprocess, "check_output", fake_check_output())
    memory.main(
        ["--fast-log", str(tmp_path / "a.log"), "--smart-log", str(tmp_path / "b.log")]
    )
    assert "n/a" in capsys.readouterr().out


def test_default_ports_come_from_the_environment(monkeypatch, capsys):
    seen = []

    def run(cmd, **kwargs):
        if cmd[0] == "pgrep":
            seen.append(cmd[-1])
            raise subprocess.CalledProcessError(1, cmd)
        return SWAPUSAGE if cmd[0] == "sysctl" else VM_STAT

    monkeypatch.setattr(subprocess, "check_output", run)
    monkeypatch.setenv("JEV_FAST_PORT", "9001")
    monkeypatch.setenv("JEV_SMART_PORT", "9002")
    memory.main(
        ["--json", "--fast-log", "/nonexistent/a", "--smart-log", "/nonexistent/b"]
    )
    assert any("--port 9001" in s for s in seen)
    assert any("--port 9002" in s for s in seen)
