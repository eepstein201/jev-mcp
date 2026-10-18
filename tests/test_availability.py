import json
import os
import signal
import threading
from datetime import datetime, timedelta, timezone

import pytest

from jev_mcp import availability, doctor

T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _ok_check(name, env=os.environ):
    return {"port": 1, "status": "up", "probe": {"ok": True, "detail": "ok"}}


def _row(minute, engine="fast", health=True, probe=True, latency=10.0, error=None):
    return {
        "ts": (T0 + timedelta(minutes=minute)).isoformat(),
        "engine": engine,
        "health_ok": health,
        "probe_ok": probe,
        "latency_ms": latency,
        "error": error,
    }


def _parsed(rows):
    return [availability._parse_row(json.dumps(r)) for r in rows]


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


# --- doctor.check_engine ---------------------------------------------------


def test_check_engine_probes_the_fast_engine_through_systemone(monkeypatch):
    monkeypatch.setattr(doctor, "_health_status", lambda port: "up")
    monkeypatch.setattr(doctor, "_probe_fast", lambda port: {"ok": True})
    monkeypatch.setattr(doctor, "_probe", lambda port: pytest.fail("chat probe"))
    got = doctor.check_engine("fast", {"JEV_FAST_PORT": "9001"})
    assert got == {"port": 9001, "status": "up", "probe": {"ok": True}}


def test_check_engine_probes_smart_and_non_kev_fast_through_chat(monkeypatch):
    monkeypatch.setattr(doctor, "_health_status", lambda port: "up")
    monkeypatch.setattr(doctor, "_probe", lambda port: {"ok": True, "p": port})
    monkeypatch.setattr(doctor, "_probe_fast", lambda port: pytest.fail("fast probe"))
    assert doctor.check_engine("smart", {})["probe"] == {"ok": True, "p": 8081}
    assert doctor.check_engine("fast", {"JEV_FAST_ENGINE": "llama"})["port"] == 8080


def test_check_engine_skips_the_probe_when_health_is_not_up(monkeypatch):
    monkeypatch.setattr(doctor, "_health_status", lambda port: "down")
    got = doctor.check_engine("smart", {})
    assert got["status"] == "down" and got["probe"] is None


def test_check_engine_rejects_an_unknown_engine():
    with pytest.raises(ValueError):
        doctor.check_engine("nope", {})


# --- sample_once -----------------------------------------------------------


def test_sample_once_returns_one_record_per_engine():
    ticks = iter([1.0, 1.25, 2.0, 2.5])
    rows = availability.sample_once(_ok_check, lambda: T0, lambda: next(ticks))
    assert [r["engine"] for r in rows] == ["fast", "smart"]
    assert rows[0] == {
        "ts": T0.isoformat(),
        "engine": "fast",
        "health_ok": True,
        "probe_ok": True,
        "latency_ms": 250.0,
        "error": None,
    }
    assert rows[1]["latency_ms"] == 500.0


def test_sample_once_records_a_down_engine_without_latency():
    def check(name, env=os.environ):
        return {"port": 1, "status": "down", "probe": None}

    rows = availability.sample_once(check, lambda: T0, lambda: 0.0)
    assert rows[0]["health_ok"] is False
    assert rows[0]["probe_ok"] is False
    assert rows[0]["latency_ms"] is None
    assert rows[0]["error"] == "health: down"


def test_sample_once_records_a_failed_probe_detail_truncated():
    def check(name, env=os.environ):
        return {"port": 1, "status": "up", "probe": {"ok": False, "detail": "x" * 999}}

    row = availability.sample_once(check, lambda: T0, lambda: 0.0)[0]
    assert row["health_ok"] is True and row["probe_ok"] is False
    assert row["latency_ms"] == 0.0
    assert len(row["error"]) == availability.MAX_ERROR_CHARS


def test_sample_once_survives_a_check_that_raises():
    def check(name, env=os.environ):
        raise RuntimeError("boom")

    rows = availability.sample_once(check, lambda: T0, lambda: 0.0)
    assert all(not r["health_ok"] and not r["probe_ok"] for r in rows)
    assert "RuntimeError: boom" in rows[0]["error"]


def test_sample_once_treats_a_probe_with_no_ok_key_as_failed():
    def check(name, env=os.environ):
        return {"port": 1, "status": "up", "probe": {}}

    row = availability.sample_once(check, lambda: T0, lambda: 0.0)[0]
    assert row["probe_ok"] is False and row["error"] == "probe failed"


# --- append / rotate -------------------------------------------------------


def test_append_creates_parent_and_writes_json_lines(tmp_path):
    path = tmp_path / "sub" / "a.jsonl"
    availability.append_rows(str(path), [_row(0), _row(1)])
    lines = path.read_text().splitlines()
    assert [json.loads(line)["ts"] for line in lines] == [
        _row(0)["ts"],
        _row(1)["ts"],
    ]


def test_append_rotates_to_one_backup_when_over_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(availability, "MAX_LOG_BYTES", 100)
    path = tmp_path / "a.jsonl"
    path.write_text("x" * 101)
    (tmp_path / "a.jsonl.1").write_text("old backup")
    availability.append_rows(str(path), [_row(0)])
    assert (tmp_path / "a.jsonl.1").read_text() == "x" * 101
    assert len(path.read_text().splitlines()) == 1


def test_append_does_not_rotate_under_the_cap(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_text("x" * 50)
    availability.append_rows(str(path), [_row(0)])
    assert not (tmp_path / "a.jsonl.1").exists()


def test_the_size_cap_is_a_few_megabytes():
    assert 1 * 1024 * 1024 <= availability.MAX_LOG_BYTES <= 10 * 1024 * 1024


# --- run loop --------------------------------------------------------------


def test_run_once_samples_one_time_and_returns(tmp_path):
    path = tmp_path / "a.jsonl"
    stop = threading.Event()
    n = availability.run(
        str(path), 60, once=True, stop=stop, check=_ok_check, now=lambda: T0
    )
    assert n == 1 and len(path.read_text().splitlines()) == 2


def test_run_loops_until_stopped_and_waits_the_interval(tmp_path):
    stop = threading.Event()
    waits = []

    def wait(seconds):
        waits.append(seconds)
        if len(waits) == 3:
            stop.set()
        return stop.is_set()

    n = availability.run(
        str(tmp_path / "a.jsonl"),
        7,
        stop=stop,
        check=_ok_check,
        now=lambda: T0,
        wait=wait,
    )
    assert n == 3 and waits == [7, 7, 7]


def test_run_survives_a_write_failure(tmp_path, monkeypatch):
    stop = threading.Event()

    def boom(path, rows):
        raise OSError("disk full")

    monkeypatch.setattr(availability, "append_rows", boom)
    n = availability.run(
        str(tmp_path / "a.jsonl"),
        1,
        stop=stop,
        check=_ok_check,
        wait=lambda s: stop.set() or True,
    )
    assert n == 1


def test_run_does_not_sample_when_already_stopped(tmp_path):
    stop = threading.Event()
    stop.set()
    assert availability.run(str(tmp_path / "a"), 1, stop=stop, check=_ok_check) == 0


def test_signal_handlers_set_the_stop_event_and_are_restored():
    stop = threading.Event()
    before = signal.getsignal(signal.SIGTERM)
    restore = availability.install_stop_handlers(stop)
    signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    assert stop.is_set()
    restore()
    assert signal.getsignal(signal.SIGTERM) == before


# --- parsing ---------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,delta",
    [("30s", 30), ("15m", 900), ("24h", 86400), ("2d", 172800)],
)
def test_parse_since_accepts_durations(raw, delta):
    assert availability.parse_since(raw, T0) == T0 - timedelta(seconds=delta)


def test_parse_since_accepts_iso_and_assumes_utc_for_naive():
    assert availability.parse_since("2026-01-01T00:00:00", T0) == datetime(
        2026, 1, 1, tzinfo=timezone.utc
    )
    assert availability.parse_since("2026-01-01T00:00:00Z", T0).tzinfo is not None


def test_parse_since_rejects_garbage():
    with pytest.raises(ValueError):
        availability.parse_since("yesterday-ish", T0)


# --- reading + summary -----------------------------------------------------


def test_read_rows_skips_corrupt_partial_and_malformed_lines(tmp_path):
    path = tmp_path / "a.jsonl"
    good = json.dumps(_row(0))
    path.write_text(
        "\n".join(
            [
                good,
                "{not json",
                '{"ts": "2026-01-01T12:00:00+00:00", "engine": "fa',
                json.dumps([1, 2]),
                json.dumps({"ts": "bad", "engine": "fast"}),
                json.dumps({"ts": T0.isoformat()}),
                "",
            ]
        )
    )
    assert len(availability.read_rows(str(path))) == 1


def test_read_rows_includes_the_backup_first_and_tolerates_missing(tmp_path):
    path = tmp_path / "a.jsonl"
    assert availability.read_rows(str(path)) == []
    _write(tmp_path / "a.jsonl.1", [_row(0)])
    _write(path, [_row(1)])
    rows = availability.read_rows(str(path))
    assert [r["ts"] for r in rows] == [_row(0)["ts"], _row(1)["ts"]]


def test_read_rows_accepts_a_z_suffixed_timestamp(tmp_path):
    path = tmp_path / "a.jsonl"
    row = _row(0)
    row["ts"] = "2026-01-01T12:00:00Z"
    _write(path, [row])
    assert len(availability.read_rows(str(path))) == 1


def test_summarize_computes_uptime_probe_success_and_percentiles():
    rows = [_row(i, latency=float(i + 1) * 10) for i in range(10)]
    rows[3] = _row(3, probe=False, latency=None, error="bad")
    s = availability.summarize(_parsed(rows), None)["fast"]
    assert s["samples"] == 10
    assert s["uptime_pct"] == 100.0
    assert s["probe_success_pct"] == 90.0
    assert s["latency_p50_ms"] == 60.0
    assert s["latency_p95_ms"] == 100.0
    assert s["outages"] == 1
    assert s["longest_outage_s"] == 60.0


def test_outages_count_consecutive_failures_and_measure_to_recovery():
    rows = [
        _row(0),
        _row(1, health=False, probe=False, latency=None),
        _row(2, health=False, probe=False, latency=None),
        _row(5),
        _row(6, probe=False),
        _row(7),
        _row(8, health=False, probe=False, latency=None),
    ]
    s = availability.summarize(_parsed(rows), None)["fast"]
    assert s["outages"] == 3
    assert s["longest_outage_s"] == 240.0  # minute 1 to recovery at minute 5
    assert s["uptime_pct"] == pytest.approx(4 / 7 * 100, abs=0.01)


def test_an_open_ended_outage_lasts_until_its_last_failed_sample():
    rows = [
        _row(0),
        _row(1, health=False, probe=False),
        _row(4, health=False, probe=False),
    ]
    s = availability.summarize(_parsed(rows), None)["fast"]
    assert s["outages"] == 1 and s["longest_outage_s"] == 180.0


def test_summarize_filters_by_since_and_separates_engines():
    rows = [_row(0), _row(10), _row(10, engine="smart", health=False, probe=False)]
    result = availability.summarize(_parsed(rows), T0 + timedelta(minutes=5))
    assert result["fast"]["samples"] == 1
    assert result["smart"]["uptime_pct"] == 0.0


def test_summarize_with_no_latency_reports_none_and_empty_engines_are_omitted():
    rows = [_row(0, health=False, probe=False, latency=None)]
    result = availability.summarize(_parsed(rows), None)
    assert result["fast"]["latency_p50_ms"] is None
    assert result["fast"]["latency_p95_ms"] is None
    assert "smart" not in result
    assert availability.summarize(_parsed([]), None) == {}


def test_percentile_of_empty_is_none():
    assert availability.percentile([], 50) is None
    assert availability.percentile([5.0], 95) == 5.0


# --- CLI -------------------------------------------------------------------


def test_cli_sample_once_writes_the_log(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(doctor, "check_engine", _ok_check)
    path = tmp_path / "a.jsonl"
    assert availability.main(["sample", "--once", "--log", str(path)]) == 0
    assert len(path.read_text().splitlines()) == 2


def test_cli_sample_rejects_a_non_positive_interval(tmp_path, capsys):
    with pytest.raises(SystemExit):
        availability.main(["sample", "--interval", "0", "--log", str(tmp_path / "x")])


def test_cli_sample_default_log_path_is_under_home_jev(monkeypatch):
    monkeypatch.setenv("HOME", "/nonexistent-home")
    assert (
        availability.default_log_path() == "/nonexistent-home/.jev/availability.jsonl"
    )


def test_cli_sample_runs_until_a_signal_and_exits_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor, "check_engine", _ok_check)

    def fake_run(path, interval, once=False, stop=None, **kw):
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        assert stop.is_set()
        return 1

    monkeypatch.setattr(availability, "run", fake_run)
    assert availability.main(["sample", "--log", str(tmp_path / "a")]) == 0


def test_cli_summary_text_and_json(tmp_path, capsys):
    path = tmp_path / "a.jsonl"
    _write(path, [_row(0), _row(1, health=False, probe=False, latency=None)])
    assert availability.main(["summary", "--log", str(path)]) == 0
    text = capsys.readouterr().out
    assert "fast" in text and "uptime" in text and "outages" in text
    assert availability.main(["summary", "--log", str(path), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["fast"]["samples"] == 2


def test_cli_summary_since_and_empty_and_bad_since(tmp_path, capsys):
    path = tmp_path / "a.jsonl"
    assert availability.main(["summary", "--log", str(path)]) == 0
    assert "no samples" in capsys.readouterr().out
    _write(path, [_row(0)])
    assert availability.main(["summary", "--log", str(path), "--since", "1h"]) == 0
    assert availability.main(["summary", "--log", str(path), "--since", "zzz"]) == 2
    assert "invalid --since" in capsys.readouterr().err


def test_cli_summary_text_shows_dashes_for_missing_latency(tmp_path, capsys):
    path = tmp_path / "a.jsonl"
    _write(path, [_row(0, health=False, probe=False, latency=None)])
    availability.main(["summary", "--log", str(path)])
    assert "-" in capsys.readouterr().out


def test_module_main_guard_runs(tmp_path, monkeypatch):
    import runpy

    monkeypatch.setattr(
        "sys.argv", ["availability", "summary", "--log", str(tmp_path / "n")]
    )
    with pytest.raises(SystemExit) as e:
        runpy.run_module("jev_mcp.availability", run_name="__main__")
    assert e.value.code == 0


def test_parse_row_assumes_utc_for_a_naive_timestamp():
    row = availability._parse_row(
        json.dumps({"ts": "2026-01-01T12:00:00", "engine": "fast"})
    )
    assert row["_when"] == T0


def test_positive_float_rejects_non_positive():
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        availability._positive_float("-1")


def test_cli_sample_accepts_a_custom_interval(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor, "check_engine", _ok_check)
    args = ["sample", "--once", "--interval", "5", "--log", str(tmp_path / "a")]
    assert availability.main(args) == 0
