import io
import json
import urllib.error

import pytest

from tests.evals import webhook_latency as wl


def test_percentile_linear_interpolation_hand_computed():
    values = [1.0, 2.0, 3.0, 4.0]
    # rank = q/100 * (n-1): p50 -> 1.5 -> 2.5 ; p90 -> 2.7 -> 3.7 ; p0/p100 -> ends
    assert wl.percentile(values, 50) == pytest.approx(2.5)
    assert wl.percentile(values, 90) == pytest.approx(3.7)
    assert wl.percentile(values, 0) == 1.0
    assert wl.percentile(values, 100) == 4.0


def test_percentile_sorts_input_and_handles_single_and_empty():
    assert wl.percentile([3.0, 1.0, 2.0], 50) == 2.0
    assert wl.percentile([7.0], 95) == 7.0
    assert wl.percentile([], 50) is None


def test_summarize_hand_computed():
    s = wl.summarize([1.0, 2.0, 3.0, 4.0])
    assert s["count"] == 4
    assert s["mean"] == pytest.approx(2.5)
    assert s["p50"] == pytest.approx(2.5)
    assert s["p90"] == pytest.approx(3.7)
    assert s["p95"] == pytest.approx(3.85)
    assert s["max"] == 4.0


def test_summarize_empty():
    s = wl.summarize([])
    assert s["count"] == 0
    assert s["p50"] is None and s["mean"] is None and s["max"] is None


def _rec(lat, routing="archive_and_label", http=200, degraded=False, **kw):
    return {
        "latency_s": lat,
        "http_status": http,
        "status": "success",
        "suggested_routing": routing,
        "degraded": degraded,
        "error": None,
        "attempts": 1,
        "http_503_count": 1 if http == 503 else 0,
        **kw,
    }


def test_group_latencies_by_routing_skips_missing_routing():
    recs = [
        _rec(1.0, "a"),
        _rec(3.0, "a"),
        _rec(5.0, "b"),
        _rec(9.0, None, http=503),
    ]
    groups = wl.group_latencies(recs, "suggested_routing")
    assert groups == {"a": [1.0, 3.0], "b": [5.0]}


def test_build_report_shares_and_cold_start():
    recs = [
        _rec(10.0, "a"),
        _rec(2.0, "a", degraded=True),
        _rec(4.0, "b"),
        _rec(6.0, None, http=503),
    ]
    rep = wl.build_report(recs)
    assert rep["requests"] == 4
    assert rep["overall"]["max"] == 10.0
    assert rep["overall"]["mean"] == pytest.approx(5.5)
    assert rep["first_request_s"] == 10.0
    assert rep["rest"]["count"] == 3
    assert rep["rest"]["mean"] == pytest.approx(4.0)
    assert rep["degraded_share"] == pytest.approx(0.25)
    assert rep["http_503_share"] == pytest.approx(0.25)
    assert rep["error_share"] == 0.0
    assert rep["by_routing"]["a"]["count"] == 2
    assert rep["by_routing"]["b"]["p50"] == 4.0
    assert "escalated" in rep["fast_vs_escalated"]["note"].lower()


def test_build_report_errors_excluded_from_latency_but_counted():
    recs = [_rec(1.0), _rec(30.0, http=None, error="timeout", status=None)]
    rep = wl.build_report(recs)
    assert rep["requests"] == 2
    assert rep["error_share"] == 0.5
    assert rep["overall"]["count"] == 2  # a timeout is still a measured wait
    assert rep["errors"] == {"timeout": 1}


def test_build_report_empty():
    rep = wl.build_report([])
    assert rep["requests"] == 0
    assert rep["first_request_s"] is None
    assert rep["degraded_share"] == 0.0


def test_fast_vs_escalated_split_uses_degraded_flag():
    recs = [_rec(1.0), _rec(5.0, degraded=True), _rec(2.0)]
    split = wl.build_report(recs)["fast_vs_escalated"]
    assert split["degraded"]["count"] == 1
    assert split["not_degraded"]["count"] == 2


class FakeResponse:
    def __init__(self, body, status=200):
        self._body = json.dumps(body).encode()
        self.status = status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeClock:
    """Advances only when the fake urlopen or sleep says so."""

    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def _opener(clock, script):
    """script: list of (duration, outcome); outcome is a body dict, int (HTTP error) or Exception."""
    calls = []

    def urlopen(req, timeout=None):
        calls.append(req)
        duration, outcome = script[len(calls) - 1]
        clock.now += duration
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, int):
            raise urllib.error.HTTPError(
                req.full_url,
                outcome,
                "x",
                {},
                io.BytesIO(json.dumps({"detail": "retry"}).encode()),
            )
        return FakeResponse(outcome)

    urlopen.calls = calls
    return urlopen


ROW = {"subject": "s", "sender": "a@b.c", "body": "hello"}
OK = {"status": "success", "suggested_routing": "archive_and_label", "degraded": False}


def _send(clock, script, mode="none", key="sekret", context_id="ctx"):
    opener = _opener(clock, script)
    rec = wl.send_message(
        "http://127.0.0.1:1",
        key,
        context_id,
        ROW,
        attempt_mode=mode,
        timeout=5.0,
        urlopen=opener,
        clock=clock,
        sleep=clock.sleep,
    )
    return rec, opener


def test_send_message_success_records_latency_and_fields():
    clock = FakeClock()
    rec, opener = _send(clock, [(1.5, OK)])
    assert rec["latency_s"] == pytest.approx(1.5)
    assert rec["http_status"] == 200
    assert rec["status"] == "success"
    assert rec["suggested_routing"] == "archive_and_label"
    assert rec["degraded"] is False
    assert rec["error"] is None
    req = opener.calls[0]
    assert req.get_header("Authorization") == "Bearer sekret"
    assert req.get_header("X-triage-attempt") is None
    sent = json.loads(req.data)
    assert sent == {
        "context_id": "ctx",
        "subject": "s",
        "sender": "a@b.c",
        "body": "hello",
    }
    assert "sekret" not in json.dumps(rec)


def test_send_message_no_key_omits_authorization():
    clock = FakeClock()
    rec, opener = _send(clock, [(0.1, OK)], key="")
    assert opener.calls[0].get_header("Authorization") is None


def test_send_message_degraded_flag_read():
    clock = FakeClock()
    body = {**OK, "degraded": True, "status": "DEGRADED"}
    rec, _ = _send(clock, [(2.0, body)])
    assert rec["degraded"] is True
    assert rec["status"] == "DEGRADED"


def test_send_message_http_error_503_recorded_without_retry_in_none_mode():
    clock = FakeClock()
    rec, opener = _send(clock, [(0.5, 503)])
    assert rec["http_status"] == 503
    assert rec["http_503_count"] == 1
    assert rec["attempts"] == 1
    assert len(opener.calls) == 1


def test_send_message_connection_error_recorded():
    clock = FakeClock()
    rec, _ = _send(clock, [(0.2, urllib.error.URLError("refused"))])
    assert rec["http_status"] is None
    assert "refused" in rec["error"]
    assert rec["latency_s"] == pytest.approx(0.2)


def test_send_message_timeout_recorded():
    clock = FakeClock()
    rec, _ = _send(clock, [(5.0, TimeoutError("timed out"))])
    assert rec["http_status"] is None
    assert "timed out" in rec["error"]


def test_send_message_non_json_body_recorded_as_error():
    clock = FakeClock()

    def urlopen(req, timeout=None):
        clock.now += 1
        r = FakeResponse({})
        r._body = b"<html>"
        return r

    rec = wl.send_message(
        "http://x",
        "k",
        "c",
        ROW,
        attempt_mode="none",
        timeout=1,
        urlopen=urlopen,
        clock=clock,
        sleep=clock.sleep,
    )
    assert rec["http_status"] == 200
    assert "non-JSON" in rec["error"]
    assert rec["suggested_routing"] is None


def test_script_mode_retries_503_with_pause_and_attempt_headers():
    clock = FakeClock()
    script = [(1.0, 503), (1.0, 503), (1.0, OK)]
    rec, opener = _send(clock, script, mode="script")
    assert [r.get_header("X-triage-attempt") for r in opener.calls] == ["1", "2", "3"]
    assert rec["attempts"] == 3
    assert rec["http_503_count"] == 2
    assert rec["http_status"] == 200
    # 3 s of requests + 2 pauses of RETRY_PAUSE_S
    assert rec["latency_s"] == pytest.approx(3.0 + 2 * wl.RETRY_PAUSE_S)


def test_script_mode_stops_after_max_attempts():
    clock = FakeClock()
    script = [(1.0, 503)] * 3
    rec, opener = _send(clock, script, mode="script")
    assert len(opener.calls) == 3
    assert rec["http_status"] == 503
    assert rec["latency_s"] == pytest.approx(3.0 + 2 * wl.RETRY_PAUSE_S)


def test_script_mode_connection_error_does_not_retry():
    clock = FakeClock()
    rec, opener = _send(clock, [(0.1, OSError("down"))], mode="script")
    assert len(opener.calls) == 1
    assert rec["error"]


def test_constants_mirror_gas_script():
    assert wl.RETRY_PAUSE_S == 5.0
    assert wl.MAX_DEGRADED_ATTEMPTS == 3


def test_load_dataset_ignores_extra_keys_and_blank_lines(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(
        json.dumps({"subject": "a", "sender": "b", "body": "c", "label": 1})
        + "\n\n"
        + json.dumps({"subject": "x"})
        + "\n"
    )
    rows = wl.load_dataset(p)
    assert rows[0] == {"subject": "a", "sender": "b", "body": "c"}
    assert rows[1] == {"subject": "x", "sender": "", "body": ""}


def test_load_dataset_bad_line_names_line_number(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"subject": "a"}\nnot json\n')
    with pytest.raises(ValueError, match="line 2"):
        wl.load_dataset(p)


def test_run_all_repeats_and_never_aborts():
    clock = FakeClock()
    script = [
        (1.0, OK),
        (0.1, urllib.error.URLError("boom")),
        (1.0, OK),
        (1.0, OK),
    ]
    opener = _opener(clock, script)
    recs = wl.run_all(
        "http://x",
        "k",
        "c",
        [ROW, ROW],
        repeats=2,
        attempt_mode="none",
        timeout=1,
        urlopen=opener,
        clock=clock,
        sleep=clock.sleep,
    )
    assert len(recs) == 4
    assert recs[1]["error"]


def test_format_text_contains_key_lines():
    rep = wl.build_report([_rec(1.0, "a"), _rec(2.0, "b", http=503)])
    text = wl.format_text(rep)
    assert "p50" in text and "by suggested_routing" in text
    assert "first request" in text
    assert "503" in text


def test_format_text_empty_report():
    assert "0 requests" in wl.format_text(wl.build_report([]))


def test_main_json_output_and_no_key_leak(tmp_path, monkeypatch, capsys):
    data = tmp_path / "d.jsonl"
    data.write_text(json.dumps(ROW) + "\n")
    monkeypatch.setenv("JEV_MCP_API_KEY", "topsecretkey")
    clock = FakeClock()
    opener = _opener(clock, [(1.0, OK)])
    monkeypatch.setattr(wl.urllib.request, "urlopen", opener)
    monkeypatch.setattr(wl.time, "perf_counter", clock)
    rc = wl.main(["--url", "http://127.0.0.1:1", "--dataset", str(data), "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    assert json.loads(out)["requests"] == 1
    assert "topsecretkey" not in out


def test_main_text_output_and_missing_key_warns(tmp_path, monkeypatch, capsys):
    data = tmp_path / "d.jsonl"
    data.write_text(json.dumps(ROW) + "\n")
    monkeypatch.delenv("JEV_MCP_API_KEY", raising=False)
    clock = FakeClock()
    opener = _opener(clock, [(1.0, OK)])
    monkeypatch.setattr(wl.urllib.request, "urlopen", opener)
    monkeypatch.setattr(wl.time, "perf_counter", clock)
    rc = wl.main(["--url", "http://127.0.0.1:1", "--dataset", str(data)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "p50" in captured.out
    assert "JEV_MCP_API_KEY" in captured.err
    assert opener.calls[0].get_header("Authorization") is None
