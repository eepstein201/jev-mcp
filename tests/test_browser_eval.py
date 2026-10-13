from contextlib import nullcontext

import pytest

from tests.evals import browser_eval


@pytest.fixture
def no_page_server(monkeypatch):
    monkeypatch.setattr(browser_eval, "serve_pages", nullcontext)


def _agent_returning(monkeypatch, results):
    calls = []
    remaining = iter(results)

    def fake_run_browser_agent(url, goal, max_steps):
        calls.append(url)
        result = next(remaining)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(browser_eval, "run_browser_agent", fake_run_browser_agent)
    return calls


def test_an_error_result_is_a_failed_run_and_the_other_runs_still_execute(
    monkeypatch, capsys, no_page_server
):
    calls = _agent_returning(
        monkeypatch,
        [
            {"result": "done"},
            {"status": "ERROR", "message": "engine is not running"},
            {"result": "done"},
            {"result": "blocked"},
        ],
    )

    exit_code = browser_eval.main(["--cases", "T3", "--repeats", "4"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert len(calls) == 4
    assert "T3" in out and "2/4" in out and "50%" in out


def test_an_error_result_writes_the_case_and_its_message_to_stderr(
    monkeypatch, capsys, no_page_server
):
    _agent_returning(
        monkeypatch, [{"status": "ERROR", "message": "engine is not running"}]
    )

    browser_eval.main(["--cases", "T3", "--repeats", "1"])

    err = capsys.readouterr().err
    assert "T3" in err and "engine is not running" in err


def test_errored_runs_are_reported_apart_from_runs_that_did_not_finish(
    monkeypatch, capsys, no_page_server
):
    _agent_returning(
        monkeypatch,
        [
            {"status": "ERROR", "message": "engine is not running"},
            {"result": "blocked"},
            {"status": "ERROR", "message": "engine is not running"},
        ],
    )

    browser_eval.main(["--cases", "T3", "--repeats", "3"])

    out = capsys.readouterr().out
    assert "0/3" in out
    assert "T3: 2 of 3 runs errored" in out


def test_an_exception_from_a_run_is_reported_as_an_errored_run(
    monkeypatch, capsys, no_page_server
):
    calls = _agent_returning(
        monkeypatch, [RuntimeError("browser went away"), {"result": "done"}]
    )

    browser_eval.main(["--cases", "D1", "--repeats", "2"])

    captured = capsys.readouterr()
    assert len(calls) == 2
    assert "1/2" in captured.out
    assert "D1: 1 of 2 runs errored" in captured.out
    assert "D1" in captured.err and "browser went away" in captured.err


def test_no_errored_line_when_every_run_reached_a_result(
    monkeypatch, capsys, no_page_server
):
    _agent_returning(monkeypatch, [{"result": "done"}, {"result": "blocked"}])

    browser_eval.main(["--cases", "S1", "--repeats", "2"])

    captured = capsys.readouterr()
    assert "errored" not in captured.out
    assert captured.err == ""
