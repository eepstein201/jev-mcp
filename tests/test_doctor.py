import importlib.metadata
import json
import os
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jev_mcp.doctor import collect_report, main, problems, render


GOOD_PROBE = {
    "choices": [
        {
            "logprobs": {
                "content": [
                    {
                        "token": "True",
                        "logprob": -0.05,
                        "top_logprobs": [
                            {"token": "True", "logprob": -0.05},
                            {"token": "False", "logprob": -3.2},
                        ],
                    }
                ]
            }
        }
    ]
}


def _healthy_model(url):
    """The model a healthy engine on this port serves: its tier's model in the active
    profile, which is what `make start` launches. With the repo profile table present
    and no active marker that resolves to the default profile, so a fixture daemon
    serving anything else would be (correctly) reported as a mismatch. Without a
    profile any file does - nothing is expected then."""
    from jev_mcp import engine_profiles

    profile = engine_profiles.active_profile()
    if profile is None:
        return "/models/Served-Q4.gguf"
    port = url.rsplit(":", 1)[1].split("/", 1)[0]
    spec = profile.smart if port == "8081" else profile.fast
    return f"/models/{spec.filename}"


@pytest.fixture
def urlopen_ok():
    """The engines answer /health, /props and a scoring probe at the HTTP boundary."""

    def _urlopen(request, timeout=None):
        url = request.full_url
        if url.endswith("/props"):
            body = {"model_path": _healthy_model(url)}
        elif url.endswith("/health"):
            body = {"status": "ok"}
        elif url.endswith("/v1/systemone"):
            keys = json.loads(request.data)["questions"]
            body = {"answers": {k: {"noul": 0.9, "type": "noul"} for k in keys}}
        else:
            body = GOOD_PROBE
        response = MagicMock()
        response.read.return_value = json.dumps(body).encode()
        context = MagicMock()
        context.__enter__.return_value = response
        return context

    with patch("urllib.request.urlopen", side_effect=_urlopen):
        yield


@pytest.fixture
def urlopen_refused():
    """No engine answers at the HTTP boundary."""
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("connection refused"),
    ):
        yield


@pytest.fixture
def tmp_router_config():
    from jev_mcp import router_config

    router_config.write_json_atomic(
        router_config.ROUTER_CONFIG_PATH, {"buckets": {}, "rules": []}
    )
    return router_config.ROUTER_CONFIG_PATH


# --- the plan's Step 1 tests --------------------------------------------------


def test_an_engine_that_does_not_answer_is_reported_down_and_is_a_problem(
    urlopen_refused,
):
    report = collect_report(env={})
    assert report["engines"]["fast"]["status"] == "down"
    assert any("fast engine" in p for p in problems(report))


def test_both_engines_up_and_nothing_else_wrong_is_no_problem(
    urlopen_ok, tmp_router_config
):
    assert problems(collect_report(env={"JEV_ALLOWED_ROOTS": "/tmp/proj"})) == []


def test_jwks_without_an_audience_is_a_problem(urlopen_ok):
    report = collect_report(env={"JEV_MCP_JWKS_URL": "https://idp.example/jwks.json"})
    assert any("JEV_MCP_JWT_AUDIENCE" in p for p in problems(report))


def test_no_secret_value_appears_in_any_output(urlopen_ok, capsys):
    env = {
        "JEV_MCP_API_KEY": "s3cr3t-value-xyz",
        "JEV_MCP_JWT_SECRET": "another-s3cr3t-value-0123456789abcdef",
    }
    report = collect_report(env=env)
    text = render(report) + json.dumps(report)
    assert "s3cr3t" not in text
    assert report["webhook_auth"]["set"] == ["JEV_MCP_API_KEY", "JEV_MCP_JWT_SECRET"]


def test_a_home_working_directory_with_no_explicit_root_is_a_problem(
    urlopen_ok, monkeypatch
):
    monkeypatch.setenv("HOME", "/Users/test")
    monkeypatch.setattr(os, "getcwd", lambda: "/Users/test")
    monkeypatch.delenv("JEV_ALLOWED_ROOTS", raising=False)
    assert any("JEV_ALLOWED_ROOTS" in p for p in problems(collect_report(env={})))


def test_main_exits_one_when_there_is_a_problem(urlopen_refused, capsys):
    assert main([]) == 1


def test_json_output_parses(urlopen_ok, capsys):
    main(["--json"])
    json.loads(capsys.readouterr().out)


# --- the review-focus pin: no secret in --json either ------------------------


def test_json_output_never_contains_a_secret_value(urlopen_ok, capsys, monkeypatch):
    monkeypatch.setenv("JEV_MCP_API_KEY", "s3cr3t-value-xyz")
    main(["--json"])
    assert "s3cr3t" not in capsys.readouterr().out


# --- the remaining problem and report branches -------------------------------


def test_a_router_config_that_is_present_but_not_valid_json_is_a_problem(urlopen_ok):
    from jev_mcp import router_config

    path = Path(router_config.ROUTER_CONFIG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")

    report = collect_report(env={})

    assert report["router_config"]["exists"] is True
    assert report["router_config"]["valid_json"] is False
    assert any("not valid JSON" in p for p in problems(report))


def test_render_lists_the_problems_and_an_empty_auth_section(urlopen_refused):
    report = collect_report(env={})
    text = render(report)
    assert "problems" in text
    assert all(problem in text for problem in problems(report))
    assert "set : none" in text  # the auth section names variables, so never values


def test_an_unusable_port_value_falls_back_to_the_default(urlopen_ok):
    report = collect_report(env={"JEV_FAST_PORT": "not-a-number"})
    assert report["engines"]["fast"]["port"] == 8080


def test_versions_omit_a_distribution_that_is_not_installed(urlopen_ok):
    real_version = importlib.metadata.version

    def _missing(name, *args, **kwargs):
        if name == "torch":
            raise importlib.metadata.PackageNotFoundError(name)
        return real_version(name, *args, **kwargs)

    with patch("importlib.metadata.version", side_effect=_missing):
        report = collect_report(env={})

    assert "torch" not in report["versions"]
    assert "mcp" in report["versions"]


# --- the server dispatches the subcommand ------------------------------------


def test_server_dispatches_the_doctor_subcommand(urlopen_refused, capsys, monkeypatch):
    import sys

    from jev_mcp import server

    monkeypatch.setattr(sys, "argv", ["jev-mcp", "doctor", "--json"])
    with pytest.raises(SystemExit) as excinfo:
        server.main()

    assert excinfo.value.code == 1
    json.loads(capsys.readouterr().out)


# --- engine profiles: served model, expected model, probe and loading ------------------

from jev_mcp import daemon_provider, engine_profiles  # noqa: E402
from jev_mcp.engine_profiles import ModelSpec, Profile  # noqa: E402
from tests.fake_llama_server import FakeLlamaServer, load_recorded  # noqa: E402

_PROPS = load_recorded()["props"]
KEV_FILE = _PROPS["fast"]["model_path"].rsplit("/", 1)[1]
SMART_FILE = _PROPS["smart"]["model_path"].rsplit("/", 1)[1]


def _spec(tier, filename):
    return ModelSpec(
        name=filename.lower(),
        tier=tier,
        repo="org/repo",
        revision="a" * 40,
        filename=filename,
        sha256="b" * 64,
        parallel=2,
        ctx_per_slot=16384,
        reasoning="auto",
        style="plain",
    )


def _profile(fast_file=KEV_FILE, smart_file=SMART_FILE):
    return Profile(
        name="lean",
        fast=_spec("fast", fast_file),
        smart=_spec("smart", smart_file),
        gate=0.85,
        eval_only=False,
    )


@pytest.fixture
def daemons(monkeypatch, tmp_router_config):
    """A fake fast and a fake smart daemon, wired to doctor through the port variables."""
    with FakeLlamaServer() as fast, FakeLlamaServer() as smart:
        fast.set_scenario("normal", tier="fast")
        monkeypatch.setenv("JEV_FAST_PORT", fast.url.rsplit(":", 1)[1])
        monkeypatch.setenv("JEV_SMART_PORT", smart.url.rsplit(":", 1)[1])
        monkeypatch.setenv("JEV_ALLOWED_ROOTS", "/tmp/proj")
        yield SimpleNamespace(fast=fast, smart=smart)


def _use_profile(monkeypatch, profile):
    """The resolver names this profile (None: the marker names a profile the table lacks)."""
    table = SimpleNamespace(get=lambda name: profile)
    monkeypatch.setattr(engine_profiles, "read_active_name", lambda *a, **k: "lean")
    monkeypatch.setattr(engine_profiles, "load_profiles", lambda *a, **k: table)


def _report():
    return collect_report(env=os.environ)


from types import SimpleNamespace  # noqa: E402


def test_a_matching_profile_is_reported_and_is_no_problem(daemons, monkeypatch, capsys):
    _use_profile(monkeypatch, _profile())

    report = _report()

    assert report["profile"] == "lean"
    assert report["engines"]["fast"]["served_model"] == KEV_FILE
    assert report["engines"]["fast"]["expected_model"] == KEV_FILE
    assert report["engines"]["smart"]["served_model"] == SMART_FILE
    assert problems(report) == []
    assert main([]) == 0
    text = capsys.readouterr().out
    assert f"served : {KEV_FILE}" in text
    assert f"expected : {SMART_FILE}" in text


def test_a_served_model_that_is_not_the_expected_one_is_a_problem(
    daemons, monkeypatch, capsys
):
    _use_profile(monkeypatch, _profile())
    daemons.smart.set_scenario("wrong_model")

    report = _report()

    mismatch = [p for p in problems(report) if "smart engine" in p]
    assert len(mismatch) == 1
    assert SMART_FILE in mismatch[0]
    assert "Unexpected" in mismatch[0]
    assert main([]) == 1
    assert "make start PROFILE=lean" in capsys.readouterr().out


def test_without_a_profile_nothing_is_expected_and_nothing_mismatches(
    daemons, monkeypatch
):
    _use_profile(monkeypatch, None)

    report = _report()

    assert report["profile"] is None
    assert report["engines"]["smart"]["expected_model"] is None
    assert report["engines"]["smart"]["served_model"] == SMART_FILE
    assert problems(report) == []


def test_a_resolver_that_raises_is_treated_as_no_profile(daemons, monkeypatch):
    def _raise(*args, **kwargs):
        raise NotImplementedError

    monkeypatch.setattr(engine_profiles, "read_active_name", _raise)

    report = _report()

    assert report["profile"] is None
    assert report["profile_error"] is None
    assert problems(report) == []


def test_a_loading_engine_is_a_problem_that_says_so(daemons, monkeypatch):
    _use_profile(monkeypatch, _profile())
    daemons.smart.set_scenario("loading")

    report = _report()

    assert report["engines"]["smart"]["status"] == "loading"
    found = [p for p in problems(report) if "smart engine" in p]
    assert len(found) == 1
    assert "loading" in found[0]
    assert "down" not in found[0]
    assert report["engines"]["smart"]["probe"] is None


def test_a_probe_whose_first_token_is_not_a_target_is_flagged(daemons, monkeypatch):
    _use_profile(monkeypatch, _profile())
    daemons.smart.set_scenario("think_first")

    report = _report()

    assert report["engines"]["smart"]["probe"]["ok"] is False
    found = [p for p in problems(report) if "probe" in p]
    assert len(found) == 1
    assert "smart engine" in found[0]
    assert "first token" in found[0]


def test_a_probe_with_too_little_target_mass_is_flagged(daemons, monkeypatch):
    _use_profile(monkeypatch, _profile())
    daemons.smart.set_scenario("normal", first_token_probs={"True": 0.6, "Maybe": 0.4})

    report = _report()

    probe = report["engines"]["smart"]["probe"]
    assert probe["ok"] is False
    assert probe["target_mass"] == pytest.approx(0.6)
    assert any("smart engine" in p and "0.6" in p for p in problems(report))


def test_a_passing_probe_reports_its_target_mass(daemons, monkeypatch):
    _use_profile(monkeypatch, _profile())

    probe = _report()["engines"]["smart"]["probe"]

    assert probe["ok"] is True
    assert probe["target_mass"] > 0.9


def test_a_probe_that_cannot_complete_is_a_problem(daemons, monkeypatch):
    _use_profile(monkeypatch, _profile())
    daemons.smart.set_scenario("context_overflow")

    report = _report()

    assert report["engines"]["smart"]["probe"]["ok"] is False
    assert any("probe" in p and "smart engine" in p for p in problems(report))


def test_an_engine_that_names_no_model_when_one_is_expected_is_a_problem(
    daemons, monkeypatch
):
    _use_profile(monkeypatch, _profile())
    daemons.smart.set_scenario("normal", props={"total_slots": 2})

    report = _report()

    assert report["engines"]["smart"]["served_model"] is None
    assert any("smart engine" in p and "/props" in p for p in problems(report))


def test_doctor_prints_nothing_outside_main(daemons, monkeypatch, capsys):
    _use_profile(monkeypatch, _profile())

    report = _report()
    render(report)
    problems(report)

    assert capsys.readouterr().out == ""


# --- the fast engine is probed through SystemOne, never through chat ------------------


def test_the_fast_probe_scores_through_systemone_and_never_uses_chat(
    daemons, monkeypatch
):
    _use_profile(monkeypatch, _profile())

    report = _report()

    assert report["engines"]["fast"]["probe"]["ok"] is True
    paths = [path for _, path, _ in daemons.fast.requests]
    assert "/v1/systemone" in paths
    assert "/v1/chat/completions" not in paths
    # The smart engine still gets the logprob probe.
    assert "/v1/chat/completions" in [p for _, p, _ in daemons.smart.requests]


def test_the_fakes_fast_chat_answers_like_the_live_kev_daemon(daemons):
    # Pins the fake's fidelity: a chat probe of the fast tier cannot pass.
    from jev_mcp.doctor import _probe

    probe = _probe(int(daemons.fast.url.rsplit(":", 1)[1]))

    assert probe["ok"] is False
    assert probe["first_token"] not in ("true", "false")


def test_a_fast_probe_without_an_answer_is_flagged(daemons, monkeypatch):
    _use_profile(monkeypatch, _profile())
    daemons.fast.set_scenario("wrong_kev_shape", tier="fast")

    report = _report()

    assert report["engines"]["fast"]["probe"]["ok"] is False
    assert any("fast engine" in p and "probe" in p for p in problems(report))


def test_a_fast_probe_whose_probabilities_do_not_sum_to_one_is_flagged(
    daemons, monkeypatch
):
    from unittest.mock import patch as _patch

    from jev_mcp.doctor import _probe_fast

    with _patch(
        "jev_mcp.doctor.KevProvider.evaluate_batch",
        return_value={"probe": {"probabilities": {"true": 0.2, "false": 0.2}}},
    ):
        probe = _probe_fast(1)

    assert probe["ok"] is False
    assert "sum" in probe["detail"]


def test_a_fast_probe_that_raises_is_flagged():
    from unittest.mock import patch as _patch

    from jev_mcp.doctor import _probe_fast

    with _patch(
        "jev_mcp.doctor.KevProvider.evaluate_batch", side_effect=RuntimeError("boom")
    ):
        assert _probe_fast(1)["ok"] is False


def test_a_non_kev_fast_engine_is_probed_with_the_logprob_probe(daemons, monkeypatch):
    # JEV_FAST_ENGINE=daemon serves the fast tier through chat completions.
    daemons.fast.set_scenario("normal")  # a chat-capable fake
    monkeypatch.setenv("JEV_FAST_ENGINE", "daemon")
    _use_profile(monkeypatch, _profile())

    assert _report()["engines"]["fast"]["probe"]["ok"] is True
    assert "/v1/chat/completions" in [p for _, p, _ in daemons.fast.requests]


# --- a broken profile is a problem, not "none" -----------------------------------------


def test_a_profile_error_is_a_problem_and_exit_one(daemons, monkeypatch, capsys):
    def _bad(*args, **kwargs):
        raise engine_profiles.ProfileError("active marker names 'x!'")

    monkeypatch.setattr(engine_profiles, "read_active_name", _bad)

    report = _report()

    assert report["profile"] is None
    assert "x!" in report["profile_error"]
    assert any("cannot be read" in p and "x!" in p for p in problems(report))
    assert main([]) == 1
    assert "profile error" in capsys.readouterr().out
