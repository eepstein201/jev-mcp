"""The server follows the served model and the active engine profile (phase 1, lane P1-c).

Model identity is the daemon's `/props` `model_path`, read with a TTL; the gate and the
smart-tier prompt style come from the active profile; responses name the profile and the
two served models. Every test runs against the fake llama-server or mocks at the HTTP
boundary: no real daemon is started, stopped or queried.
"""

import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from jev_mcp import app, daemon_provider, engine_profiles, router_config
from jev_mcp.daemon_provider import UNKNOWN_MODEL, DaemonProvider
from jev_mcp.engine_profiles import DEFAULT_GATE, ModelSpec, Profile
from jev_mcp.kev_provider import KevProvider
from jev_mcp.provider import NoulQuestion
from jev_mcp.routing_provider import RoutingProvider
from tests.fake_llama_server import FakeLlamaServer, load_recorded

CHAT = "/v1/chat/completions"
SYSTEMONE = "/v1/systemone"
RECORDED = load_recorded()
SMART_MODEL = RECORDED["props"]["smart"]["model_path"]
KEV_MODEL = RECORDED["props"]["fast"]["model_path"]
SMART_FILE = SMART_MODEL.rsplit("/", 1)[1]
KEV_FILE = KEV_MODEL.rsplit("/", 1)[1]
OTHER_MODEL = "/models/Other-Model-Q4_K_M.gguf"
TTL = daemon_provider.SERVED_MODEL_TTL_SECONDS


class Clock:
    """An injectable monotonic clock."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _spec(tier, filename, *, style="plain", reasoning="auto"):
    return ModelSpec(
        name=filename.lower().removesuffix(".gguf"),
        tier=tier,
        repo="org/repo",
        revision="a" * 40,
        filename=filename,
        sha256="b" * 64,
        parallel=4 if tier == "fast" else 2,
        ctx_per_slot=16384,
        reasoning=reasoning,
        style=style,
    )


def _profile(*, gate=DEFAULT_GATE, smart_style="sandbox", smart_file=SMART_FILE):
    return Profile(
        name="lean",
        fast=_spec("fast", KEV_FILE),
        smart=_spec("smart", smart_file, style=smart_style),
        gate=gate,
        eval_only=False,
    )


@pytest.fixture
def use_profile(monkeypatch):
    """Make `engine_profiles.active_profile` answer the given profile (or None)."""

    def _use(profile):
        monkeypatch.setattr(engine_profiles, "active_profile", lambda *a, **k: profile)

    return _use


@pytest.fixture
def broken_profiles(monkeypatch):
    """The resolver as it is on this lane's base: every function raises."""

    def _raise(*args, **kwargs):
        raise NotImplementedError

    monkeypatch.setattr(engine_profiles, "active_profile", _raise)
    monkeypatch.setattr(daemon_provider, "_profile_last_error", None)
    monkeypatch.setattr(daemon_provider, "_profile_last_warned_at", float("-inf"))


@pytest.fixture(autouse=True)
def fresh_prior_cache(monkeypatch):
    monkeypatch.setattr(DaemonProvider, "_prior_cache", {})
    monkeypatch.setattr(DaemonProvider, "_get_daemon_pid", lambda self: "pid-1")


@pytest.fixture
def smart_daemon():
    with FakeLlamaServer() as running:
        yield running


@pytest.fixture
def fast_daemon():
    with FakeLlamaServer() as running:
        running.set_scenario("normal", tier="fast")
        yield running


def _props_gets(daemon):
    return [r for r in daemon.requests if r[:2] == ("GET", "/props")]


def _ask(provider, key="q"):
    return provider.evaluate_batch("a state", [NoulQuestion(key=key, prompt="Is it?")])


# --- the profile helper -----------------------------------------------------------------


def test_helper_returns_none_and_rate_limits_the_same_warning(
    broken_profiles, caplog, monkeypatch
):
    clock = Clock()
    monkeypatch.setattr(daemon_provider, "_PROFILE_CLOCK", clock)
    with caplog.at_level(logging.WARNING, logger="jev_mcp.daemon_provider"):
        for _ in range(3):
            assert daemon_provider.active_engine_profile() is None
            clock.advance(daemon_provider.PROFILE_TTL_SECONDS + 1)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_helper_warns_again_when_the_failure_reason_changes(monkeypatch, caplog):
    clock = Clock()
    monkeypatch.setattr(daemon_provider, "_PROFILE_CLOCK", clock)
    monkeypatch.setattr(daemon_provider, "_profile_last_error", None)
    monkeypatch.setattr(daemon_provider, "_profile_last_warned_at", float("-inf"))
    errors = iter(
        [NotImplementedError(), ValueError("bad marker"), ValueError("bad marker")]
    )

    def _active(*args, **kwargs):
        raise next(errors)

    monkeypatch.setattr(engine_profiles, "active_profile", _active)
    with caplog.at_level(logging.WARNING, logger="jev_mcp.daemon_provider"):
        for _ in range(3):
            daemon_provider.active_engine_profile()
            clock.advance(daemon_provider.PROFILE_TTL_SECONDS + 1)

    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(messages) == 2
    assert "bad marker" in messages[1]


def test_helper_warns_again_after_the_resolver_recovers_and_fails(monkeypatch, caplog):
    clock = Clock()
    monkeypatch.setattr(daemon_provider, "_PROFILE_CLOCK", clock)
    monkeypatch.setattr(daemon_provider, "_profile_last_error", None)
    monkeypatch.setattr(daemon_provider, "_profile_last_warned_at", float("-inf"))
    outcomes = iter([ValueError("x"), _profile(), ValueError("x")])

    def _active(*args, **kwargs):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(engine_profiles, "active_profile", _active)
    with caplog.at_level(logging.WARNING, logger="jev_mcp.daemon_provider"):
        for _ in range(3):
            daemon_provider.active_engine_profile()
            clock.advance(daemon_provider.PROFILE_TTL_SECONDS + 1)

    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 2


def test_helper_repeats_an_unchanged_warning_after_the_interval(monkeypatch, caplog):
    clock = Clock()
    monkeypatch.setattr(daemon_provider, "_PROFILE_CLOCK", clock)
    monkeypatch.setattr(daemon_provider, "_profile_last_error", None)
    monkeypatch.setattr(daemon_provider, "_profile_last_warned_at", float("-inf"))

    def _active(*args, **kwargs):
        raise ValueError("same")

    monkeypatch.setattr(engine_profiles, "active_profile", _active)
    with caplog.at_level(logging.WARNING, logger="jev_mcp.daemon_provider"):
        daemon_provider.active_engine_profile()
        clock.advance(daemon_provider.PROFILE_WARN_INTERVAL_SECONDS + 1)
        daemon_provider.active_engine_profile()

    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 2


def test_helper_returns_the_profile_the_resolver_names(use_profile):
    profile = _profile()
    use_profile(profile)

    assert daemon_provider.active_engine_profile() is profile


def test_helper_caches_the_profile_for_a_short_ttl(monkeypatch):
    clock = Clock()
    calls = []

    def _active(*args, **kwargs):
        calls.append(1)
        return _profile(gate=0.5 + len(calls) / 100)

    monkeypatch.setattr(engine_profiles, "active_profile", _active)
    monkeypatch.setattr(daemon_provider, "_PROFILE_CLOCK", clock)

    first = daemon_provider.active_engine_profile()
    clock.advance(daemon_provider.PROFILE_TTL_SECONDS - 0.1)
    assert daemon_provider.active_engine_profile() is first
    clock.advance(0.2)
    refreshed = daemon_provider.active_engine_profile()

    assert refreshed is not first
    assert len(calls) == 2


def test_active_gate_is_the_profile_gate_or_the_default(use_profile):
    use_profile(_profile(gate=0.7))
    assert daemon_provider.active_gate() == 0.7

    use_profile(None)
    assert daemon_provider.active_gate() == DEFAULT_GATE


def test_active_gate_is_the_default_when_the_resolver_raises(broken_profiles):
    assert daemon_provider.active_gate() == DEFAULT_GATE


# --- model identity follows /props ------------------------------------------------------


def test_the_smart_model_id_is_the_served_model_path(smart_daemon):
    provider = DaemonProvider(base_url=smart_daemon.url + CHAT)

    _ask(provider)

    assert provider.current_model_id == SMART_MODEL


def test_props_is_read_once_within_the_ttl(smart_daemon):
    clock = Clock()
    provider = DaemonProvider(base_url=smart_daemon.url + CHAT, clock=clock)

    _ask(provider)
    clock.advance(TTL - 1)
    _ask(provider)

    assert len(_props_gets(smart_daemon)) == 1


def test_the_model_id_follows_a_model_change_after_the_ttl(smart_daemon):
    smart_daemon.set_scenario("normal", alias="same")
    clock = Clock()
    provider = DaemonProvider(base_url=smart_daemon.url + CHAT, clock=clock)
    _ask(provider)

    # A constant alias: only the TTL (no detected swap) decides when the id is re-read.
    smart_daemon.set_scenario("normal", model=OTHER_MODEL, alias="same")
    clock.advance(TTL - 1)
    _ask(provider)
    assert provider.current_model_id == SMART_MODEL  # still inside the TTL

    clock.advance(2)
    _ask(provider)

    assert provider.current_model_id == OTHER_MODEL
    assert len(_props_gets(smart_daemon)) == 2


def test_an_alias_in_the_chat_reply_does_not_change_the_model_id(smart_daemon):
    smart_daemon.set_scenario("normal", alias="qwen2.5-7b")
    provider = DaemonProvider(base_url=smart_daemon.url + CHAT)

    _ask(provider)

    assert provider.current_model_id == SMART_MODEL


def test_an_alias_does_not_change_the_temperature_key(smart_daemon):
    config = router_config.with_temperature(
        {"buckets": {}, "rules": []}, SMART_MODEL, 0.6
    )
    router_config.save_router_config(config)
    smart_daemon.set_scenario("normal", alias="some-other-alias")
    provider = DaemonProvider(base_url=smart_daemon.url + CHAT)

    _ask(provider)

    assert provider.fitted_temperature == 0.6
    assert router_config.model_key(provider.current_model_id) == SMART_FILE.lower()


def test_a_model_id_survives_a_failed_props_read(smart_daemon):
    clock = Clock()
    provider = DaemonProvider(base_url=smart_daemon.url + CHAT, clock=clock)
    _ask(provider)

    smart_daemon.set_scenario("loading")
    clock.advance(TTL + 1)
    provider.evaluate_batch("a state", [NoulQuestion(key="q", prompt="Is it?")])

    assert provider.current_model_id == SMART_MODEL


def test_a_prior_is_cached_when_both_replies_name_the_same_model(smart_daemon):
    provider = DaemonProvider(base_url=smart_daemon.url + CHAT)

    _ask(provider)

    assert len(DaemonProvider._prior_cache) == 1


def test_a_prior_is_not_cached_when_the_reply_names_no_model(smart_daemon):
    smart_daemon.set_scenario("missing_model")
    provider = DaemonProvider(base_url=smart_daemon.url + CHAT)

    _ask(provider)

    assert provider.current_model_id == SMART_MODEL  # /props still names it
    assert DaemonProvider._prior_cache == {}


def test_a_prior_is_not_cached_across_a_model_change_between_the_two_requests(
    smart_daemon,
):
    smart_daemon.set_scenario("swap_after", n=1)
    provider = DaemonProvider(base_url=smart_daemon.url + CHAT)

    _ask(provider)

    assert DaemonProvider._prior_cache == {}


def test_the_fast_model_id_is_the_served_model_path(fast_daemon):
    provider = KevProvider(base_url=fast_daemon.url + SYSTEMONE)

    provider.evaluate_batch("a state", [NoulQuestion(key="q", prompt="Is it?")])

    assert provider.current_model_id == KEV_MODEL


def test_the_fast_model_id_refreshes_after_the_ttl(fast_daemon):
    clock = Clock()
    provider = KevProvider(base_url=fast_daemon.url + SYSTEMONE, clock=clock)
    question = [NoulQuestion(key="q", prompt="Is it?")]
    provider.evaluate_batch("a state", question)

    fast_daemon.set_scenario("normal", tier="fast", model=OTHER_MODEL)
    clock.advance(TTL + 1)
    provider.evaluate_batch("a state", question)

    assert provider.current_model_id == OTHER_MODEL


def test_the_fast_model_id_is_unknown_when_the_daemon_is_down():
    provider = KevProvider(base_url="http://127.0.0.1:1/v1/systemone")

    assert provider.evaluate_batch("a state", [NoulQuestion(key="q", prompt="?")]) == {}
    assert provider.current_model_id == UNKNOWN_MODEL


# --- plain or sandbox follows the served smart model's style ---------------------------


def _prompt_for(use_profile, profile, served=SMART_MODEL):
    use_profile(profile)
    provider = DaemonProvider()
    provider.current_model_id = served
    sent = []

    def _logprobs(prompt, targets):
        sent.append(prompt)
        return {"true": -0.1, "false": -2.5}, served

    with patch.object(provider, "get_logprobs", side_effect=_logprobs):
        _ask(provider)
    return sent[0]


def test_a_matching_plain_model_gets_the_plain_prompt(use_profile):
    prompt = _prompt_for(use_profile, _profile(smart_style="plain"))

    assert "strict classification model" in prompt
    assert "<payload_" not in prompt


def test_a_matching_sandbox_model_gets_the_sandbox_prompt(use_profile):
    prompt = _prompt_for(use_profile, _profile(smart_style="sandbox"))

    assert "evaluation sandbox" in prompt


def test_a_profile_without_a_filename_match_gets_the_sandbox_prompt(use_profile):
    profile = _profile(smart_style="plain", smart_file="Different-Q4_K_M.gguf")

    assert "evaluation sandbox" in _prompt_for(use_profile, profile)


def test_no_profile_gets_the_sandbox_prompt(use_profile):
    assert "evaluation sandbox" in _prompt_for(use_profile, None)


def test_a_failing_resolver_gets_the_sandbox_prompt(broken_profiles):
    provider = DaemonProvider()
    provider.current_model_id = SMART_MODEL
    sent = []

    def _logprobs(prompt, targets):
        sent.append(prompt)
        return {"true": -0.1, "false": -2.5}, SMART_MODEL

    with patch.object(provider, "get_logprobs", side_effect=_logprobs):
        _ask(provider)

    assert "evaluation sandbox" in sent[0]


def test_an_unknown_served_model_gets_the_sandbox_prompt(use_profile):
    prompt = _prompt_for(use_profile, _profile(smart_style="plain"), UNKNOWN_MODEL)

    assert "evaluation sandbox" in prompt


def test_the_model_name_alone_no_longer_selects_the_plain_prompt(use_profile):
    # The old rule keyed on a "0.5b" substring of the model id.
    prompt = _prompt_for(use_profile, None, served="/m/Qwen2.5-0.5B-Instruct.gguf")

    assert "evaluation sandbox" in prompt


def test_the_prior_cache_separates_plain_from_sandbox(use_profile):
    plain = _prompt_for(use_profile, _profile(smart_style="plain"))
    sandbox = _prompt_for(use_profile, _profile(smart_style="sandbox"))

    assert plain != sandbox


# --- the gate ----------------------------------------------------------------------------


def _routing(fast_prob, smart_calls):
    router = RoutingProvider()
    fast = {"q": {"probabilities": {"true": fast_prob, "false": 1 - fast_prob}}}
    smart = {"q": {"probabilities": {"true": 0.99, "false": 0.01}, "noul": 0.99}}

    def _smart(state, questions):
        smart_calls.append(1)
        return smart

    router.fast_provider = SimpleNamespace(
        evaluate_batch=lambda state, questions: fast,
        check_token_limit=lambda state: 10,
        max_tokens=65536,
        current_model_id=KEV_MODEL,
    )
    router.smart_provider = SimpleNamespace(
        evaluate_batch=_smart, max_tokens=8192, current_model_id=SMART_MODEL
    )
    return router


def test_the_router_escalates_below_the_default_gate_without_a_profile(use_profile):
    use_profile(None)
    smart_calls = []
    result = _routing(0.8, smart_calls).evaluate_batch(
        "s", [NoulQuestion(key="q", prompt="?")]
    )

    assert smart_calls == [1]
    assert result["q"]["served_by"] == "smart"


def test_a_lower_profile_gate_keeps_the_fast_answer(use_profile):
    use_profile(_profile(gate=0.7))
    smart_calls = []
    result = _routing(0.8, smart_calls).evaluate_batch(
        "s", [NoulQuestion(key="q", prompt="?")]
    )

    assert smart_calls == []
    assert result["q"]["served_by"] == "fast"


def test_a_higher_profile_gate_escalates_a_fast_answer_the_default_keeps(use_profile):
    use_profile(_profile(gate=0.95))
    smart_calls = []

    _routing(0.9, smart_calls).evaluate_batch("s", [NoulQuestion(key="q", prompt="?")])

    assert smart_calls == [1]


def _generative_lint(use_profile, gate, noul):
    from jev_mcp.tools import evaluate

    use_profile(_profile(gate=gate) if gate is not None else None)
    provider = MagicMock()
    provider.evaluate_batch.return_value = {"q_0": {"noul": noul}}
    with patch("jev_mcp.app.provider", provider):
        return evaluate._semantic_lint_findings(
            [NoulQuestion(key="q", prompt="Write a summary")]
        )


def test_the_semantic_lint_uses_the_profile_gate(use_profile):
    assert _generative_lint(use_profile, 0.7, 0.8) != []
    assert _generative_lint(use_profile, 0.95, 0.9) == []


def test_the_semantic_lint_uses_the_default_gate_without_a_profile(use_profile):
    assert _generative_lint(use_profile, None, 0.8) == []
    assert _generative_lint(use_profile, None, 0.9) != []


def _rule_result(use_profile, gate, noul):
    from jev_mcp import server

    use_profile(_profile(gate=gate) if gate is not None else None)
    router_config.save_router_config(
        {
            "buckets": {"b1": "small", "b3": "big"},
            "rules": [{"condition": "touches many files", "target": "b3"}],
        }
    )
    provider = MagicMock()
    provider.evaluate_batch.side_effect = lambda state, questions: {
        questions[0].key: {
            "noul": noul,
            "score": "1",
            "probabilities": {"1": 1.0},
        }
    }
    with patch("jev_mcp.app.RoutingProvider", return_value=provider):
        return json.loads(
            server.jev_determine_best_model(task_description="task", estimated_tokens=0)
        )


def test_a_rule_triggers_at_the_profile_gate(use_profile):
    assert _rule_result(use_profile, 0.7, 0.8)["status"] == "RULE_TRIGGERED"
    assert _rule_result(use_profile, 0.95, 0.9)["status"] == "EVALUATED"


def test_a_rule_uses_the_default_gate_without_a_profile(use_profile):
    assert _rule_result(use_profile, None, 0.8)["status"] == "EVALUATED"
    assert _rule_result(use_profile, None, 0.9)["status"] == "RULE_TRIGGERED"


# --- responses name the profile and the served models -----------------------------------


def _routing_ids(fast, smart):
    return SimpleNamespace(
        fast_provider=SimpleNamespace(current_model_id=fast),
        smart_provider=SimpleNamespace(current_model_id=smart),
    )


def test_identity_names_the_profile_and_the_two_model_keys(use_profile):
    use_profile(_profile())

    identity = app.engine_identity(_routing_ids(KEV_MODEL, SMART_MODEL))

    assert identity == {
        "profile": "lean",
        "fast": KEV_FILE.lower(),
        "smart": SMART_FILE.lower(),
    }


def test_identity_is_null_for_engines_that_reported_nothing(use_profile):
    use_profile(None)

    identity = app.engine_identity(_routing_ids(UNKNOWN_MODEL, ""))

    assert identity == {"profile": None, "fast": None, "smart": None}


def test_identity_tolerates_a_provider_without_engines(use_profile):
    use_profile(None)

    assert app.engine_identity(MagicMock()) == {
        "profile": None,
        "fast": None,
        "smart": None,
    }
    assert app.engine_identity(object()) == {
        "profile": None,
        "fast": None,
        "smart": None,
    }


def test_identity_is_null_profile_when_the_resolver_raises(broken_profiles):
    assert app.engine_identity(_routing_ids(KEV_MODEL, None))["profile"] is None


def test_the_evaluate_response_carries_the_identity(use_profile):
    from jev_mcp import server

    use_profile(_profile())
    provider = MagicMock()
    provider.check_token_limit.return_value = 10
    provider.max_tokens = 1000
    provider.current_model_id = "hybrid_routed"
    provider.fast_provider = SimpleNamespace(current_model_id=KEV_MODEL)
    provider.smart_provider = SimpleNamespace(current_model_id=UNKNOWN_MODEL)
    provider.evaluate_batch.return_value = {
        "q": {"noul": 0.9, "probabilities": {"true": 0.9, "false": 0.1}}
    }

    with patch("jev_mcp.app.provider", provider):
        res = json.loads(
            server.jev_evaluate_batch(
                state={"a": 1},
                questions=[NoulQuestion(key="q", prompt="Is it on?")],
                auto_apply_fixes=False,
            )
        )

    assert res["status"].startswith("SUCCESS")
    assert (res["profile"], res["fast"], res["smart"]) == (
        "lean",
        KEV_FILE.lower(),
        None,
    )


def test_the_evaluate_error_response_carries_the_identity_with_nulls(use_profile):
    from jev_mcp import server

    use_profile(None)
    provider = MagicMock()
    provider.check_token_limit.return_value = 10
    provider.max_tokens = 1000
    provider.evaluate_batch.side_effect = RuntimeError("engines down")

    with patch("jev_mcp.app.provider", provider):
        res = json.loads(
            server.jev_evaluate_batch(
                state={"a": 1},
                questions=[NoulQuestion(key="q", prompt="Is it on?")],
                auto_apply_fixes=False,
            )
        )

    assert res["status"] == "ERROR"
    assert (res["profile"], res["fast"], res["smart"]) == (None, None, None)


def test_determine_best_model_carries_the_identity(use_profile):
    from jev_mcp import server

    use_profile(_profile())
    router_config.save_router_config({"buckets": {"b1": "small"}, "rules": []})
    provider = MagicMock()
    provider.fast_provider = SimpleNamespace(current_model_id=KEV_MODEL)
    provider.smart_provider = SimpleNamespace(current_model_id=SMART_MODEL)
    provider.evaluate_batch.return_value = {
        "complexity": {"score": "1", "probabilities": {"1": 0.9}}
    }

    with patch("jev_mcp.app.RoutingProvider", return_value=provider):
        res = json.loads(
            server.jev_determine_best_model(task_description="task", estimated_tokens=0)
        )

    assert res["status"] == "EVALUATED"
    assert (res["profile"], res["fast"], res["smart"]) == (
        "lean",
        KEV_FILE.lower(),
        SMART_FILE.lower(),
    )


def test_a_triggered_rule_carries_the_identity(use_profile):
    use_profile(_profile(gate=0.7))
    res = _rule_result(use_profile, 0.7, 0.8)

    assert res["status"] == "RULE_TRIGGERED"
    assert res["profile"] == "lean"


def test_agent_handoff_carries_the_identity(use_profile):
    from jev_mcp import server

    use_profile(_profile())
    provider = MagicMock()
    provider.fast_provider = SimpleNamespace(current_model_id=KEV_MODEL)
    provider.smart_provider = SimpleNamespace(current_model_id=SMART_MODEL)
    provider.evaluate_batch.return_value = {
        "best_agent": {"choice": "A", "probabilities": {"A": 0.9}}
    }

    with patch("jev_mcp.app.RoutingProvider", return_value=provider):
        res = json.loads(server.jev_agent_handoff("task", {"A": "does things"}))

    assert res["status"] == "HANDOFF_RECOMMENDATION"
    assert (res["profile"], res["fast"], res["smart"]) == (
        "lean",
        KEV_FILE.lower(),
        SMART_FILE.lower(),
    )


# --- a swap inside the TTL (fix round 1) ------------------------------------------------

OTHER_KEY = router_config.model_key(OTHER_MODEL)
TEMP_A, TEMP_B = 0.6, 2.5


def _save_temperatures():
    config = router_config.with_temperature(
        {"buckets": {}, "rules": []}, SMART_MODEL, TEMP_A
    )
    router_config.save_router_config(
        router_config.with_temperature(config, OTHER_MODEL, TEMP_B)
    )


def _set_pid(monkeypatch, pid):
    monkeypatch.setattr(DaemonProvider, "_get_daemon_pid", lambda self: pid)


def test_a_new_daemon_pid_forces_a_props_read_inside_the_ttl(smart_daemon, monkeypatch):
    _save_temperatures()
    smart_daemon.set_scenario("normal", alias="same")
    provider = DaemonProvider(base_url=smart_daemon.url + CHAT, clock=Clock())
    _ask(provider)
    assert provider.fitted_temperature == TEMP_A

    smart_daemon.set_scenario("normal", model=OTHER_MODEL, alias="same")
    _set_pid(monkeypatch, "pid-2")
    _ask(provider)

    assert provider.current_model_id == OTHER_MODEL
    assert provider.fitted_temperature == TEMP_B
    assert len(_props_gets(smart_daemon)) == 2


def test_an_unchanged_pid_does_not_force_a_props_read(smart_daemon):
    provider = DaemonProvider(base_url=smart_daemon.url + CHAT, clock=Clock())
    smart_daemon.set_scenario("normal", alias="same")

    _ask(provider)
    _ask(provider)

    assert len(_props_gets(smart_daemon)) == 1


def test_a_pid_that_could_not_be_found_is_not_a_swap(smart_daemon, monkeypatch):
    smart_daemon.set_scenario("normal", alias="same")
    provider = DaemonProvider(base_url=smart_daemon.url + CHAT, clock=Clock())
    _ask(provider)

    _set_pid(monkeypatch, "unknown_pid")
    _ask(provider)

    assert len(_props_gets(smart_daemon)) == 1


def test_a_changed_reply_alias_forces_a_props_read_before_the_temperature(
    smart_daemon,
):
    _save_temperatures()
    smart_daemon.set_scenario("normal", alias="model-a")
    provider = DaemonProvider(base_url=smart_daemon.url + CHAT, clock=Clock())
    _ask(provider)
    assert provider.fitted_temperature == TEMP_A

    smart_daemon.set_scenario("normal", model=OTHER_MODEL, alias="model-b")
    result = _ask(provider)

    assert provider.current_model_id == OTHER_MODEL
    assert provider.fitted_temperature == TEMP_B


def test_a_swap_that_props_cannot_name_drops_the_old_model(smart_daemon, monkeypatch):
    _save_temperatures()
    smart_daemon.set_scenario("normal", alias="same")
    provider = DaemonProvider(base_url=smart_daemon.url + CHAT, clock=Clock())
    _ask(provider)

    smart_daemon.set_scenario("loading")
    _set_pid(monkeypatch, "pid-2")
    _ask(provider)

    assert provider.current_model_id == UNKNOWN_MODEL
    assert provider.fitted_temperature == router_config.DEFAULT_TEMPERATURE


def test_the_prior_is_not_cached_for_the_question_that_straddles_a_swap(
    smart_daemon,
):
    smart_daemon.set_scenario("normal", alias="model-a")
    provider = DaemonProvider(base_url=smart_daemon.url + CHAT, clock=Clock())
    _ask(provider)
    cached_before = dict(DaemonProvider._prior_cache)

    smart_daemon.set_scenario("normal", model=OTHER_MODEL, alias="model-b")
    _ask(provider, key="q2")

    assert DaemonProvider._prior_cache == cached_before


def test_reported_smart_model_sees_a_restart_inside_the_ttl(smart_daemon, monkeypatch):
    smart_daemon.set_scenario("normal", alias="same")
    provider = DaemonProvider(base_url=smart_daemon.url + CHAT, clock=Clock())
    _ask(provider)

    smart_daemon.set_scenario("normal", model=OTHER_MODEL, alias="same")
    _set_pid(monkeypatch, "pid-2")

    assert app.reported_smart_model(provider)[0] == OTHER_MODEL


def test_calibration_after_a_swap_is_saved_under_the_new_models_key(
    smart_daemon, monkeypatch
):
    from jev_mcp import server

    smart_daemon.set_scenario("normal", alias="same")
    smart = DaemonProvider(base_url=smart_daemon.url + CHAT, clock=Clock())
    _ask(smart)
    routing = SimpleNamespace(smart_provider=smart)
    monkeypatch.setattr(app, "provider", routing)

    smart_daemon.set_scenario("normal", model=OTHER_MODEL, alias="same")
    _set_pid(monkeypatch, "pid-2")
    dataset = [{"state": {"t": str(i)}, "expected": i % 2 == 0} for i in range(6)]
    out = json.loads(
        server.jev_calibrate_threshold(
            dataset=dataset,
            question=NoulQuestion(key="q", prompt="Is it?"),
            apply_platt_scaling="always",
            save_to_config=True,
        )
    )

    assert out["status"] == "CALIBRATION_COMPLETE"
    assert out["model"] == OTHER_KEY
    stored = router_config.load_router_config().get("temperatures", {})
    assert SMART_FILE.lower() not in stored
    if out["temperature_saved"]:
        assert OTHER_KEY in stored


def test_the_kev_provider_re_reads_props_when_its_reply_model_changes(fast_daemon):
    fast_daemon.set_scenario("normal", tier="fast", alias="a")
    provider = KevProvider(base_url=fast_daemon.url + SYSTEMONE, clock=Clock())
    question = [NoulQuestion(key="q", prompt="Is it?")]
    provider.evaluate_batch("a state", question)

    fast_daemon.set_scenario("normal", tier="fast", alias="b", model=OTHER_MODEL)
    provider.evaluate_batch("a state", question)

    assert provider.current_model_id == OTHER_MODEL


def test_the_kev_provider_drops_its_id_when_a_swap_cannot_be_named(fast_daemon):
    fast_daemon.set_scenario("normal", tier="fast", alias="a")
    provider = KevProvider(base_url=fast_daemon.url + SYSTEMONE, clock=Clock())
    question = [NoulQuestion(key="q", prompt="Is it?")]
    provider.evaluate_batch("a state", question)

    fast_daemon.set_scenario("normal", tier="fast", alias="b", props={"model_path": 42})
    provider.evaluate_batch("a state", question)

    assert provider.current_model_id == UNKNOWN_MODEL


# --- failed /props reads are remembered briefly (M4) ------------------------------------


def test_a_failed_props_read_is_not_repeated_inside_the_negative_ttl(smart_daemon):
    smart_daemon.set_scenario("normal", props={"model_path": 42})
    clock = Clock()
    provider = DaemonProvider(base_url=smart_daemon.url + CHAT, clock=clock)

    _ask(provider)
    clock.advance(daemon_provider.FAILED_PROPS_TTL_SECONDS - 1)
    _ask(provider)
    assert len(_props_gets(smart_daemon)) == 1

    smart_daemon.set_scenario("normal")
    clock.advance(2)
    _ask(provider)

    assert len(_props_gets(smart_daemon)) == 2
    assert provider.current_model_id == SMART_MODEL


def test_the_kev_provider_remembers_a_failed_props_read(fast_daemon):
    fast_daemon.set_scenario("normal", tier="fast", props={})
    provider = KevProvider(base_url=fast_daemon.url + SYSTEMONE, clock=Clock())
    question = [NoulQuestion(key="q", prompt="Is it?")]

    provider.evaluate_batch("a state", question)
    provider.evaluate_batch("a state", question)

    assert len(_props_gets(fast_daemon)) == 1


# --- the handoff error path and the response model label (M2, M6) -----------------------


def test_agent_handoff_error_carries_the_identity_with_nulls(use_profile):
    from jev_mcp import server

    use_profile(None)
    provider = MagicMock()
    provider.evaluate_batch.side_effect = RuntimeError("engines down")

    with patch("jev_mcp.app.RoutingProvider", return_value=provider):
        res = json.loads(server.jev_agent_handoff("task", {"A": "does things"}))

    assert res["status"] == "ERROR"
    assert "engines down" in res["message"]
    assert (res["profile"], res["fast"], res["smart"]) == (None, None, None)


def test_the_evaluate_response_model_is_a_file_name_not_a_path(use_profile):
    from jev_mcp import server

    use_profile(None)
    provider = MagicMock()
    provider.check_token_limit.return_value = 10
    provider.max_tokens = 1000
    provider.current_model_id = "/Users/someone/models/Qwen/Qwen2.5-7B-Q4_K_M.gguf"
    provider.evaluate_batch.return_value = {
        "q": {"noul": 0.9, "probabilities": {"true": 0.9, "false": 0.1}}
    }

    with patch("jev_mcp.app.provider", provider):
        res = json.loads(
            server.jev_evaluate_batch(
                state={"a": 1},
                questions=[NoulQuestion(key="q", prompt="Is it on?")],
                auto_apply_fixes=False,
            )
        )

    assert res["model"] == "qwen2.5-7b-q4_k_m.gguf"
    assert "/" not in json.dumps(res["model"])


def test_the_evaluate_response_model_keeps_the_routers_own_labels():
    from jev_mcp.tools.evaluate import _model_label

    assert _model_label("hybrid_routed") == "hybrid_routed"
    assert _model_label(None) == "unknown_model"
    assert _model_label(MagicMock()) == "unknown_model"


def test_the_routing_bypass_reports_the_smart_model_after_its_batch(use_profile):
    use_profile(None)
    router = RoutingProvider()
    seen = {}

    def _smart(state, questions):
        router.smart_provider.current_model_id = SMART_MODEL  # learnt during the batch
        return {"q": {"probabilities": {"true": 0.9, "false": 0.1}}}

    router.fast_provider = SimpleNamespace(
        check_token_limit=lambda state: 99999,
        max_tokens=65536,
        current_model_id=KEV_MODEL,
    )
    router.smart_provider = SimpleNamespace(
        evaluate_batch=_smart, max_tokens=8192, current_model_id=UNKNOWN_MODEL
    )

    router.evaluate_batch("s" * 100, [NoulQuestion(key="q", prompt="?")])
    seen["model"] = router.current_model_id

    assert seen["model"] == SMART_MODEL
