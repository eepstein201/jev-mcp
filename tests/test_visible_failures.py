"""Bad or malformed engine answers are visible, never a normal SUCCESS.

The engine side runs against the loopback FakeLlamaServer scenarios; the tool side feeds
each changed tool a provider result that carries `escalation_failed`.
"""

import json
import logging
import math
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from jev_mcp import app, server
from jev_mcp.daemon_provider import DaemonProvider
from jev_mcp.kev_provider import KevProvider
from jev_mcp.provider import (
    ChoiceQuestion,
    EngineUnavailableError,
    NoulQuestion,
    ScoreQuestion,
    escalation_failures,
    mark_degraded,
    post_json,
)
from jev_mcp.routing_provider import RoutingProvider
from tests.fake_llama_server import FakeLlamaServer

CHAT = "/v1/chat/completions"
SYSTEMONE = "/v1/systemone"
STATE = "Email from shipping@example.com: Your order has shipped."


@pytest.fixture
def server_a():
    with FakeLlamaServer() as running:
        yield running


@pytest.fixture
def server_b():
    with FakeLlamaServer() as running:
        yield running


@pytest.fixture
def daemon(server_a, monkeypatch):
    """A DaemonProvider on a fake llama-server with an empty, private prior cache."""
    monkeypatch.setattr(DaemonProvider, "_prior_cache", {})
    monkeypatch.setattr(DaemonProvider, "_get_daemon_pid", lambda self: "1")
    return DaemonProvider(base_url=server_a.url + CHAT)


def _port(running: FakeLlamaServer) -> str:
    return running.url.rsplit(":", 1)[1]


# --- 1. a first token that is not a target is no answer, not an all-zero map -------------


def test_a_think_first_answer_is_no_answer_and_the_target_mass_is_logged(
    daemon, server_a, caplog
):
    server_a.set_scenario("think_first")
    caplog.set_level(logging.WARNING, logger="jev_mcp.daemon_provider")

    results = daemon.evaluate_batch(STATE, [NoulQuestion(key="is_ship", prompt="?")])

    assert results == {}
    warning = " ".join(r.getMessage() for r in caplog.records if r.levelno >= 30)
    assert "is_ship" in warning
    assert "mass" in warning
    assert "/models/" in warning  # the model id the daemon reported


def test_a_low_probability_answer_above_the_gate_is_still_an_answer(daemon, server_a):
    server_a.set_scenario("normal", first_token_probs={"True": 0.08, "False": 0.07})

    results = daemon.evaluate_batch(STATE, [NoulQuestion(key="q", prompt="?")])

    probabilities = results["q"]["probabilities"]
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_normalize_returns_an_empty_map_below_the_mass_gate():
    provider = DaemonProvider(base_url="http://127.0.0.1:1/v1/chat/completions")
    low = {"true": math.log(0.01), "false": math.log(0.01)}
    prior = {"true": math.log(0.5), "false": math.log(0.5)}

    assert provider._normalize_logprobs(low, ["true", "false"], prior) == {}


# --- 4. an HTTP error keeps its status and body -----------------------------------------


def test_a_context_overflow_logs_the_400_status_and_body_and_is_unanswered(
    daemon, server_a, caplog
):
    server_a.set_scenario("context_overflow")
    caplog.set_level(logging.ERROR, logger="jev_mcp.daemon_provider")

    results = daemon.evaluate_batch(STATE, [NoulQuestion(key="q", prompt="?")])

    assert results == {}
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "400" in logged
    assert "exceed_context_size_error" in logged


def test_post_json_keeps_the_status_code_and_a_truncated_body_in_the_error(server_a):
    server_a.set_scenario("context_overflow")

    with pytest.raises(urllib.error.HTTPError) as caught:
        post_json(server_a.url + CHAT, {"messages": []})

    assert caught.value.code == 400
    assert "exceed_context_size_error" in str(caught.value)


def test_post_json_truncates_a_long_error_body(monkeypatch):
    import io

    body = ("x" * 5000).encode()

    def raise_http(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 500, "Server Error", {}, io.BytesIO(body)
        )

    monkeypatch.setattr("urllib.request.urlopen", raise_http)

    with pytest.raises(urllib.error.HTTPError) as caught:
        post_json("http://127.0.0.1:1/x", {})

    assert caught.value.code == 500
    assert len(str(caught.value)) < 700


def test_post_json_survives_an_error_body_that_cannot_be_read(monkeypatch):
    class Unreadable:
        def read(self, *args):
            raise OSError("connection reset")

        def close(self):
            pass

    def raise_http(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 503, "Unavailable", {}, Unreadable())

    monkeypatch.setattr("urllib.request.urlopen", raise_http)

    with pytest.raises(urllib.error.HTTPError) as caught:
        post_json("http://127.0.0.1:1/x", {})

    assert caught.value.code == 503


# --- 6. the prior is never cached under an unknown or changed model ----------------------


def test_a_prior_is_not_cached_under_unknown_model(daemon, server_a):
    server_a.set_scenario("missing_model")

    results = daemon.evaluate_batch(STATE, [NoulQuestion(key="q", prompt="?")])

    assert set(results) == {"q"}  # the answer is still used for this batch
    assert DaemonProvider._prior_cache == {}


def test_a_prior_is_not_cached_when_the_model_changes_between_the_two_requests(
    daemon, server_a
):
    server_a.set_scenario(
        "swap_after", n=1, model_a="/models/a.gguf", model_b="/models/b.gguf"
    )

    results = daemon.evaluate_batch(STATE, [NoulQuestion(key="q", prompt="?")])

    assert set(results) == {"q"}
    assert DaemonProvider._prior_cache == {}


def test_a_prior_from_the_same_model_in_both_responses_is_cached(daemon, server_a):
    results = daemon.evaluate_batch(STATE, [NoulQuestion(key="q", prompt="?")])

    assert set(results) == {"q"}
    assert len(DaemonProvider._prior_cache) == 1


def test_a_prior_fetch_that_reports_no_model_is_not_cached(daemon, monkeypatch):
    answer = ({"true": math.log(0.8), "false": math.log(0.2)}, "m1")
    prior = ({"true": math.log(0.5), "false": math.log(0.5)}, "unknown_model")
    monkeypatch.setattr(daemon, "get_logprobs", MagicMock(side_effect=[answer, prior]))

    results = daemon.evaluate_batch(STATE, [NoulQuestion(key="q", prompt="?")])

    assert set(results) == {"q"}
    assert DaemonProvider._prior_cache == {}


# --- 5. a Kev answer missing its required key is no answer -------------------------------


def test_a_kev_answer_in_the_wrong_shape_is_no_answer(server_a):
    server_a.set_scenario("wrong_kev_shape")
    kev = KevProvider(base_url=server_a.url + SYSTEMONE)
    questions = [
        NoulQuestion(key="n", prompt="?"),
        ChoiceQuestion(key="c", prompt="?", options=["a", "b"]),
        ScoreQuestion(key="s", prompt="?", labels=["low", "high"]),
    ]

    assert kev.evaluate_batch(STATE, questions) == {}


@pytest.mark.parametrize(
    "question, answer",
    [
        (NoulQuestion(key="q", prompt="?"), {}),
        (NoulQuestion(key="q", prompt="?"), {"noul": None}),
        (NoulQuestion(key="q", prompt="?"), {"noul": "high"}),
        (ChoiceQuestion(key="q", prompt="?", options=["a"]), {}),
        (ChoiceQuestion(key="q", prompt="?", options=["a"]), {"confidence": 0.9}),
        (ScoreQuestion(key="q", prompt="?", labels=["a", "b"]), {}),
        (ScoreQuestion(key="q", prompt="?", labels=["a", "b"]), {"probabilities": {}}),
        (ScoreQuestion(key="q", prompt="?", labels=["a", "b"]), {"probabilities": [1]}),
    ],
)
def test_parse_answer_returns_an_empty_dict_for_a_missing_required_key(
    question, answer
):
    kev = KevProvider(base_url="http://127.0.0.1:1/v1/systemone")

    assert kev._parse_answer(question, answer) == {}


def test_parse_answer_still_reads_a_complete_answer():
    kev = KevProvider(base_url="http://127.0.0.1:1/v1/systemone")

    noul = kev._parse_answer(NoulQuestion(key="q", prompt="?"), {"noul": 0.0})
    choice = kev._parse_answer(
        ChoiceQuestion(key="q", prompt="?", options=["a", "b"]),
        {"choice": "a", "probabilities": {"a": 0.7, "b": 0.3}},
    )

    assert noul["noul"] == 0.0  # a real zero is an answer
    assert choice["choice"] == "a"


def test_a_kev_answer_that_is_not_an_object_is_no_answer():
    kev = KevProvider(base_url="http://127.0.0.1:1/v1/systemone")

    with patch(
        "jev_mcp.kev_provider.post_json",
        return_value={"answers": {"q": "yes", "r": ["no"]}},
    ):
        results = kev.evaluate_batch(
            STATE,
            [NoulQuestion(key="q", prompt="?"), NoulQuestion(key="r", prompt="?")],
        )

    assert results == {}


def test_compact_context_keeps_every_chunk_when_the_fast_answer_is_empty(
    server_a, monkeypatch
):
    server_a.set_scenario("wrong_kev_shape")
    fast = KevProvider(base_url=server_a.url + SYSTEMONE)
    shared = MagicMock(spec=["fast_provider", "evaluate_batch"])
    shared.fast_provider = fast
    monkeypatch.setattr(app, "provider", shared)

    res = json.loads(
        server.jev_compact_context(
            state={"text": "alpha\n\nbeta"}, goal="g", confidence_threshold=0.3
        )
    )

    assert res["kept_chunks"] == res["original_chunks"] == 2
    assert res["status"] == "ENGINE_UNAVAILABLE"


# --- 2. routing marks who answered and what failed to escalate ---------------------------


@pytest.fixture
def engines(server_a, server_b, monkeypatch):
    """fast = Kev on server_a, smart = llama-server on server_b."""
    monkeypatch.setenv("JEV_FAST_PORT", _port(server_a))
    monkeypatch.setenv("JEV_SMART_PORT", _port(server_b))
    monkeypatch.setenv("JEV_FAST_ENGINE", "kev")
    monkeypatch.setattr(DaemonProvider, "_prior_cache", {})
    monkeypatch.setattr(DaemonProvider, "_get_daemon_pid", lambda self: "1")
    return server_a, server_b


def test_a_confident_fast_answer_is_served_by_fast(engines):
    fast, _ = engines
    fast.set_scenario("normal", noul_true=0.97)

    results = RoutingProvider().evaluate_batch(
        STATE, [NoulQuestion(key="q", prompt="?")]
    )

    assert results["q"]["served_by"] == "fast"
    assert "escalation_failed" not in results["q"]


def test_an_escalated_answer_is_served_by_smart(engines):
    fast, smart = engines
    fast.set_scenario("normal", noul_true=0.5)
    smart.set_scenario("normal", first_token_probs={"True": 0.9, "False": 0.1})

    results = RoutingProvider().evaluate_batch(
        STATE, [NoulQuestion(key="q", prompt="?")]
    )

    assert results["q"]["served_by"] == "smart"
    assert results["q"]["escalated"] is True
    assert "escalation_failed" not in results["q"]


def test_a_smart_engine_that_does_not_answer_marks_the_fast_answer_failed(engines):
    fast, smart = engines
    fast.set_scenario("normal", noul_true=0.5)
    smart.set_scenario("think_first")

    results = RoutingProvider().evaluate_batch(
        STATE, [NoulQuestion(key="q", prompt="?")]
    )

    assert results["q"]["escalation_failed"] is True
    assert results["q"]["served_by"] == "fast"
    assert results["q"]["noul"] == pytest.approx(
        0.5
    )  # the fast answer is still returned


def test_an_all_zero_fast_answer_with_a_silent_smart_engine_raises(engines):
    # An all-zero map is silence, not a confident zero: same rule as an empty map.
    fast, smart = engines
    fast.set_scenario("normal", score_probs=[0.0, 0.0, 0.0])
    smart.set_scenario("think_first")

    with pytest.raises(EngineUnavailableError):
        RoutingProvider().evaluate_batch(
            STATE,
            [ScoreQuestion(key="rank", prompt="?", labels=["low", "mid", "high"])],
        )


def test_only_the_question_that_failed_to_escalate_is_marked(engines):
    fast, smart = engines
    fast.set_scenario("normal", noul_true=0.5)
    smart.set_scenario("context_overflow")

    questions = [
        NoulQuestion(key="a", prompt="?"),
        ChoiceQuestion(key="b", prompt="?", options=["x", "y"]),
    ]
    fast.set_scenario(
        "normal", noul_true=0.5, choice_probs={"x": 0.99, "y": 0.01}
    )  # b is confident

    results = RoutingProvider().evaluate_batch(STATE, questions)

    assert results["a"]["escalation_failed"] is True
    assert results["b"]["served_by"] == "fast"
    assert "escalation_failed" not in results["b"]


def test_a_large_state_goes_straight_to_smart_and_says_so(engines):
    _, smart = engines
    smart.set_scenario("normal", first_token_probs={"True": 0.9, "False": 0.1})
    big_state = "x" * 20000

    results = RoutingProvider().evaluate_batch(
        big_state, [NoulQuestion(key="q", prompt="?")]
    )

    assert results["q"]["served_by"] == "smart"


def test_no_answer_from_either_engine_still_raises(engines):
    from jev_mcp.provider import EngineUnavailableError

    fast, smart = engines
    fast.set_scenario("wrong_kev_shape")
    smart.set_scenario("think_first")

    with pytest.raises(EngineUnavailableError):
        RoutingProvider().evaluate_batch(STATE, [NoulQuestion(key="q", prompt="?")])


def test_a_routing_result_does_not_alter_the_fast_provider_result(monkeypatch):
    fast_result = {"q": {"noul": 0.5, "probabilities": {"true": 0.5, "false": 0.5}}}
    provider = RoutingProvider()
    provider.fast_provider = MagicMock(**{"evaluate_batch.return_value": fast_result})
    provider.fast_provider.check_token_limit.return_value = 10
    provider.smart_provider = MagicMock(**{"evaluate_batch.return_value": {}})

    results = provider.evaluate_batch(STATE, [NoulQuestion(key="q", prompt="?")])

    assert results["q"]["escalation_failed"] is True
    assert "escalation_failed" not in fast_result["q"]
    assert "served_by" not in fast_result["q"]


# --- helpers -----------------------------------------------------------------------------


def test_escalation_failures_lists_the_keys_in_order():
    results = {
        "a": {"escalation_failed": True},
        "b": {"probabilities": {}},
        "c": {"escalation_failed": True},
    }

    assert escalation_failures(results) == ["a", "c"]


def test_mark_degraded_returns_a_new_payload_and_leaves_the_original_alone():
    payload = {"status": "SUCCESS", "data": 1}

    marked = mark_degraded(payload, ["q"])

    assert marked == {
        "status": "DEGRADED",
        "degraded": True,
        "escalation_failed": ["q"],
        "data": 1,
    }
    assert payload == {"status": "SUCCESS", "data": 1}


def test_mark_degraded_leaves_a_payload_with_no_failures_unchanged():
    payload = {"status": "SUCCESS"}

    assert mark_degraded(payload, []) is payload


# --- 3. the tools say DEGRADED, not SUCCESS ----------------------------------------------

FAILED = {
    "noul": 0.5,
    "probabilities": {"true": 0.5, "false": 0.5},
    "served_by": "fast",
    "escalation_failed": True,
}
HEALTHY = {
    "noul": 0.1,
    "probabilities": {"true": 0.1, "false": 0.9},
    "served_by": "fast",
}


def _assert_degraded(payload, keys):
    assert payload["status"] == "DEGRADED"
    assert payload["degraded"] is True
    assert payload["escalation_failed"] == keys


def test_evaluate_batch_tool_reports_degraded_when_an_escalation_failed():
    provider = MagicMock()
    provider.check_token_limit.return_value = 10
    provider.max_tokens = 1000
    provider.current_model_id = "m"
    provider.evaluate_batch.side_effect = lambda state, qs: {
        q.key: FAILED if q.key == "q1" else HEALTHY for q in qs
    }
    questions = [
        NoulQuestion(key="q1", prompt="Is it on?"),
        NoulQuestion(key="q2", prompt="Is it off?"),
    ]

    with patch("jev_mcp.app.provider", provider):
        res = json.loads(server.jev_evaluate_batch(state={"a": 1}, questions=questions))

    _assert_degraded(res, ["q1"])
    assert res["data"]["q1"]["escalation_failed"] is True


def test_evaluate_batch_tool_end_to_end_a_think_first_smart_engine_degrades(engines):
    # The whole path over the wire: a below-gate fast answer escalates, the smart
    # engine starts with a think tag, and the tool must not read as a plain success.
    fast, smart = engines
    fast.set_scenario("normal", noul_true=0.5)
    smart.set_scenario("think_first")

    with patch("jev_mcp.app.provider", RoutingProvider()):
        res = json.loads(
            server.jev_evaluate_batch(
                state={"note": STATE},
                questions=[NoulQuestion(key="is_ship", prompt="Is it about shipping?")],
            )
        )

    _assert_degraded(res, ["is_ship"])
    assert res["data"]["is_ship"]["served_by"] == "fast"


def test_evaluate_batch_tool_still_says_success_when_nothing_failed():
    provider = MagicMock()
    provider.check_token_limit.return_value = 10
    provider.max_tokens = 1000
    provider.current_model_id = "m"
    provider.evaluate_batch.side_effect = lambda state, qs: {q.key: HEALTHY for q in qs}

    with patch("jev_mcp.app.provider", provider):
        res = json.loads(
            server.jev_evaluate_batch(
                state={"a": 1}, questions=[NoulQuestion(key="q1", prompt="Is it on?")]
            )
        )

    assert res["status"].startswith("SUCCESS")
    assert "degraded" not in res


def test_triage_reports_degraded_when_an_escalation_failed(tmp_path, monkeypatch):
    from jev_mcp.email_triage.mcp_tool import (
        configure_triage_labels,
        triage_email_content,
    )

    monkeypatch.setattr(
        "jev_mcp.email_triage.mcp_tool.get_config_path",
        lambda: tmp_path / "triage_configs.json",
    )
    configure_triage_labels("a@mailbox.com", ["Notifications", "Receipts"])
    provider = MagicMock()
    provider.evaluate_batch.return_value = {
        "requires_action": {**FAILED, "probabilities": {"true": 0.5}},
        "is_important": {"probabilities": {"true": 0.93}},
        "bucket": {"probabilities": {"Notifications": 0.9, "Receipts": 0.1}},
    }
    monkeypatch.setattr("jev_mcp.routing_provider.RoutingProvider", lambda: provider)

    res = json.loads(triage_email_content("a@mailbox.com", "s", "x@y.com", "body"))

    _assert_degraded(res, ["requires_action"])
    assert res["decision"]  # the decision is still returned


def test_read_file_whole_file_reports_degraded(tmp_path):
    f = tmp_path / "mod.py"
    f.write_text("def relevant():\n    return 1\n")
    answer = {"file_eval": {**FAILED, "probabilities": {"true": 0.6, "false": 0.4}}}

    with patch(
        "jev_mcp.routing_provider.RoutingProvider.evaluate_batch", return_value=answer
    ):
        res = json.loads(server.jev_read_file(str(f), "task", filter_by_chunk=False))

    _assert_degraded(res, ["file_eval"])
    assert "relevant" in res["content"]


def test_read_file_chunk_filter_reports_degraded(tmp_path):
    f = tmp_path / "mod.py"
    f.write_text("def relevant():\n    return 1\n")
    answer = {"chunk_0": {**FAILED, "probabilities": {"true": 0.9, "false": 0.1}}}

    with patch(
        "jev_mcp.routing_provider.RoutingProvider.evaluate_batch", return_value=answer
    ):
        res = json.loads(server.jev_read_file(str(f), "task", filter_by_chunk=True))

    _assert_degraded(res, ["chunk_0"])


def test_scan_repo_reports_degraded(tmp_path):
    (tmp_path / "mod.py").write_text("def relevant():\n    return 1\n")
    answer = {"chunk_0": {**FAILED, "probabilities": {"true": 0.9, "false": 0.1}}}

    with patch(
        "jev_mcp.routing_provider.RoutingProvider.evaluate_batch", return_value=answer
    ):
        res = json.loads(server.jev_scan_repo(str(tmp_path), "task"))

    assert res["status"] == "DEGRADED"
    assert res["degraded"] is True
    assert res["escalation_failed"] == ["chunk_0"]


def test_read_file_whole_file_block_keeps_its_status_and_reports_degraded(tmp_path):
    f = tmp_path / "mod.py"
    f.write_text("def irrelevant():\n    return 1\n")
    answer = {"file_eval": {**FAILED, "probabilities": {"true": 0.4, "false": 0.6}}}

    with patch(
        "jev_mcp.routing_provider.RoutingProvider.evaluate_batch", return_value=answer
    ):
        res = json.loads(server.jev_read_file(str(f), "task", filter_by_chunk=False))

    assert res["status"] == "BLOCKED"
    assert res["degraded"] is True
    assert res["escalation_failed"] == ["file_eval"]


def test_read_file_chunk_filter_block_keeps_its_status_and_reports_degraded(tmp_path):
    f = tmp_path / "mod.py"
    f.write_text("def irrelevant():\n    return 1\n")
    answer = {"chunk_0": {**FAILED, "probabilities": {"true": 0.1, "false": 0.9}}}

    with patch(
        "jev_mcp.routing_provider.RoutingProvider.evaluate_batch", return_value=answer
    ):
        res = json.loads(server.jev_read_file(str(f), "task", filter_by_chunk=True))

    assert res["status"] == "BLOCKED"
    assert res["degraded"] is True
    assert res["escalation_failed"] == ["chunk_0"]


def test_scan_repo_block_keeps_its_status_and_reports_degraded(tmp_path):
    (tmp_path / "mod.py").write_text("def irrelevant():\n    return 1\n")
    answer = {"chunk_0": {**FAILED, "probabilities": {"true": 0.1, "false": 0.9}}}

    with patch(
        "jev_mcp.routing_provider.RoutingProvider.evaluate_batch", return_value=answer
    ):
        res = json.loads(server.jev_scan_repo(str(tmp_path), "task"))

    assert res["status"] == "BLOCKED"
    assert res["degraded"] is True
    assert res["escalation_failed"] == ["chunk_0"]


@patch("jev_mcp.app.RoutingProvider")
def test_determine_best_model_keeps_its_status_and_reports_degraded(mock_class):
    provider = MagicMock()
    mock_class.return_value = provider
    provider.smart_provider.current_model_id = "m"
    provider.evaluate_batch.return_value = {
        "complexity": {
            "score": "2",
            "probabilities": {"1": 0.1, "2": 0.5, "3": 0.2, "4": 0.2},
            "served_by": "fast",
            "escalation_failed": True,
        }
    }

    res = json.loads(server.jev_determine_best_model("a task", estimated_tokens=0))

    # EVALUATED is the tool's own status; degradation is flagged, not a status swap.
    assert res["status"] == "EVALUATED"
    assert res["degraded"] is True
    assert res["escalation_failed"] == ["complexity"]
    assert res["complexity_score"] == 2


@patch("jev_mcp.app.RoutingProvider")
def test_agent_handoff_keeps_its_status_and_reports_degraded(mock_class):
    provider = MagicMock()
    mock_class.return_value = provider
    provider.smart_provider.current_model_id = "m"
    provider.evaluate_batch.return_value = {
        "best_agent": {
            "choice": "A",
            "probabilities": {"A": 0.6, "B": 0.4},
            "served_by": "fast",
            "escalation_failed": True,
        }
    }

    res = json.loads(server.jev_agent_handoff("task", {"A": "x", "B": "y"}))

    assert res["status"] == "HANDOFF_RECOMMENDATION"
    assert res["degraded"] is True
    assert res["escalation_failed"] == ["best_agent"]
    assert res["recommended_agent"] == "A"


def test_optimize_prompt_reports_degraded_when_a_variation_failed_to_escalate():
    failed = {"noul": 0.6, "probabilities": {"true": 0.6, "false": 0.4}}
    results = {
        "var_0": {**failed, "escalation_failed": True},
        "var_1": {**failed, "noul": 0.7, "served_by": "fast"},
    }
    provider = MagicMock()
    provider.evaluate_batch.return_value = results

    with (
        patch("jev_mcp.app.provider", provider),
        patch(
            "jev_mcp.app._chat_completion_json",
            return_value={"variations": ["Is it really on?"]},
        ),
    ):
        res = json.loads(
            server.jev_optimize_prompt(
                question=NoulQuestion(key="q", prompt="Is it on?"), state={"a": 1}
            )
        )

    _assert_degraded(res, ["var_0"])


def test_local_choose_reports_the_questions_that_failed_to_escalate(monkeypatch):
    from jev_mcp.ultrafast_adapter import local_choose

    class FailingProvider:
        def evaluate_batch(self, state, questions):
            return {
                q.key: {
                    "choice": "CLICK",
                    "confidence": 0.6,
                    "probabilities": {"CLICK": 0.6, "TYPE_TEXT": 0.4},
                    "escalation_failed": True,
                }
                for q in questions
            }

    monkeypatch.setattr("jev_mcp.ultrafast_adapter.RoutingProvider", FailingProvider)

    result = local_choose(
        {"actions": []},
        "goal",
        [],
        {"CLICK": "Click", "TYPE_TEXT": "Type"},
        {},
        {},
    )

    assert result["escalation_failed"] == ["operation"]
    assert result["answers"]["operation"]["choice"] == "CLICK"


# --- 4. a degraded triage fails toward attention, never toward silence ---------------------


def _triage_results(requires_action, important, bucket_probs, failed_key=None):
    results = {
        "requires_action": {"probabilities": {"true": requires_action}},
        "is_important": {"probabilities": {"true": important}},
        "bucket": {"probabilities": bucket_probs},
    }
    if failed_key:
        results[failed_key] = {**FAILED, **results[failed_key]}
    return results


def _evaluate_email(monkeypatch, results):
    from jev_mcp.email_triage.mcp_tool import evaluate_email_with_mlx

    provider = MagicMock()
    provider.evaluate_batch.return_value = results
    monkeypatch.setattr("jev_mcp.routing_provider.RoutingProvider", lambda: provider)
    return evaluate_email_with_mlx("s", "x@y.com", "body", ["Notifications"])


CONFIDENT_BUCKET = {"Notifications": 0.95, "Receipts": 0.05}


def test_degraded_triage_never_archives_silently(monkeypatch):
    results = _triage_results(0.05, 0.1, CONFIDENT_BUCKET, "bucket")

    res = _evaluate_email(monkeypatch, results)

    assert res["status"] == "DEGRADED"
    assert res["suggested_routing"] == "hitl_slack_block"
    assert res["decision"] == "Review"
    assert res["reason"] == "escalation_failed_unconfirmed"


def test_degraded_triage_keeps_an_escalation(monkeypatch):
    results = _triage_results(0.95, 0.9, CONFIDENT_BUCKET, "requires_action")

    res = _evaluate_email(monkeypatch, results)

    assert res["status"] == "DEGRADED"
    assert res["suggested_routing"] == "escalate_to_slack"
    assert res["reason"] == "requires_action"


def test_degraded_triage_keeps_a_review(monkeypatch):
    results = _triage_results(0.5, 0.1, CONFIDENT_BUCKET, "requires_action")

    res = _evaluate_email(monkeypatch, results)

    assert res["status"] == "DEGRADED"
    assert res["suggested_routing"] == "hitl_slack_block"
    assert res["reason"] == "ambiguous_action"


def test_clean_triage_still_archives(monkeypatch):
    results = _triage_results(0.05, 0.1, CONFIDENT_BUCKET)

    res = _evaluate_email(monkeypatch, results)

    assert "degraded" not in res
    assert res["suggested_routing"] == "archive_and_label"
    assert res["decision"] == "Notifications"
