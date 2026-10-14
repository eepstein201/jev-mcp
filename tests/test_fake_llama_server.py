"""Self-tests for the loopback fake llama-server fixture (tests/fake_llama_server.py)."""

import itertools
import json
import math
import socket
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Optional
from unittest.mock import patch

import pytest

from jev_mcp.daemon_provider import DaemonProvider
from jev_mcp.kev_provider import KevProvider
from jev_mcp.provider import ChoiceQuestion, NoulQuestion, ScoreQuestion
from tests.fake_llama_server import (
    FakeLlamaServer,
    is_fake_llama_port,
    load_recorded,
)

CHAT = "/v1/chat/completions"
SYSTEMONE = "/v1/systemone"
LOADING_BODY = {
    "error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}
}
EMAIL = "Email from shipping@example.com: Your order #4821 has shipped and will arrive Tuesday."
RECORDED = load_recorded()
RECORDED_SMART_MODEL = RECORDED["smart_exchanges"][0]["response"]["model"]
RECORDED_KEV_MODEL = RECORDED["kev_exchanges"][0]["response"]["model"]


def call(
    server: FakeLlamaServer,
    path: str,
    body: Optional[Any] = None,
    method: Optional[str] = None,
    timeout: float = 5.0,
) -> tuple[int, Any]:
    """One real HTTP call to the fake; returns (status, parsed JSON body)."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    verb = method or ("GET" if body is None else "POST")
    req = urllib.request.Request(
        server.url + path,
        data=data,
        method=verb,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode())


def chat_body(prompt: str) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": 11,
        "temperature": 0.0,
    }


def top_logprobs(response: dict[str, Any]) -> list[dict[str, Any]]:
    return response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]


@pytest.fixture
def server():
    with FakeLlamaServer() as running:
        yield running


# --- endpoints -------------------------------------------------------------------


def test_health_reports_ok_in_the_normal_scenario(server):
    status, body = call(server, "/health")

    assert (status, body) == (200, {"status": "ok"})


def test_props_serves_the_recorded_smart_engine_by_default(server):
    recorded = RECORDED["props"]["smart"]

    status, body = call(server, "/props")

    assert status == 200
    assert body["model_path"] == recorded["model_path"]
    assert body["total_slots"] == recorded["total_slots"]
    assert body["default_generation_settings"]["n_ctx"] == recorded["n_ctx"]


def test_props_values_are_configurable(server):
    server.set_scenario(
        "normal", model_path="/models/other.gguf", total_slots=7, n_ctx=2048
    )

    _, body = call(server, "/props")

    assert body == {
        "model_path": "/models/other.gguf",
        "total_slots": 7,
        "default_generation_settings": {"n_ctx": 2048},
    }


def test_chat_synthesizes_an_openai_shaped_answer_for_an_unrecorded_prompt(server):
    prompt = "Answer True or False.\nAnswer:"

    status, body = call(server, CHAT, chat_body(prompt))

    assert status == 200
    assert body["model"] == RECORDED_SMART_MODEL
    assert body["object"] == "chat.completion"
    tokens = {item["token"]: item["logprob"] for item in top_logprobs(body)}
    assert set(tokens) == {"True", "False"}
    assert math.exp(tokens["True"]) == pytest.approx(0.8)
    assert body["choices"][0]["message"]["content"] == "True"
    assert body["choices"][0]["logprobs"]["content"][0]["token"] == "True"


def test_chat_reads_option_letters_from_the_prompt(server):
    prompt = "Options: A: Receipt, B: Delivery\nAnswer strictly with the corresponding letter.\nAnswer:"

    _, body = call(server, CHAT, chat_body(prompt))

    assert [item["token"] for item in top_logprobs(body)] == ["A", "B"]


def test_chat_reads_score_indexes_from_the_prompt(server):
    prompt = "Options: 0: low, 1: high\nAnswer strictly with the corresponding index number.\nAnswer:"

    _, body = call(server, CHAT, chat_body(prompt))

    assert [item["token"] for item in top_logprobs(body)] == ["0", "1"]


def test_chat_converts_first_token_probs_to_logprobs_capped_at_eleven(server):
    probs = {f"t{i}": 0.5**i for i in range(1, 15)}
    server.set_scenario("normal", first_token_probs=probs, model="/models/x.gguf")

    _, body = call(server, CHAT, chat_body("anything"))

    entries = top_logprobs(body)
    assert len(entries) == 11
    assert entries[0]["token"] == "t1"
    assert entries[0]["logprob"] == pytest.approx(math.log(0.5))
    assert body["model"] == "/models/x.gguf"


def test_systemone_synthesizes_every_question_type_in_the_recorded_shape(server):
    questions = {
        "n": {"type": "noul", "instructions": "x"},
        "c": {"type": "choice", "instructions": "x", "criteria": {"a": "", "b": ""}},
        "s": {"type": "score", "instructions": "x", "criteria": ["lo", "hi"]},
    }
    server.set_scenario("normal", noul_true=0.25)

    status, body = call(server, SYSTEMONE, {"state": "s", "questions": questions})

    assert status == 200
    answers = body["answers"]
    assert answers["n"] == {"noul": 0.25, "type": "noul"}
    assert answers["c"]["choice"] == "a"
    assert set(answers["c"]["probabilities"]) == {"a", "b"}
    assert answers["s"]["legend"] == {"0": "lo", "1": "hi"}
    assert set(answers["s"]["probabilities"]) == {"0", "1"}
    assert answers["s"]["type"] == "score"
    assert body["model"] == RECORDED_KEV_MODEL


def test_systemone_choice_and_score_probabilities_are_configurable(server):
    questions = {
        "c": {"type": "choice", "instructions": "x", "criteria": {"a": "", "b": ""}},
        "s": {"type": "score", "instructions": "x", "criteria": ["lo", "hi"]},
    }
    server.set_scenario(
        "normal", choice_probs={"a": 0.1, "b": 0.9}, score_probs=[0.25, 0.75]
    )

    _, body = call(server, SYSTEMONE, {"state": "s", "questions": questions})

    assert body["answers"]["c"]["choice"] == "b"
    assert body["answers"]["c"]["probabilities"] == {"a": 0.1, "b": 0.9}
    assert body["answers"]["s"]["probabilities"] == {"0": 0.25, "1": 0.75}
    assert body["answers"]["s"]["score"] == pytest.approx(0.75)


def test_systemone_answers_an_unknown_question_type_with_only_its_type(server):
    questions = {"w": {"type": "mystery", "instructions": "x"}}

    _, body = call(server, SYSTEMONE, {"state": "s", "questions": questions})

    assert body["answers"] == {"w": {"type": "mystery"}}


# --- scenarios ------------------------------------------------------------------


def test_think_first_puts_think_in_the_first_token_with_no_target_tokens(server):
    server.set_scenario("think_first")
    prompt = "Answer True or False.\nAnswer:"

    status, body = call(server, CHAT, chat_body(prompt))

    assert status == 200
    tokens = [item["token"].strip().lower() for item in top_logprobs(body)]
    assert tokens[0] == "<think>"
    assert "true" not in tokens and "false" not in tokens
    assert body["choices"][0]["message"]["content"] == "<think>"


def test_context_overflow_returns_the_recorded_400_body(server):
    server.set_scenario("context_overflow")

    status, body = call(server, CHAT, chat_body("anything"))

    assert status == 400
    assert body == RECORDED["context_overflow"]["body"]
    assert body["error"]["type"] == "exceed_context_size_error"


def test_missing_model_drops_the_model_key_from_the_chat_response(server):
    server.set_scenario("missing_model")

    status, body = call(server, CHAT, chat_body("anything"))

    assert status == 200
    assert "model" not in body
    assert top_logprobs(body)


def test_wrong_kev_shape_answers_lack_the_noul_and_choice_keys(server):
    server.set_scenario("wrong_kev_shape")
    questions = {
        "n": {"type": "noul", "instructions": "x"},
        "c": {"type": "choice", "instructions": "x", "criteria": {"a": ""}},
    }

    status, body = call(server, SYSTEMONE, {"state": "s", "questions": questions})

    assert status == 200
    assert body["answers"] == {"n": {"value": 0.9}, "c": {"value": 0.9}}


def test_wrong_kev_shape_does_not_replay_a_recorded_request(server):
    server.set_scenario("wrong_kev_shape")
    recorded_request = RECORDED["kev_exchanges"][0]["request"]

    _, body = call(server, SYSTEMONE, recorded_request)

    assert all("value" in answer for answer in body["answers"].values())


def test_loading_makes_health_return_503(server):
    server.set_scenario("loading")

    status, body = call(server, "/health")

    assert (status, body) == (503, LOADING_BODY)


def test_loading_makes_props_return_503(server):
    server.set_scenario("loading")

    status, body = call(server, "/props")

    assert (status, body) == (503, LOADING_BODY)


def test_props_option_serves_a_custom_props_body_verbatim(server):
    server.set_scenario("normal", props={"model_path": 42})

    status, body = call(server, "/props")

    assert (status, body) == (200, {"model_path": 42})


@pytest.mark.parametrize(
    "path, body",
    [
        (CHAT, chat_body("anything")),
        (SYSTEMONE, {"state": "s", "questions": {"n": {"type": "noul"}}}),
    ],
)
def test_loading_makes_both_post_endpoints_return_503(server, path, body):
    server.set_scenario("loading")

    status, payload = call(server, path, body)

    assert (status, payload) == (503, LOADING_BODY)


def test_swap_after_reports_model_a_then_model_b(server):
    server.set_scenario(
        "swap_after", n=2, model_a="/models/a.gguf", model_b="/models/b.gguf"
    )

    models = [call(server, CHAT, chat_body("q"))[1]["model"] for _ in range(4)]

    assert models == [
        "/models/a.gguf",
        "/models/a.gguf",
        "/models/b.gguf",
        "/models/b.gguf",
    ]


def test_swap_after_restarts_counting_when_the_scenario_is_set_again(server):
    server.set_scenario(
        "swap_after", n=1, model_a="/models/a.gguf", model_b="/models/b.gguf"
    )
    call(server, CHAT, chat_body("q"))
    call(server, CHAT, chat_body("q"))

    server.set_scenario(
        "swap_after", n=1, model_a="/models/a.gguf", model_b="/models/b.gguf"
    )
    _, body = call(server, CHAT, chat_body("q"))

    assert body["model"] == "/models/a.gguf"


def test_slow_delays_the_answer_and_trips_a_short_client_timeout(server):
    server.set_scenario("slow", seconds=1.0)

    started = time.monotonic()
    with pytest.raises((TimeoutError, urllib.error.URLError, socket.timeout)):
        call(server, "/health", timeout=0.2)
    assert time.monotonic() - started < 1.0

    status, body = call(server, "/health", timeout=5.0)
    assert (status, body) == (200, {"status": "ok"})


def test_unknown_scenario_is_rejected(server):
    with pytest.raises(ValueError, match="unknown scenario"):
        server.set_scenario("not_a_scenario")


# --- replay of recorded exchanges ----------------------------------------------------


@pytest.mark.parametrize(
    "exchange",
    RECORDED["smart_exchanges"],
    ids=lambda e: e["request"]["messages"][0]["content"][-60:],
)
def test_a_recorded_chat_request_replays_the_recorded_response_verbatim(
    server, exchange
):
    status, body = call(server, CHAT, exchange["request"])

    assert status == 200
    assert body == exchange["response"]


@pytest.mark.parametrize("exchange", RECORDED["kev_exchanges"])
def test_a_recorded_systemone_request_replays_the_recorded_response_verbatim(
    server, exchange
):
    status, body = call(server, SYSTEMONE, exchange["request"])

    assert status == 200
    assert body == exchange["response"]


def test_a_request_that_differs_from_a_recording_is_not_replayed(server):
    changed = json.loads(json.dumps(RECORDED["smart_exchanges"][0]["request"]))
    changed["messages"][0]["content"] += " "

    _, body = call(server, CHAT, changed)

    assert body != RECORDED["smart_exchanges"][0]["response"]


# --- fidelity: the real providers against the fake -----------------------------------


def test_kev_provider_parses_the_fake_like_the_recorded_response(server):
    exchange = RECORDED["kev_exchanges"][0]
    answers = exchange["response"]["answers"]
    questions = [
        NoulQuestion(
            key="is_delivery", prompt="Is this email about a package delivery?"
        ),
        ChoiceQuestion(
            key="kind",
            prompt="Which kind of message is this?",
            options=["Delivery", "Newsletter", "Receipt"],
        ),
        ScoreQuestion(
            key="urgency",
            prompt="How urgent is this?",
            labels=["low", "medium", "high"],
        ),
    ]
    provider = KevProvider(base_url=server.url + SYSTEMONE)

    results = provider.evaluate_batch(EMAIL, questions)

    noul = answers["is_delivery"]["noul"]
    assert results == {
        "is_delivery": {
            "noul": noul,
            "probabilities": {"true": noul, "false": 1.0 - noul},
        },
        "kind": {
            "choice": answers["kind"]["choice"],
            "confidence": answers["kind"]["confidence"],
            "probabilities": answers["kind"]["probabilities"],
        },
        "urgency": {
            "score": "medium",
            "probabilities": {
                "low": answers["urgency"]["probabilities"]["0"],
                "medium": answers["urgency"]["probabilities"]["1"],
                "high": answers["urgency"]["probabilities"]["2"],
            },
        },
    }
    # The provider asks /props for its model id once, then posts the questions.
    assert [path for _, path, _ in server.requests] == ["/props", SYSTEMONE]
    assert server.requests[1][2] == exchange["request"]


def test_daemon_provider_scores_the_fake_like_the_recorded_logprobs(
    server, monkeypatch
):
    # The recorded prompts carry their own salted tags; pin the salts so the provider
    # sends the recorded requests byte for byte and the fake replays the real answers.
    salts = itertools.cycle(["jdy51018", "99q9anip"])
    monkeypatch.setattr("random.choices", lambda *a, **k: list(next(salts)))
    monkeypatch.setattr(DaemonProvider, "_prior_cache", {})
    monkeypatch.setattr(DaemonProvider, "_get_daemon_pid", lambda self: "1")
    provider = DaemonProvider(base_url=server.url + CHAT)
    questions = [
        NoulQuestion(
            key="is_delivery", prompt="Is this email about a package delivery?"
        ),
        ChoiceQuestion(
            key="kind",
            prompt="Which kind of message is this?",
            options=["Receipt", "Delivery", "Newsletter"],
        ),
    ]

    results = provider.evaluate_batch(EMAIL, questions)

    recorded_requests = [e["request"] for e in RECORDED["smart_exchanges"]]
    posted = [body for method, _, body in server.requests if method == "POST"]
    assert posted == recorded_requests
    assert provider.current_model_id == RECORDED_SMART_MODEL
    assert set(results) == {"is_delivery", "kind"}
    # The recorded 7B put its mass on "False" for this delivery email.
    assert results["is_delivery"]["noul"] == pytest.approx(1.6234e-05, rel=1e-3)
    assert results["is_delivery"]["probabilities"].keys() == {"true", "false"}
    assert results["kind"]["choice"] == "Delivery"
    assert results["kind"]["confidence"] == pytest.approx(1.0, abs=1e-6)
    assert results["kind"]["probabilities"].keys() == {
        "Receipt",
        "Delivery",
        "Newsletter",
    }


def test_daemon_provider_reads_a_synthesized_answer_without_replay(server, monkeypatch):
    monkeypatch.setattr(DaemonProvider, "_prior_cache", {})
    monkeypatch.setattr(DaemonProvider, "_get_daemon_pid", lambda self: "2")
    server.set_scenario("normal", first_token_probs={"True": 0.7, "False": 0.3})
    provider = DaemonProvider(base_url=server.url + CHAT)

    results = provider.evaluate_batch(
        "some state", [NoulQuestion(key="q", prompt="Is it?")]
    )

    assert results["q"]["probabilities"].keys() == {"true", "false"}
    assert sum(results["q"]["probabilities"].values()) == pytest.approx(1.0)


# --- request log, lifecycle, guard ------------------------------------------------------


def test_requests_logs_method_path_and_parsed_body_in_order(server):
    body = chat_body("hello")

    call(server, "/health")
    call(server, CHAT, body)

    assert server.requests == [("GET", "/health", None), ("POST", CHAT, body)]


def test_requests_returns_a_copy_the_caller_cannot_use_to_alter_the_log(server):
    call(server, "/health")

    server.requests.clear()

    assert len(server.requests) == 1


def test_a_body_that_is_not_json_is_a_400_and_is_logged_without_a_body(server):
    req = urllib.request.Request(server.url + CHAT, data=b"{not json", method="POST")

    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(req, timeout=5)

    assert caught.value.code == 400
    assert server.requests == [("POST", CHAT, None)]


@pytest.mark.parametrize("path", [CHAT, SYSTEMONE])
def test_a_json_body_that_is_not_an_object_is_a_400(server, path):
    status, body = call(server, path, [1, 2, 3])

    assert status == 400
    assert body["error"]["code"] == 400


def test_systemone_without_a_questions_object_is_a_400(server):
    status, body = call(server, SYSTEMONE, {"state": "s"})

    assert status == 400
    assert "questions" in body["error"]["message"]


def test_an_unknown_path_is_a_404(server):
    status, body = call(server, "/nope")

    assert status == 404
    assert body["error"]["code"] == 404


def test_the_server_binds_only_the_loopback_interface(server):
    host = server.url.split("//", 1)[1].rsplit(":", 1)[0]

    assert host == "127.0.0.1"


def test_stop_frees_the_port():
    fake = FakeLlamaServer().start()
    port = int(fake.url.rsplit(":", 1)[1])
    assert is_fake_llama_port(port)

    fake.stop()

    assert not is_fake_llama_port(port)
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=1).close()
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))


def test_stop_returns_promptly_while_a_slow_request_is_in_flight():
    fake = FakeLlamaServer().start()
    fake.set_scenario("slow", seconds=30)
    url = fake.url

    started = time.monotonic()
    request = urllib.request.Request(url + "/health")
    thread = threading.Thread(
        target=lambda: _swallow(lambda: urllib.request.urlopen(request, timeout=10)),
        daemon=True,
    )
    thread.start()
    while not fake.requests and time.monotonic() - started < 5:
        time.sleep(0.01)
    fake.stop()

    assert time.monotonic() - started < 5
    thread.join(timeout=5)


def _swallow(action) -> None:
    try:
        action()
    except Exception:  # the in-flight call ending when the server stops is expected
        pass


def test_stop_and_start_are_idempotent():
    fake = FakeLlamaServer()
    fake.stop()
    fake.start()
    first_url = fake.url

    assert fake.start() is fake
    assert fake.url == first_url
    fake.stop()
    fake.stop()


def test_url_before_start_is_an_error():
    with pytest.raises(RuntimeError, match="not running"):
        FakeLlamaServer().url


def test_the_daemon_http_guard_still_blocks_other_loopback_ports(server):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]

    with pytest.raises(urllib.error.URLError, match="no daemon in tests"):
        urllib.request.urlopen(f"http://127.0.0.1:{free_port}/health", timeout=1)


def test_a_test_that_patches_urlopen_still_overrides_the_guard(server):
    with patch("urllib.request.urlopen", side_effect=RuntimeError("patched")):
        with pytest.raises(RuntimeError, match="patched"):
            urllib.request.urlopen(server.url + "/health")


# --- fix round 1 ---------------------------------------------------------------------


def test_stop_is_prompt_because_the_serve_loop_polls_quickly():
    fake = FakeLlamaServer().start()

    started = time.monotonic()
    fake.stop()

    assert time.monotonic() - started < 0.3


def test_recorded_props_model_paths_equal_the_recorded_response_models():
    props = RECORDED["props"]

    assert props["smart"]["model_path"] == RECORDED_SMART_MODEL
    assert props["fast"]["model_path"] == RECORDED_KEV_MODEL


def test_props_model_path_equals_the_chat_model_for_the_smart_tier(server):
    _, props = call(server, "/props")
    _, chat = call(server, CHAT, chat_body("Answer True or False.\nAnswer:"))

    assert props["model_path"] == chat["model"]


def test_props_model_path_equals_the_systemone_model_for_the_fast_tier(server):
    server.set_scenario("normal", tier="fast")
    questions = {"n": {"type": "noul", "instructions": "x"}}

    _, props = call(server, "/props")
    _, kev = call(server, SYSTEMONE, {"state": "s", "questions": questions})
    _, chat = call(server, CHAT, chat_body("anything"))

    assert props["model_path"] == kev["model"] == RECORDED_KEV_MODEL
    assert chat["model"] == RECORDED_KEV_MODEL
    assert props["total_slots"] == RECORDED["props"]["fast"]["total_slots"]


def test_model_path_option_moves_props_and_chat_model_together(server):
    server.set_scenario("normal", model_path="/models/other.gguf")

    _, props = call(server, "/props")
    _, chat = call(server, CHAT, chat_body("anything"))

    assert props["model_path"] == chat["model"] == "/models/other.gguf"


def test_confidence_option_sets_choice_and_score_confidence(server):
    questions = {
        "c": {"type": "choice", "instructions": "x", "criteria": {"a": "", "b": ""}},
        "s": {"type": "score", "instructions": "x", "criteria": ["lo", "hi"]},
    }
    server.set_scenario("normal", confidence=0.31)

    _, body = call(server, SYSTEMONE, {"state": "s", "questions": questions})

    assert body["answers"]["c"]["confidence"] == 0.31
    assert body["answers"]["s"]["confidence"] == 0.31


def test_confidence_defaults_to_the_top_probability(server):
    questions = {"c": {"type": "choice", "instructions": "x", "criteria": {"a": ""}}}

    _, body = call(server, SYSTEMONE, {"state": "s", "questions": questions})

    assert body["answers"]["c"]["confidence"] == 1.0


def test_guard_blocks_a_malformed_port_with_url_error_not_value_error():
    with pytest.raises(urllib.error.URLError, match="no daemon in tests"):
        urllib.request.urlopen("http://127.0.0.1:99999/health", timeout=1)


def test_guard_accepts_the_url_keyword_form_for_a_fake(server):
    with urllib.request.urlopen(url=server.url + "/health", timeout=5) as response:
        assert json.loads(response.read().decode()) == {"status": "ok"}


def test_a_broken_pipe_is_logged_quietly_not_printed(server, capsys):
    try:
        raise BrokenPipeError("client went away")
    except BrokenPipeError:
        server._server.handle_error(None, ("127.0.0.1", 0))

    assert capsys.readouterr().err == ""


def test_other_handler_errors_still_use_the_default_traceback(server, capsys):
    try:
        raise ValueError("a real bug")
    except ValueError:
        server._server.handle_error(None, ("127.0.0.1", 0))

    assert "a real bug" in capsys.readouterr().err


def test_unknown_tier_is_rejected(server):
    with pytest.raises(ValueError, match="unknown tier"):
        server.set_scenario("normal", tier="medium")


# --- engine-profiles additions: recorded props, wrong_model, alias ---------------------


@pytest.mark.parametrize("tier", ["smart", "fast"])
def test_every_recorded_props_carries_the_nested_n_ctx_beside_the_top_level_one(tier):
    recorded = RECORDED["props"][tier]

    assert recorded["default_generation_settings"]["n_ctx"] == recorded["n_ctx"]


def test_wrong_model_scenario_serves_a_different_model_than_the_recorded_one(server):
    server.set_scenario("wrong_model")

    _, props = call(server, "/props")
    _, chat = call(server, CHAT, chat_body("Answer True or False.\nAnswer:"))

    assert props["model_path"] != RECORDED_SMART_MODEL
    assert chat["model"] == props["model_path"]
    # Only the identity differs: the slot and context figures stay the recorded ones.
    assert props["total_slots"] == RECORDED["props"]["smart"]["total_slots"]


def test_wrong_model_scenario_follows_the_requested_tier(server):
    server.set_scenario("wrong_model", tier="fast")

    _, props = call(server, "/props")

    assert props["model_path"] != RECORDED_KEV_MODEL
    assert props["total_slots"] == RECORDED["props"]["fast"]["total_slots"]


def test_wrong_model_scenario_accepts_an_explicit_model(server):
    server.set_scenario("wrong_model", model="/models/Chosen-Q4.gguf")

    _, props = call(server, "/props")

    assert props["model_path"] == "/models/Chosen-Q4.gguf"


def test_alias_option_renames_the_chat_reply_but_not_props(server):
    server.set_scenario("normal", alias="qwen2.5-7b")

    _, props = call(server, "/props")
    _, chat = call(server, CHAT, chat_body("Answer True or False.\nAnswer:"))

    assert chat["model"] == "qwen2.5-7b"
    assert props["model_path"] == RECORDED_SMART_MODEL


def test_alias_option_overrides_a_replayed_recording_too(server):
    server.set_scenario("normal", alias="qwen2.5-7b")
    recorded_request = RECORDED["smart_exchanges"][0]["request"]

    _, chat = call(server, CHAT, recorded_request)

    assert chat["model"] == "qwen2.5-7b"


def test_alias_option_renames_the_systemone_reply(server):
    server.set_scenario("normal", tier="fast", alias="kev-0.8b")

    _, props = call(server, "/props")
    _, kev = call(server, SYSTEMONE, RECORDED["kev_exchanges"][0]["request"])

    assert kev["model"] == "kev-0.8b"
    assert props["model_path"] == RECORDED_KEV_MODEL
