"""A loopback stand-in for the llama-server daemons, built from real recordings.

Tests that need a daemon start a `FakeLlamaServer` and point a provider at its `url`.
It speaks the three endpoints the providers use (`/health`, `/props`,
`/v1/chat/completions`, `/v1/systemone`) over a real socket bound to 127.0.0.1 only.

Why it replays recordings: a mock that echoes the caller's input cannot catch a provider
that misreads what a daemon really sends. A request body equal to one in
`tests/fixtures/llama/recorded.json` gets that exchange's recorded response verbatim;
any other request gets an answer synthesised in the same shape.

Scenarios (`set_scenario`) make the daemon misbehave the ways real ones do.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

LOOPBACK_HOST = "127.0.0.1"
RECORDED_PATH = Path(__file__).parent / "fixtures" / "llama" / "recorded.json"

TIERS = ("smart", "fast")
SCENARIOS = frozenset(
    {
        "normal",
        "think_first",
        "context_overflow",
        "missing_model",
        "wrong_kev_shape",
        "loading",
        "swap_after",
        "slow",
        "wrong_model",
    }
)

CHAT_PATH = "/v1/chat/completions"
SYSTEMONE_PATH = "/v1/systemone"
MAX_TOP_LOGPROBS = 11
# serve_forever notices shutdown() only between polls; its 0.5 s default makes stop() slow.
SERVE_POLL_SECONDS = 0.02
DEFAULT_SLOW_SECONDS = 1.0
DEFAULT_SWAP_AFTER = 1
DEFAULT_NOUL_TRUE = 0.9
LEADING_SHARE = 0.8
THINK_TOKEN = "<think>"
# What the wrong_model scenario serves unless the test names a model.
WRONG_MODEL_PATH = "/models/Unexpected-Model-GGUF/Unexpected-Model-Q4_K_M.gguf"
LOADING_BODY = {
    "error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}
}

# Loopback ports of running fakes. The conftest daemon-HTTP guard consults this so traffic
# to a fake passes while every other unmocked call still fails.
_active_ports: set[int] = set()
_active_ports_lock = threading.Lock()


def is_fake_llama_port(port: int) -> bool:
    """True while a FakeLlamaServer is listening on `port`."""
    with _active_ports_lock:
        return port in _active_ports


def load_recorded() -> dict[str, Any]:
    """The real daemon recordings, parsed fresh so callers cannot corrupt a shared copy."""
    return json.loads(RECORDED_PATH.read_text(encoding="utf-8"))


def _default_probs(names: list[str]) -> dict[str, float]:
    """The first name leads with LEADING_SHARE; the rest split what remains evenly."""
    if len(names) <= 1:
        return {name: 1.0 for name in names}
    rest = (1.0 - LEADING_SHARE) / (len(names) - 1)
    return {
        name: LEADING_SHARE if index == 0 else rest for index, name in enumerate(names)
    }


def _prompt_targets(prompt: str) -> list[str]:
    """The answer tokens a daemon prompt asks for: True/False, option letters or indexes."""
    if "Answer True or False" in prompt:
        return ["True", "False"]
    match = re.search(r"^Options: (.+)$", prompt, flags=re.MULTILINE)
    if match:
        labels = re.findall(r"(?:^|, )(\w+): ", match.group(1))
        if labels:
            return labels
    return ["True", "False"]


def _token_entry(token: str, logprob: float) -> dict[str, Any]:
    return {
        "bytes": list(token.encode("utf-8")),
        "id": 0,
        "logprob": logprob,
        "token": token,
    }


def _chat_response(
    model: Optional[str], tokens: list[tuple[str, float]], prompt_tokens: int
) -> dict[str, Any]:
    """An OpenAI-shaped completion whose first token carries `tokens` as its top_logprobs."""
    top = [_token_entry(token, logprob) for token, logprob in tokens]
    first = top[0]
    response: dict[str, Any] = {
        "choices": [
            {
                "finish_reason": "length",
                "index": 0,
                "logprobs": {"content": [{**first, "top_logprobs": top}]},
                "message": {"content": first["token"], "role": "assistant"},
            }
        ],
        "created": int(time.time()),
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "system_fingerprint": "fake",
        "usage": {
            "completion_tokens": 1,
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens + 1,
        },
    }
    if model is not None:
        response["model"] = model
    return response


def _probs_to_tokens(probs: Mapping[str, float]) -> list[tuple[str, float]]:
    """Probabilities as log-probs, most likely first, capped at the daemon's top-k."""
    ranked = sorted(
        ((token, p) for token, p in probs.items() if p > 0),
        key=lambda item: item[1],
        reverse=True,
    )
    return [(token, math.log(p)) for token, p in ranked[:MAX_TOP_LOGPROBS]]


def _kev_answer(spec: Mapping[str, Any], options: Mapping[str, Any]) -> dict[str, Any]:
    """One SystemOne answer in the recorded shape for the question's type.

    The `confidence` option sets choice and score confidence explicitly; the recording
    shows it is not the top probability, so gate tests need to choose it.
    """
    explicit_confidence = options.get("confidence")
    qtype = spec.get("type")
    if qtype == "noul":
        return {
            "noul": float(options.get("noul_true", DEFAULT_NOUL_TRUE)),
            "type": "noul",
        }
    if qtype == "choice":
        names = list(spec.get("criteria") or {})
        probs = dict(options.get("choice_probs") or _default_probs(names))
        choice = max(probs, key=lambda name: probs[name])
        return {
            "choice": choice,
            "confidence": (
                probs[choice] if explicit_confidence is None else explicit_confidence
            ),
            "probabilities": probs,
            "type": "choice",
        }
    if qtype == "score":
        labels = list(spec.get("criteria") or [])
        values = list(options.get("score_probs") or _default_probs(labels).values())
        return {
            "confidence": (
                max(values) if explicit_confidence is None else explicit_confidence
            ),
            "legend": {str(i): label for i, label in enumerate(labels)},
            "probabilities": {str(i): p for i, p in enumerate(values)},
            "score": sum(i * p for i, p in enumerate(values)),
            "type": "score",
        }
    return {"type": qtype}


class _Handler(BaseHTTPRequestHandler):
    """Routes one request to its FakeLlamaServer; the server owns all state."""

    server: "_Server"

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug("fake llama-server: " + format, *args)

    def _send(self, status: int, body: Any) -> None:
        raw = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        self.server.fake.handle(self, "GET", None)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body: Any = json.loads(raw.decode("utf-8")) if raw else None
        except ValueError:
            self.server.fake.record("POST", self.path, None)
            self._send(400, {"error": {"code": 400, "message": "invalid JSON body"}})
            return
        self.server.fake.handle(self, "POST", body)


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, fake: "FakeLlamaServer") -> None:
        super().__init__((LOOPBACK_HOST, 0), _Handler)
        self.fake = fake

    def handle_error(self, request: Any, client_address: Any) -> None:
        """A client that timed out and left is expected; only that case goes quiet."""
        error = sys.exc_info()[1]
        if isinstance(error, (BrokenPipeError, ConnectionResetError)):
            logger.debug(
                "fake llama-server: client %s went away: %s", client_address, error
            )
            return
        super().handle_error(request, client_address)


class FakeLlamaServer:
    """A loopback llama-server. Use as a context manager or with start()/stop()."""

    def __init__(self) -> None:
        self._recorded = load_recorded()
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._requests: list[tuple[str, str, Any]] = []
        self._scenario = "normal"
        self._options: dict[str, Any] = {}
        self._chat_count = 0
        self._server: Optional[_Server] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        if self._server is None:
            raise RuntimeError("FakeLlamaServer is not running")
        return f"http://{LOOPBACK_HOST}:{self._server.server_address[1]}"

    @property
    def requests(self) -> list[tuple[str, str, Any]]:
        """Every request seen so far as (method, path, parsed JSON body or None)."""
        with self._lock:
            return list(self._requests)

    def start(self) -> "FakeLlamaServer":
        if self._server is not None:
            return self
        self._stopping.clear()
        self._server = _Server(self)
        with _active_ports_lock:
            _active_ports.add(self._server.server_address[1])
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": SERVE_POLL_SECONDS},
            name="fake-llama-server",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        server, thread = self._server, self._thread
        if server is None:
            return
        self._server = None
        self._thread = None
        self._stopping.set()
        try:
            server.shutdown()
            if thread is not None:
                thread.join(timeout=5)
        finally:
            server.server_close()
            with _active_ports_lock:
                _active_ports.discard(server.server_address[1])

    def __enter__(self) -> "FakeLlamaServer":
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def set_scenario(self, name: str = "normal", **options: Any) -> None:
        """Switch behaviour. Options replace the previous set; swap_after restarts counting.

        Options: model, first_token_probs, noul_true, choice_probs, score_probs, confidence,
        model_path, total_slots, n_ctx, tier ("smart" or "fast": which recorded engine
        the defaults come from), model_a, model_b, n (swap_after), seconds (slow),
        props (a value served verbatim as the /props body, for malformed shapes),
        alias (the `model` of every chat and SystemOne reply, as `--alias` makes a real
        daemon report it, while /props keeps the full path).
        `model` and `model_path` are the same setting for /props and every answer.
        The wrong_model scenario serves WRONG_MODEL_PATH (or the `model` given) in place
        of the tier's recorded model, with the recorded slot and context figures.
        """
        if name not in SCENARIOS:
            raise ValueError(f"unknown scenario {name!r}; known: {sorted(SCENARIOS)}")
        if options.get("tier", "smart") not in TIERS:
            raise ValueError(f"unknown tier {options['tier']!r}; known: {TIERS}")
        stored = dict(options)
        if name == "wrong_model" and not (
            stored.get("model") or stored.get("model_path")
        ):
            stored["model"] = WRONG_MODEL_PATH
        with self._lock:
            self._scenario = name
            self._options = stored
            self._chat_count = 0

    def record(self, method: str, path: str, body: Any) -> None:
        with self._lock:
            self._requests.append((method, path, copy.deepcopy(body)))

    def _state(self) -> tuple[str, dict[str, Any]]:
        with self._lock:
            return self._scenario, dict(self._options)

    def _next_chat_index(self) -> int:
        with self._lock:
            index = self._chat_count
            self._chat_count += 1
            return index

    def handle(self, handler: _Handler, method: str, body: Any) -> None:
        self.record(method, handler.path, body)
        scenario, options = self._state()
        if scenario == "slow":
            self._stopping.wait(float(options.get("seconds", DEFAULT_SLOW_SECONDS)))
        status, payload = self._route(method, handler.path, body, scenario, options)
        handler._send(status, payload)

    def _route(
        self,
        method: str,
        path: str,
        body: Any,
        scenario: str,
        options: dict[str, Any],
    ) -> tuple[int, Any]:
        loading = scenario == "loading"
        if method == "GET" and path == "/health":
            return (503, LOADING_BODY) if loading else (200, {"status": "ok"})
        if method == "GET" and path == "/props":
            return (503, LOADING_BODY) if loading else (200, self._props(options))
        if method == "POST" and path in (CHAT_PATH, SYSTEMONE_PATH):
            if loading:
                return 503, LOADING_BODY
            if not isinstance(body, dict):
                return 400, {"error": {"code": 400, "message": "JSON object expected"}}
            if path == CHAT_PATH:
                return self._chat(body, scenario, options)
            return self._systemone(body, scenario, options)
        return 404, {"error": {"code": 404, "message": f"no route for {method} {path}"}}

    def _model(self, options: Mapping[str, Any], default_tier: str) -> str:
        """The one model path /props and the answers report; recorded props are the source."""
        explicit = options.get("model") or options.get("model_path")
        if explicit:
            return str(explicit)
        tier = options.get("tier", default_tier)
        return str(self._recorded["props"][tier]["model_path"])

    def _props(self, options: Mapping[str, Any]) -> Any:
        if "props" in options:
            return copy.deepcopy(options["props"])
        recorded = self._recorded["props"][options.get("tier", "smart")]
        return {
            "model_path": self._model(options, "smart"),
            "total_slots": options.get("total_slots", recorded["total_slots"]),
            "default_generation_settings": {
                "n_ctx": options.get(
                    "n_ctx", recorded["default_generation_settings"]["n_ctx"]
                )
            },
        }

    def _replay(self, kind: str, body: Mapping[str, Any]) -> Optional[dict[str, Any]]:
        for exchange in self._recorded[kind]:
            if exchange["request"] == body:
                return copy.deepcopy(exchange["response"])
        return None

    def _chat(
        self, body: dict[str, Any], scenario: str, options: Mapping[str, Any]
    ) -> tuple[int, Any]:
        index = self._next_chat_index()
        if scenario == "context_overflow":
            overflow = self._recorded["context_overflow"]
            return overflow["status"], copy.deepcopy(overflow["body"])

        response = (
            None if scenario == "think_first" else self._replay("smart_exchanges", body)
        )
        if response is None:
            response = self._synthesize_chat(body, scenario, options)
        if scenario == "missing_model":
            response.pop("model", None)
        elif scenario == "swap_after":
            swap_at = int(options.get("n", DEFAULT_SWAP_AFTER))
            model_a = options.get("model_a", "/models/fake-model-a.gguf")
            model_b = options.get("model_b", "/models/fake-model-b.gguf")
            response["model"] = model_a if index < swap_at else model_b
        if "alias" in options:
            response["model"] = options["alias"]
        return 200, response

    def _synthesize_chat(
        self, body: Mapping[str, Any], scenario: str, options: Mapping[str, Any]
    ) -> dict[str, Any]:
        messages = body.get("messages") or [{}]
        prompt = str(messages[-1].get("content", ""))
        if scenario == "think_first":
            tokens = [(THINK_TOKEN, -0.001), ("Okay", -7.0), ("Let", -9.0)]
        else:
            probs = options.get("first_token_probs") or _default_probs(
                _prompt_targets(prompt)
            )
            tokens = _probs_to_tokens(probs)
        return _chat_response(
            self._model(options, "smart"), tokens, max(1, len(prompt) // 4)
        )

    def _systemone(
        self, body: dict[str, Any], scenario: str, options: Mapping[str, Any]
    ) -> tuple[int, Any]:
        questions = body.get("questions")
        if not isinstance(questions, dict):
            return 400, {"error": {"code": 400, "message": "questions object expected"}}
        if scenario != "wrong_kev_shape":
            replayed = self._replay("kev_exchanges", body)
            if replayed is not None:
                if "alias" in options:
                    replayed["model"] = options["alias"]
                return 200, replayed
        if scenario == "wrong_kev_shape":
            answers = {key: {"value": 0.9} for key in questions}
        else:
            answers = {
                key: _kev_answer(spec, options) for key, spec in questions.items()
            }
        return 200, {
            "answers": answers,
            "model": options.get("alias") or self._model(options, "fast"),
            "usage": {
                "input_tokens": max(1, len(str(body.get("state", ""))) // 4),
                "output_tokens": 0,
            },
        }
