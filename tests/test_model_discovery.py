"""Served-model discovery (GET /props) and the fresh-process temperature tools.

A fresh server process has not run an evaluation, so the smart engine has never
reported a model. These tests pin the discovery probe and the temperature tools
against the fake llama-server, including the usage eval's 1.7 legacy value.
"""

import json
import socket
from types import SimpleNamespace

import pytest

from jev_mcp import app, router_config, server
from jev_mcp.daemon_provider import UNKNOWN_MODEL, DaemonProvider
from tests.evals.usage import seed_home
from tests.fake_llama_server import FakeLlamaServer, load_recorded

CHAT = "/v1/chat/completions"
SMART_MODEL = load_recorded()["props"]["smart"]["model_path"]
SMART_KEY = router_config.model_key(SMART_MODEL)


@pytest.fixture
def daemon():
    with FakeLlamaServer() as running:
        yield running


def _provider(daemon: FakeLlamaServer) -> DaemonProvider:
    return DaemonProvider(base_url=daemon.url + CHAT)


def _dead_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


# --- discover_model ------------------------------------------------------------------


def test_discover_model_names_the_served_model_without_running_inference(daemon):
    provider = _provider(daemon)

    discovered = provider.discover_model()

    assert discovered == SMART_MODEL
    assert provider.current_model_id == SMART_MODEL
    # A model probe is a plain GET /props: no completion request was made.
    assert [(method, path) for method, path, _ in daemon.requests] == [
        ("GET", "/props")
    ]


def test_discover_model_records_the_file_size_of_the_served_model(daemon, tmp_path):
    gguf = tmp_path / "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    gguf.write_bytes(b"\x00" * 4096)
    daemon.set_scenario("normal", model=str(gguf))
    provider = _provider(daemon)

    assert provider.discover_model() == str(gguf)
    assert provider.model_size == 4096


def test_discover_model_returns_none_while_the_daemon_is_loading(daemon):
    daemon.set_scenario("loading")
    provider = _provider(daemon)

    assert provider.discover_model() is None
    assert provider.current_model_id == UNKNOWN_MODEL


@pytest.mark.parametrize(
    "props",
    [
        {},
        {"model_path": 42},
        {"model_path": "   "},
        {"model_path": "unknown_model"},
        ["not", "an", "object"],
    ],
    ids=["missing", "not-a-string", "blank", "unknown", "not-an-object"],
)
def test_discover_model_rejects_a_props_body_without_a_usable_model(daemon, props):
    daemon.set_scenario("normal", props=props)
    provider = _provider(daemon)

    assert provider.discover_model() is None
    assert provider.current_model_id == UNKNOWN_MODEL


def test_discover_model_returns_none_when_the_daemon_is_unreachable():
    provider = DaemonProvider(base_url=f"http://127.0.0.1:{_dead_port()}{CHAT}")

    assert provider.discover_model() is None
    assert provider.current_model_id == UNKNOWN_MODEL


# --- reported_smart_model probes once before giving up --------------------------------


@pytest.fixture
def fresh_smart(daemon, monkeypatch):
    """The shared smart engine as a fresh process sees it, pointed at the fake daemon.

    monkeypatch restores every attribute afterwards, so a discovery inside the test
    never leaks into tests that expect the shared engine's model to be unknown.
    """
    smart = app.provider.smart_provider
    monkeypatch.setattr(smart, "base_url", daemon.url + CHAT)
    monkeypatch.setattr(smart, "current_model_id", UNKNOWN_MODEL)
    monkeypatch.setattr(smart, "model_size", None)
    return smart


def test_reported_smart_model_discovers_once_and_then_reads_the_cached_id(
    daemon, fresh_smart
):
    first = app.reported_smart_model()
    probes = len(daemon.requests)
    second = app.reported_smart_model()

    assert first == (SMART_MODEL, None)
    assert second == first
    assert len(daemon.requests) == probes  # the second call asked the engine nothing


def test_reported_smart_model_tolerates_a_provider_that_cannot_discover():
    smart = SimpleNamespace(current_model_id=UNKNOWN_MODEL)

    assert app.reported_smart_model(smart) == ("", None)


def test_reported_smart_model_tolerates_a_discovery_that_raises():
    def blow_up():
        raise RuntimeError("probe failed")

    smart = SimpleNamespace(current_model_id=UNKNOWN_MODEL, discover_model=blow_up)

    assert app.reported_smart_model(smart) == ("", None)


# --- the temperature tools in a fresh process -----------------------------------------


def _seed_legacy_config():
    """The usage eval's seeded home: the legacy 1.7 was fitted on qwen2.5-7b."""
    router_config.save_router_config(dict(seed_home.ROUTER_CONFIG))


def test_a_fresh_process_views_the_legacy_temperature_for_qwen(daemon, fresh_smart):
    # The usage eval's "configured temperature" question: once discovery knows the
    # engine serves qwen2.5-7b, view reports 1.7 through the legacy rule.
    _seed_legacy_config()

    out = json.loads(server.jev_manage_temperature("view"))

    assert out["status"] == "SUCCESS"
    assert out["model"] == SMART_KEY
    assert out["calibrated"] is True
    assert out["current_temperature"] == 1.7
    assert out["updated_at"] == "2026-01-01 00:00:00"


def test_a_fresh_process_can_set_and_reset_the_temperature(daemon, fresh_smart):
    _seed_legacy_config()

    set_out = json.loads(server.jev_manage_temperature("set", 2.5))
    reset_out = json.loads(server.jev_manage_temperature("reset"))
    view_out = json.loads(server.jev_manage_temperature("view"))

    assert set_out["status"] == "SUCCESS"
    assert set_out["model"] == SMART_KEY
    assert set_out["current_temperature"] == 2.5
    assert reset_out["status"] == "SUCCESS"
    assert reset_out["current_temperature"] == 1.0
    assert (view_out["current_temperature"], view_out["calibrated"]) == (1.0, True)
    # Reset stores an explicit 1.0 entry, which overrides the legacy 1.7.
    entry = router_config.load_router_config()["temperatures"][SMART_KEY]
    assert entry["temperature"] == 1.0


# --- the engine genuinely down --------------------------------------------------------


@pytest.fixture
def dead_smart(monkeypatch):
    """The shared smart engine pointing at a port where no daemon listens."""
    smart = app.provider.smart_provider
    monkeypatch.setattr(smart, "base_url", f"http://127.0.0.1:{_dead_port()}{CHAT}")
    monkeypatch.setattr(smart, "current_model_id", UNKNOWN_MODEL)
    monkeypatch.setattr(smart, "model_size", None)
    return smart


def test_set_and_reset_name_the_unavailable_engine_when_it_is_down(dead_smart):
    _seed_legacy_config()
    before = router_config.load_router_config()

    set_out = json.loads(server.jev_manage_temperature("set", 2.0))
    reset_out = json.loads(server.jev_manage_temperature("reset"))

    assert set_out["status"] == "ERROR"
    assert reset_out["status"] == "ERROR"
    for out in (set_out, reset_out):
        assert "did not report a model" in out["message"]
        assert "unavailable" in out["message"]
    assert router_config.load_router_config() == before


def test_view_degrades_when_the_engine_is_down_and_hides_an_invalid_legacy_value(
    dead_smart,
):
    router_config.save_router_config(
        {"buckets": {}, "rules": [], "fitted_temperature": "hot"}
    )

    out = json.loads(server.jev_manage_temperature("view"))

    assert out["status"] == "SUCCESS"
    assert out["model"] is None
    assert out["current_temperature"] == router_config.DEFAULT_TEMPERATURE
    assert out["legacy_fitted_temperature"] is None  # not the raw unusable "hot"
