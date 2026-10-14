import os
import json
from datetime import datetime
from unittest.mock import patch, mock_open

import pytest

from jev_mcp import router_config
from jev_mcp.router_config import (
    DEFAULT_TEMPERATURE,
    RouterConfig,
    get_temperature,
    is_calibrated,
    load_router_config,
    model_key,
    save_router_config,
    with_temperature,
)


def test_save_router_config_is_atomic_on_failure(monkeypatch, tmp_path):
    # The live config survives a write that dies partway through.
    save_router_config({"buckets": {"b1": "original"}, "rules": []})
    target = router_config.ROUTER_CONFIG_PATH
    before = open(target).read()

    import json as _json

    def exploding_dump(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(_json, "dump", exploding_dump)

    import pytest

    with pytest.raises(OSError):
        save_router_config({"buckets": {"b1": "clobbered"}, "rules": []})
    assert open(target).read() == before


def test_save_router_config_cleans_up_temp_on_rename_failure(monkeypatch):
    save_router_config({"buckets": {"b1": "original"}, "rules": []})
    target = router_config.ROUTER_CONFIG_PATH
    before = open(target).read()

    def failing_replace(src, dst):
        raise OSError("rename blocked")

    monkeypatch.setattr(router_config.os, "replace", failing_replace)

    import pytest

    with pytest.raises(OSError):
        save_router_config({"buckets": {"b1": "clobbered"}, "rules": []})
    assert open(target).read() == before
    assert not [n for n in os.listdir(os.path.dirname(target)) if n.endswith(".tmp")]


def test_seed_returns_none_for_corrupt_config():
    from jev_mcp.router_config import seed_defaults_for_client

    os.makedirs(os.path.dirname(router_config.ROUTER_CONFIG_PATH), exist_ok=True)
    with open(router_config.ROUTER_CONFIG_PATH, "w") as f:
        f.write("{not json")
    assert seed_defaults_for_client("Claude Code") is None


def test_load_router_config_default():
    with patch("os.path.exists", return_value=False):
        config = load_router_config()
        assert "buckets" in config
        assert "rules" in config
        assert config["buckets"]["b1"] == "claude-haiku-5-5"


def test_load_router_config_existing():
    mock_config = {"buckets": {"b1": "test-model"}, "rules": []}
    mock_json = json.dumps(mock_config)

    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=mock_json)):
            config = load_router_config()
            assert config["buckets"]["b1"] == "test-model"


def test_every_default_key_is_declared_on_the_typed_config():
    # The autouse fixture gives this a fresh temp config path with no file, so
    # load_router_config() returns the default config: every key it carries must
    # be declared on RouterConfig, or the TypedDict has drifted from the loaders.
    assert set(load_router_config()) <= set(RouterConfig.__annotations__)


def test_save_router_config():
    # Runs against the conftest-provided temp config path.
    mock_config = {"buckets": {"b1": "test-model"}, "rules": []}
    save_router_config(mock_config)
    assert load_router_config() == mock_config
    assert not os.path.exists(router_config.ROUTER_CONFIG_PATH + ".tmp")


def test_save_router_config_uses_a_fresh_temp_file_for_every_write(monkeypatch):
    # Two writers sharing one temp name would interleave and rename a mixed file.
    real_replace = os.replace
    temp_names = []

    def recording_replace(src, dst):
        temp_names.append(src)
        real_replace(src, dst)

    monkeypatch.setattr(router_config.os, "replace", recording_replace)

    save_router_config({"buckets": {"b1": "one"}, "rules": []})
    save_router_config({"buckets": {"b1": "two"}, "rules": []})

    assert len(set(temp_names)) == 2
    folder = os.path.dirname(router_config.ROUTER_CONFIG_PATH)
    assert os.path.basename(router_config.ROUTER_CONFIG_PATH) in os.listdir(folder)
    assert not [n for n in os.listdir(folder) if n.endswith(".tmp")]


# ---- Per-model temperature accessor ----

QWEN = "/Users/me/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
LLAMA = "/Users/me/models/Llama-3.1-8B-Instruct-Q4_K_M.gguf"


def test_the_default_config_carries_an_empty_temperatures_map():
    assert load_router_config()["temperatures"] == {}


def test_the_default_temperature_is_one():
    assert DEFAULT_TEMPERATURE == 1.0


@pytest.mark.parametrize(
    "model_id,expected",
    [
        (
            "/Users/me/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
            "qwen2.5-7b-instruct-q4_k_m.gguf",
        ),
        ("Qwen2.5-7B.GGUF", "qwen2.5-7b.gguf"),
        ("C:\\models\\Llama.gguf", "llama.gguf"),
        ("  /m/Model.gguf  ", "model.gguf"),
    ],
)
def test_model_key_is_the_lower_cased_basename(model_id, expected):
    assert model_key(model_id) == expected


def test_a_stored_entry_wins_over_everything():
    config = with_temperature(
        {"buckets": {}, "rules": [], "fitted_temperature": 1.7}, QWEN, 2.4
    )

    assert get_temperature(config, QWEN) == 2.4
    assert is_calibrated(config, QWEN) is True


def test_the_entry_is_found_through_any_path_to_the_same_file():
    config = with_temperature({"buckets": {}, "rules": []}, QWEN, 2.4)

    assert get_temperature(config, "/elsewhere/QWEN2.5-7B-Instruct-Q4_K_M.gguf") == 2.4


def test_the_legacy_global_applies_to_qwen2_5_7b_only():
    config = {"buckets": {}, "rules": [], "fitted_temperature": 1.7}

    assert get_temperature(config, QWEN) == 1.7
    assert is_calibrated(config, QWEN) is True
    assert get_temperature(config, LLAMA) == DEFAULT_TEMPERATURE
    assert is_calibrated(config, LLAMA) is False


def test_a_model_with_no_entry_and_no_legacy_value_gets_the_default():
    config = {"buckets": {}, "rules": []}

    assert get_temperature(config, QWEN) == DEFAULT_TEMPERATURE
    assert is_calibrated(config, QWEN) is False


def test_an_entry_for_a_different_size_is_not_used():
    # The same file name with different bytes is a different model.
    config = with_temperature({"buckets": {}, "rules": []}, LLAMA, 2.4, model_size=100)

    assert get_temperature(config, LLAMA, model_size=200) == DEFAULT_TEMPERATURE
    assert is_calibrated(config, LLAMA, model_size=200) is False


def test_a_size_mismatch_falls_through_to_the_legacy_rule():
    config = with_temperature(
        {"buckets": {}, "rules": [], "fitted_temperature": 1.7},
        QWEN,
        2.4,
        model_size=100,
    )

    assert get_temperature(config, QWEN, model_size=200) == 1.7


@pytest.mark.parametrize(
    "stored,reported", [(100, None), (None, 100), (None, None), (100, 100)]
)
def test_an_entry_applies_unless_both_sizes_are_known_and_differ(stored, reported):
    config = with_temperature(
        {"buckets": {}, "rules": []}, LLAMA, 2.4, model_size=stored
    )

    assert get_temperature(config, LLAMA, model_size=reported) == 2.4


def test_with_temperature_returns_a_new_config_and_leaves_the_original_untouched():
    original = {
        "buckets": {"b1": "m"},
        "rules": [],
        "temperatures": {
            "kept.gguf": {"temperature": 1.5, "model_size": None, "updated_at": "x"}
        },
    }

    updated = with_temperature(
        original, LLAMA, 0.8, model_size=7, now=datetime(2026, 1, 2, 3, 4, 5)
    )

    assert original["temperatures"] == {
        "kept.gguf": {"temperature": 1.5, "model_size": None, "updated_at": "x"}
    }
    assert "temperature_updated_at" not in original
    assert updated["temperatures"]["kept.gguf"]["temperature"] == 1.5
    assert updated["temperatures"][model_key(LLAMA)] == {
        "temperature": 0.8,
        "model_size": 7,
        "updated_at": "2026-01-02 03:04:05",
    }
    assert updated["temperature_updated_at"] == "2026-01-02 03:04:05"
    assert updated["buckets"] == {"b1": "m"}


def test_with_temperature_stamps_the_current_time_by_default():
    updated = with_temperature({"buckets": {}, "rules": []}, LLAMA, 0.8)

    stamp = updated["temperature_updated_at"]
    assert datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")


@pytest.mark.parametrize("model_id", ["unknown_model", "", "  ", "/models/"])
def test_with_temperature_refuses_a_model_the_engine_never_reported(model_id):
    config = {"buckets": {}, "rules": []}

    with pytest.raises(ValueError, match="did not report a model"):
        with_temperature(config, model_id, 1.5)

    assert config == {"buckets": {}, "rules": []}


def test_temperatures_survive_a_save_and_load_round_trip():
    save_router_config(
        with_temperature({"buckets": {}, "rules": []}, LLAMA, 0.8, model_size=7)
    )

    assert get_temperature(load_router_config(), LLAMA, model_size=7) == 0.8


# ---- A hand-edited temperatures section is never trusted and never raises ----

GOOD = {"temperature": 2.4, "model_size": None, "updated_at": "2026-01-01 00:00:00"}
LLAMA_KEY = model_key(LLAMA)
QWEN_KEY = model_key(QWEN)

MALFORMED_TEMPERATURES = {
    "temperatures is null": None,
    "temperatures is a list": [GOOD],
    "temperatures is a string": "hot",
    "entry is a number": {QWEN_KEY: 2.4, LLAMA_KEY: 2.4},
    "entry is null": {QWEN_KEY: None, LLAMA_KEY: None},
    "entry lacks temperature": {
        QWEN_KEY: {"updated_at": "x"},
        LLAMA_KEY: {"updated_at": "x"},
    },
    "entry lacks updated_at": {
        QWEN_KEY: {"temperature": 2.4},
        LLAMA_KEY: {"temperature": 2.4},
    },
    "temperature is text": {
        QWEN_KEY: {**GOOD, "temperature": "hot"},
        LLAMA_KEY: {**GOOD, "temperature": "hot"},
    },
    "temperature is a boolean": {
        QWEN_KEY: {**GOOD, "temperature": True},
        LLAMA_KEY: {**GOOD, "temperature": True},
    },
    "temperature is not finite": {
        QWEN_KEY: {**GOOD, "temperature": float("nan")},
        LLAMA_KEY: {**GOOD, "temperature": float("inf")},
    },
    "temperature is not positive": {
        QWEN_KEY: {**GOOD, "temperature": 0},
        LLAMA_KEY: {**GOOD, "temperature": -1.0},
    },
    "updated_at is not text": {
        QWEN_KEY: {**GOOD, "updated_at": 5},
        LLAMA_KEY: {**GOOD, "updated_at": 5},
    },
    "model_size is text": {
        QWEN_KEY: {**GOOD, "model_size": "big"},
        LLAMA_KEY: {**GOOD, "model_size": "big"},
    },
}


@pytest.mark.parametrize(
    "temperatures", MALFORMED_TEMPERATURES.values(), ids=MALFORMED_TEMPERATURES.keys()
)
def test_a_malformed_entry_counts_as_missing_and_falls_through(temperatures):
    config = {
        "buckets": {},
        "rules": [],
        "fitted_temperature": 1.7,
        "temperatures": temperatures,
    }

    assert get_temperature(config, QWEN) == 1.7
    assert is_calibrated(config, QWEN) is True
    assert get_temperature(config, LLAMA) == DEFAULT_TEMPERATURE
    assert is_calibrated(config, LLAMA) is False
    assert router_config.temperature_entry(config, LLAMA, None) is None


def test_a_well_formed_entry_is_returned_as_is():
    config = {"buckets": {}, "rules": [], "temperatures": {LLAMA_KEY: GOOD}}

    assert router_config.temperature_entry(config, LLAMA, None) == GOOD
    assert router_config.valid_temperatures(config) == {LLAMA_KEY: GOOD}


def test_valid_temperatures_drops_only_the_malformed_entries():
    config = {
        "buckets": {},
        "rules": [],
        "temperatures": {LLAMA_KEY: GOOD, QWEN_KEY: "broken"},
    }

    assert router_config.valid_temperatures(config) == {LLAMA_KEY: GOOD}


@pytest.mark.parametrize("legacy", ["hot", None, True, float("nan"), -2.0, 0])
def test_a_malformed_legacy_global_is_ignored(legacy):
    config = {"buckets": {}, "rules": [], "fitted_temperature": legacy}

    assert get_temperature(config, QWEN) == DEFAULT_TEMPERATURE
    assert is_calibrated(config, QWEN) is False


@pytest.mark.parametrize("temperatures", [None, [GOOD], "hot", {LLAMA_KEY: 5}])
def test_with_temperature_replaces_a_malformed_section_without_raising(temperatures):
    config = {"buckets": {}, "rules": [], "temperatures": temperatures}

    updated = with_temperature(config, LLAMA, 0.8)

    assert get_temperature(updated, LLAMA) == 0.8
