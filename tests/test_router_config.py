import os
import json
from unittest.mock import patch, mock_open
from jev_mcp import router_config
from jev_mcp.router_config import load_router_config, save_router_config


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
