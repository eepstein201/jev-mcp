"""The jev_train_lora tool's validation branches; the training steps are lora_training's."""

import json

from jev_mcp.tools.training import jev_train_lora


def test_a_dataset_outside_the_allowed_roots_is_blocked(monkeypatch):
    monkeypatch.setattr("jev_mcp.security.is_safe_path", lambda path: False)

    result = json.loads(jev_train_lora(dataset_path="/etc/secrets.jsonl"))

    assert result["status"] == "BLOCKED"


def test_missing_mlx_lm_reports_not_available(tmp_path, monkeypatch):
    dataset = tmp_path / "train.jsonl"
    dataset.write_text("{}\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("jev_mcp.security.is_safe_path", lambda path: True)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)

    result = json.loads(jev_train_lora(dataset_path=str(dataset)))

    assert result["status"] == "NOT_AVAILABLE"
