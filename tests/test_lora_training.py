import json
import os
from unittest.mock import MagicMock, patch

import pytest

from jev_mcp import lora_training, server

SMART_LABEL = "com.jev.mlx_server_smart"


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A throwaway HOME so ~/.jev and ~/Library/LaunchAgents stay inside the test."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("JEV_LLAMA_CPP_SRC", raising=False)
    return tmp_path


@pytest.fixture
def converter(home, monkeypatch):
    src = home / "llama.cpp-src"
    src.mkdir()
    (src / "convert_hf_to_gguf.py").write_text("# stand-in converter\n")
    monkeypatch.setenv("JEV_LLAMA_CPP_SRC", str(src))
    return src


def _agent(home, label, model_path):
    agents = home / "Library" / "LaunchAgents"
    agents.mkdir(parents=True, exist_ok=True)
    (agents / f"{label}.plist").write_text(f"<plist><string>-m</string><string>{model_path}</string></plist>")


def _dataset(home):
    ds = home / "mydata.jsonl"
    ds.write_text('{"messages": []}\n')
    return str(ds)


def _started_script(mock_popen):
    with open(mock_popen.call_args[0][0][1]) as f:
        return f.read()


# --- finding the launchd agent that serves a model ---


def test_label_is_the_agent_whose_plist_names_the_model(home):
    _agent(home, "com.jev.mlx_server_fast", "/models/kev.gguf")
    _agent(home, SMART_LABEL, "/models/qwen.gguf")

    assert lora_training.find_launchd_label("/models/qwen.gguf") == SMART_LABEL


def test_no_label_when_no_agent_serves_the_model(home):
    _agent(home, "com.jev.mlx_server_fast", "/models/kev.gguf")

    assert lora_training.find_launchd_label("/models/qwen.gguf") is None


def test_no_label_when_there_is_no_launch_agents_directory(home):
    assert lora_training.find_launchd_label("/models/qwen.gguf") is None


# --- converter prerequisites ---


def test_missing_converter_is_reported_with_where_it_was_looked_for(home):
    problem = lora_training.converter_problem()

    assert problem is not None
    assert "convert_hf_to_gguf.py" in problem
    assert str(home / ".jev" / "llama.cpp-src") in problem


def test_no_problem_when_the_converter_and_its_modules_are_present(converter):
    with patch("importlib.util.find_spec", return_value=object()):
        assert lora_training.converter_problem() is None


def test_missing_converter_dependency_is_named(converter):
    with patch("importlib.util.find_spec", side_effect=lambda name: None if name == "torch" else object()):
        problem = lora_training.converter_problem()

    assert problem is not None and "torch" in problem


# --- the fuse script ---


def _fuse_script(target=None, label=None):
    return lora_training.build_fuse_script(
        python="/venv/bin/python",
        model_name="mlx-community/Qwen2.5-7B-Instruct-4bit",
        adapter_path="/adapters/run1",
        converter_src="/src/llama.cpp",
        target_gguf_path=target,
        label=label,
    )


def test_fuse_script_dequantizes_to_a_directory_instead_of_exporting_gguf():
    # mlx_lm.fuse --export-gguf only supports unquantized llama/mistral/mixtral models.
    script = _fuse_script()

    assert "--dequantize" in script
    assert "--save-path /adapters/run1/fused" in script
    assert "--export-gguf" not in script


def test_fuse_script_converts_with_the_local_converter_then_quantizes():
    script = _fuse_script()

    assert "/src/llama.cpp" in script
    assert "convert_hf_to_gguf.py" in script
    assert "llama-quantize --allow-requantize" in script
    assert script.index("--dequantize") < script.index("convert_hf_to_gguf.py") < script.index("llama-quantize")
    assert "curl" not in script


def test_fuse_script_frees_the_dequantized_copy_before_quantizing():
    # The dequantized 7B model is ~14 GB; it must be gone before the next file is written.
    script = _fuse_script()

    assert script.index("rm -rf /adapters/run1/fused") < script.index("llama-quantize")


def test_fuse_script_without_a_target_leaves_the_served_model_alone():
    script = _fuse_script()

    assert "launchctl" not in script
    assert ".bak" not in script


def test_fuse_script_backs_up_the_original_once_and_never_overwrites_that_backup():
    script = _fuse_script(target="/models/qwen.gguf", label=SMART_LABEL)

    assert "[ ! -f /models/qwen.gguf.bak ]" in script
    assert "mv /models/qwen.gguf /models/qwen.gguf.bak" in script


def test_fuse_script_restarts_only_the_agent_serving_the_replaced_model():
    script = _fuse_script(target="/models/qwen.gguf", label=SMART_LABEL)

    assert f"launchctl kickstart -k gui/$(id -u)/{SMART_LABEL}" in script
    assert "pgrep" not in script
    assert "nohup" not in script
    assert "kill " not in script


def test_fuse_script_says_so_when_no_agent_serves_the_target():
    script = _fuse_script(target="/models/qwen.gguf", label=None)

    assert "launchctl" not in script
    assert "No launchd agent" in script


def test_fuse_script_quotes_paths_with_spaces():
    script = _fuse_script(target="/my models/qwen.gguf", label=SMART_LABEL)

    assert "'/my models/qwen.gguf'" in script


# --- restore ---


def test_restore_puts_the_backup_back_and_restarts_its_agent(home):
    target = home / "qwen.gguf"
    target.write_text("fine-tuned")
    (home / "qwen.gguf.bak").write_text("original")
    _agent(home, SMART_LABEL, str(target))

    with patch("subprocess.run", return_value=MagicMock(returncode=0)) as run:
        result = lora_training.restore_model(str(target))

    assert result["status"] == "RESTORED"
    assert target.read_text() == "original"
    assert not (home / "qwen.gguf.bak").exists()
    assert result["restarted_agent"] == SMART_LABEL
    assert run.call_args[0][0] == ["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/{SMART_LABEL}"]


def test_restore_without_a_backup_changes_nothing(home):
    target = home / "qwen.gguf"
    target.write_text("current")

    with patch("subprocess.run") as run:
        result = lora_training.restore_model(str(target))

    assert result["status"] == "ERROR"
    assert ".bak" in result["message"]
    assert target.read_text() == "current"
    run.assert_not_called()


def test_restore_reports_when_no_agent_was_restarted(home):
    target = home / "qwen.gguf"
    (home / "qwen.gguf.bak").write_text("original")

    with patch("subprocess.run") as run:
        result = lora_training.restore_model(str(target))

    assert result["status"] == "RESTORED"
    assert result["restarted_agent"] is None
    run.assert_not_called()


def test_restore_reports_a_failed_restart(home):
    target = home / "qwen.gguf"
    (home / "qwen.gguf.bak").write_text("original")
    _agent(home, SMART_LABEL, str(target))

    with patch("subprocess.run", return_value=MagicMock(returncode=3, stderr="Could not find service")):
        result = lora_training.restore_model(str(target))

    assert result["status"] == "RESTORED"
    assert result["restarted_agent"] is None
    assert "Could not find service" in result["restart_error"]


# --- the train tool ---


@patch("subprocess.Popen")
def test_train_with_fuse_refuses_to_start_without_the_converter(mock_popen, home):
    res = json.loads(server.jev_train_lora(_dataset(home)))

    assert res["status"] == "NOT_AVAILABLE"
    assert "convert_hf_to_gguf.py" in res["message"]
    assert "fuse=False" in res["message"]
    mock_popen.assert_not_called()


@patch("subprocess.Popen")
def test_train_with_fuse_writes_the_dequantize_convert_quantize_script(mock_popen, home, converter):
    mock_popen.return_value = MagicMock(pid=7)
    target = home / "qwen.gguf"
    target.write_text("original")
    _agent(home, SMART_LABEL, str(target))

    res = json.loads(server.jev_train_lora(_dataset(home), target_gguf_path=str(target), iters=50))

    script = _started_script(mock_popen)
    assert res["status"] == "TRAINING_STARTED", res.get("message")
    assert "--dequantize" in script
    assert str(converter) in script
    assert f"launchctl kickstart -k gui/$(id -u)/{SMART_LABEL}" in script
    assert res["restart_agent"] == SMART_LABEL


@patch("subprocess.Popen")
def test_train_rejects_a_target_that_is_not_a_gguf_file(mock_popen, home, converter):
    res = json.loads(server.jev_train_lora(_dataset(home), target_gguf_path=str(home / "notes.txt")))

    assert res["status"] == "ERROR"
    assert ".gguf" in res["message"]
    mock_popen.assert_not_called()


@patch("subprocess.Popen")
def test_train_rejects_a_target_in_a_sensitive_location(mock_popen, home, converter):
    res = json.loads(server.jev_train_lora(_dataset(home), target_gguf_path="~/.ssh/model.gguf"))

    assert res["status"] == "BLOCKED"
    mock_popen.assert_not_called()


def test_restore_action_restores_the_backup_through_the_tool(home):
    target = home / "qwen.gguf"
    target.write_text("fine-tuned")
    (home / "qwen.gguf.bak").write_text("original")

    res = json.loads(server.jev_train_lora(action="restore", target_gguf_path=str(target)))

    assert res["status"] == "RESTORED"
    assert target.read_text() == "original"


def test_restore_action_needs_a_target(home):
    res = json.loads(server.jev_train_lora(action="restore"))

    assert res["status"] == "ERROR"
    assert "target_gguf_path" in res["message"]
