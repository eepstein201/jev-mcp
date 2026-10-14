import io
import json
import os
import tarfile
from unittest.mock import MagicMock, patch

import pytest

from jev_mcp import lora_training, server

SMART_LABEL = "com.jev.mlx_server_smart"
COMMIT = "abc1234de"
_real_installed_llama_commit = lora_training.installed_llama_commit


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A throwaway HOME so ~/.jev and ~/Library/LaunchAgents stay inside the test."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("JEV_LLAMA_CPP_SRC", raising=False)
    # No llama-server in the test environment unless a test says otherwise.
    monkeypatch.setattr(lora_training, "installed_llama_commit", lambda: None)
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
    (agents / f"{label}.plist").write_text(
        f"<plist><string>-m</string><string>{model_path}</string></plist>"
    )


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
    with patch(
        "importlib.util.find_spec",
        side_effect=lambda name: None if name == "torch" else object(),
    ):
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
    assert (
        script.index("--dequantize")
        < script.index("convert_hf_to_gguf.py")
        < script.index("llama-quantize")
    )
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
    assert run.call_args[0][0] == [
        "launchctl",
        "kickstart",
        "-k",
        f"gui/{os.getuid()}/{SMART_LABEL}",
    ]


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

    with patch(
        "subprocess.run",
        return_value=MagicMock(returncode=3, stderr="Could not find service"),
    ):
        result = lora_training.restore_model(str(target))

    assert result["status"] == "RESTORED"
    assert result["restarted_agent"] is None
    assert "Could not find service" in result["restart_error"]


# --- the jev_train_lora tool ---


@patch("subprocess.Popen")
def test_train_with_fuse_refuses_to_start_without_the_converter(mock_popen, home):
    res = json.loads(server.jev_train_lora(_dataset(home)))

    assert res["status"] == "NOT_AVAILABLE"
    assert "convert_hf_to_gguf.py" in res["message"]
    assert "fuse=False" in res["message"]
    mock_popen.assert_not_called()


@patch("subprocess.Popen")
def test_train_with_fuse_writes_the_dequantize_convert_quantize_script(
    mock_popen, home, converter, monkeypatch
):
    # The target must be the active profile's smart model (the repo table makes the
    # default profile active), so name this target as that model.
    _use_profile(monkeypatch)
    mock_popen.return_value = MagicMock(pid=7)
    target = home / "qwen.gguf"
    target.write_text("original")
    _agent(home, SMART_LABEL, str(target))

    res = json.loads(
        server.jev_train_lora(_dataset(home), target_gguf_path=str(target), iters=50)
    )

    script = _started_script(mock_popen)
    assert res["status"] == "TRAINING_STARTED", res.get("message")
    assert "--dequantize" in script
    assert str(converter) in script
    assert f"launchctl kickstart -k gui/$(id -u)/{SMART_LABEL}" in script
    assert res["restart_agent"] == SMART_LABEL


@patch("subprocess.Popen")
def test_train_rejects_a_target_that_is_not_a_gguf_file(mock_popen, home, converter):
    res = json.loads(
        server.jev_train_lora(_dataset(home), target_gguf_path=str(home / "notes.txt"))
    )

    assert res["status"] == "ERROR"
    assert ".gguf" in res["message"]
    mock_popen.assert_not_called()


@patch("subprocess.Popen")
def test_train_rejects_a_target_in_a_sensitive_location(mock_popen, home, converter):
    res = json.loads(
        server.jev_train_lora(_dataset(home), target_gguf_path="~/.ssh/model.gguf")
    )

    assert res["status"] == "BLOCKED"
    mock_popen.assert_not_called()


def test_restore_action_restores_the_backup_through_the_tool(home):
    target = home / "qwen.gguf"
    target.write_text("fine-tuned")
    (home / "qwen.gguf.bak").write_text("original")

    res = json.loads(
        server.jev_train_lora(action="restore", target_gguf_path=str(target))
    )

    assert res["status"] == "RESTORED"
    assert target.read_text() == "original"


def test_restore_action_needs_a_target(home):
    res = json.loads(server.jev_train_lora(action="restore"))

    assert res["status"] == "ERROR"
    assert "target_gguf_path" in res["message"]


# --- keeping the converter in step with the installed llama.cpp ---


def _archive(files):
    """A GitHub-style source tarball: every path sits under one top-level directory."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(f"llama.cpp-{COMMIT}full/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    buf.seek(0)
    return buf


def _good_archive(marker_text="# converter"):
    return _archive(
        {
            "convert_hf_to_gguf.py": marker_text,
            "conversion/__init__.py": "",
            "gguf-py/gguf/__init__.py": "",
            "README.md": "not needed",
            "tools/server/server.cpp": "not needed",
        }
    )


@pytest.fixture
def llama(home, monkeypatch):
    """llama-server reports COMMIT; the converter's Python modules count as installed."""
    monkeypatch.setattr(lora_training, "installed_llama_commit", lambda: COMMIT)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    return home / ".jev" / "llama.cpp-src"


def test_converter_is_downloaded_for_the_installed_commit_when_missing(llama):
    with patch("urllib.request.urlopen", return_value=_good_archive()) as urlopen:
        problem = lora_training.ensure_converter()

    assert problem is None
    assert COMMIT in urlopen.call_args[0][0]
    assert (llama / "convert_hf_to_gguf.py").is_file()
    assert (llama / "conversion" / "__init__.py").is_file()
    assert (llama / "gguf-py" / "gguf" / "__init__.py").is_file()
    assert lora_training.converter_commit() == COMMIT


def test_only_the_converter_files_are_kept_from_the_archive(llama):
    with patch("urllib.request.urlopen", return_value=_good_archive()):
        lora_training.ensure_converter()

    assert not (llama / "README.md").exists()
    assert not (llama / "tools").exists()


def test_nothing_is_downloaded_when_the_converter_matches_the_installed_commit(llama):
    with patch("urllib.request.urlopen", return_value=_good_archive()):
        lora_training.ensure_converter()

    with patch("urllib.request.urlopen") as urlopen:
        problem = lora_training.ensure_converter()

    assert problem is None
    urlopen.assert_not_called()


def test_converter_is_replaced_after_llama_cpp_is_upgraded(llama, monkeypatch):
    with patch("urllib.request.urlopen", return_value=_good_archive("# old converter")):
        lora_training.ensure_converter()
    (llama / "conversion" / "removed_upstream.py").write_text("stale")
    monkeypatch.setattr(lora_training, "installed_llama_commit", lambda: "fff9999aa")

    with patch(
        "urllib.request.urlopen", return_value=_good_archive("# new converter")
    ) as urlopen:
        problem = lora_training.ensure_converter()

    assert problem is None
    assert "fff9999aa" in urlopen.call_args[0][0]
    assert (llama / "convert_hf_to_gguf.py").read_text() == "# new converter"
    assert not (llama / "conversion" / "removed_upstream.py").exists()
    assert lora_training.converter_commit() == "fff9999aa"


def test_a_directory_the_user_chose_is_never_downloaded_into(converter, monkeypatch):
    monkeypatch.setattr(lora_training, "installed_llama_commit", lambda: COMMIT)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())

    with patch("urllib.request.urlopen") as urlopen:
        problem = lora_training.ensure_converter()

    assert problem is None
    urlopen.assert_not_called()
    assert (converter / "convert_hf_to_gguf.py").read_text() == "# stand-in converter\n"


def test_a_failed_download_with_no_converter_is_reported_with_the_commit(llama):
    with patch("urllib.request.urlopen", side_effect=OSError("network unreachable")):
        problem = lora_training.ensure_converter()

    assert problem is not None
    assert COMMIT in problem and "network unreachable" in problem
    assert not llama.exists()


def test_a_failed_download_keeps_the_converter_already_installed(llama, monkeypatch):
    with patch("urllib.request.urlopen", return_value=_good_archive("# old converter")):
        lora_training.ensure_converter()
    monkeypatch.setattr(lora_training, "installed_llama_commit", lambda: "fff9999aa")

    with patch("urllib.request.urlopen", side_effect=OSError("network unreachable")):
        problem = lora_training.ensure_converter()

    assert problem is None
    assert (llama / "convert_hf_to_gguf.py").read_text() == "# old converter"
    assert lora_training.converter_commit() == COMMIT


def test_an_archive_without_the_converter_is_rejected_and_nothing_is_installed(llama):
    with patch(
        "urllib.request.urlopen", return_value=_archive({"README.md": "nothing useful"})
    ):
        problem = lora_training.ensure_converter()

    assert problem is not None
    assert not llama.exists()


def test_archive_entries_cannot_be_written_outside_the_converter_directory(llama, home):
    archive = _archive(
        {
            "convert_hf_to_gguf.py": "# converter",
            "conversion/../../escaped.py": "bad",
        }
    )

    with patch("urllib.request.urlopen", return_value=archive):
        lora_training.ensure_converter()

    assert not (home / ".jev" / "escaped.py").exists()
    assert not (home / "escaped.py").exists()


def test_without_llama_server_the_existing_converter_is_used_as_is(home):
    with patch("urllib.request.urlopen") as urlopen:
        problem = lora_training.ensure_converter()

    assert problem is not None and "convert_hf_to_gguf.py" in problem
    urlopen.assert_not_called()


def test_installed_commit_is_read_from_llama_server_version():
    out = MagicMock(
        stdout="",
        stderr="version: 0.6.0 (build 11429, commit d81235049)\nbuilt with AppleClang",
    )

    with patch("subprocess.run", return_value=out) as run:
        commit = _real_installed_llama_commit()

    assert commit == "d81235049"
    assert run.call_args[0][0] == ["llama-server", "--version"]


def test_installed_commit_is_unknown_when_llama_server_is_missing_or_unparseable():
    with patch("subprocess.run", side_effect=FileNotFoundError("llama-server")):
        assert _real_installed_llama_commit() is None
    with patch(
        "subprocess.run", return_value=MagicMock(stdout="version: unknown", stderr="")
    ):
        assert _real_installed_llama_commit() is None


@patch("subprocess.Popen")
def test_train_fetches_the_converter_itself_and_reports_its_commit(
    mock_popen, llama, home
):
    mock_popen.return_value = MagicMock(pid=7)

    with patch("urllib.request.urlopen", return_value=_good_archive()):
        res = json.loads(server.jev_train_lora(_dataset(home)))

    assert res["status"] == "TRAINING_STARTED", res.get("message")
    assert res["converter_commit"] == COMMIT
    assert str(llama) in _started_script(mock_popen)


def test_fuse_script_does_not_let_the_shell_expand_a_target_path():
    import shlex

    target = "/models/$(touch pwned).gguf"
    script = _fuse_script(target=target, label=SMART_LABEL)

    assert (
        "echo " + shlex.quote(f"Backed up the original model to {target}.bak") in script
    )
    for line in script.splitlines():
        if "touch pwned" in line:
            assert '"' not in line, line


def test_fuse_script_quotes_the_output_path_in_its_messages():
    import shlex

    script = lora_training.build_fuse_script(
        python="/venv/bin/python",
        model_name="mlx-community/Qwen2.5-7B-Instruct-4bit",
        adapter_path="/my adapters/$(id)",
        converter_src="/src/llama.cpp",
        target_gguf_path=None,
        label=None,
    )

    saved = [line for line in script.splitlines() if "Fused model saved to" in line]
    assert len(saved) == 1
    assert shlex.split(saved[0])[0] == "echo"
    assert '"' not in saved[0]


# --- engine profiles: the target must be the active profile's smart model ---

import plistlib  # noqa: E402

from jev_mcp import engine_profiles  # noqa: E402
from jev_mcp.engine_profiles import ModelSpec, Profile  # noqa: E402


def _model(tier, filename, *, reasoning="auto"):
    return ModelSpec(
        name=filename.lower().removesuffix(".gguf"),
        tier=tier,
        repo="org/repo",
        revision="a" * 40,
        filename=filename,
        sha256="b" * 64,
        parallel=2,
        ctx_per_slot=16384,
        reasoning=reasoning,
        style="sandbox",
    )


def _use_profile(monkeypatch, smart_file="qwen.gguf", *, reasoning="auto"):
    profile = Profile(
        name="lean",
        fast=_model("fast", "kev.gguf"),
        smart=_model("smart", smart_file, reasoning=reasoning),
        gate=0.85,
        eval_only=False,
    )
    monkeypatch.setattr(engine_profiles, "active_profile", lambda *a, **k: profile)
    return profile


@patch("subprocess.Popen")
def test_train_refuses_a_target_that_is_not_the_active_smart_model(
    mock_popen, home, converter, monkeypatch
):
    _use_profile(monkeypatch)
    target = home / "other.gguf"
    target.write_text("original")

    res = json.loads(
        server.jev_train_lora(_dataset(home), target_gguf_path=str(target))
    )

    assert res["status"] == "ERROR"
    assert "other.gguf" in res["message"]
    assert "qwen.gguf" in res["message"]
    assert "lean" in res["message"]
    mock_popen.assert_not_called()
    assert not (home / "other.gguf.bak").exists()


@patch("subprocess.Popen")
def test_train_refuses_the_active_fast_model_as_a_target(
    mock_popen, home, converter, monkeypatch
):
    _use_profile(monkeypatch)
    target = home / "kev.gguf"
    target.write_text("original")

    res = json.loads(
        server.jev_train_lora(_dataset(home), target_gguf_path=str(target))
    )

    assert res["status"] == "ERROR"
    mock_popen.assert_not_called()


@patch("subprocess.Popen")
def test_train_accepts_the_active_smart_model_as_a_target(
    mock_popen, home, converter, monkeypatch
):
    _use_profile(monkeypatch)
    mock_popen.return_value = MagicMock(pid=7)
    target = home / "qwen.gguf"
    target.write_text("original")
    _agent(home, SMART_LABEL, str(target))

    res = json.loads(
        server.jev_train_lora(_dataset(home), target_gguf_path=str(target))
    )

    assert res["status"] == "TRAINING_STARTED", res.get("message")


@patch("subprocess.Popen")
def test_train_refuses_a_reasoning_off_target_it_cannot_build_rows_for(
    mock_popen, home, converter, monkeypatch
):
    _use_profile(monkeypatch, reasoning="off")
    target = home / "qwen.gguf"
    target.write_text("original")

    res = json.loads(
        server.jev_train_lora(_dataset(home), target_gguf_path=str(target))
    )

    assert res["status"] == "ERROR"
    assert "think" in res["message"]
    assert 'reasoning = "off"' in res["message"]
    mock_popen.assert_not_called()


@patch("subprocess.Popen")
def test_train_without_a_target_is_not_checked_against_the_profile(
    mock_popen, home, converter, monkeypatch
):
    _use_profile(monkeypatch, reasoning="off")
    mock_popen.return_value = MagicMock(pid=7)

    res = json.loads(server.jev_train_lora(_dataset(home), fuse=False))

    assert res["status"] in ("TRAINING_STARTED", "NOT_AVAILABLE")
    assert "refus" not in res.get("message", "")


@patch("subprocess.Popen")
def test_train_with_a_failing_resolver_keeps_todays_behaviour(
    mock_popen, home, converter, monkeypatch
):
    def _raise(*args, **kwargs):
        raise NotImplementedError

    monkeypatch.setattr(engine_profiles, "active_profile", _raise)
    mock_popen.return_value = MagicMock(pid=7)
    target = home / "anything.gguf"
    target.write_text("original")

    res = json.loads(
        server.jev_train_lora(_dataset(home), target_gguf_path=str(target))
    )

    assert res["status"] == "TRAINING_STARTED", res.get("message")


def test_restore_is_allowed_for_any_target_whatever_the_profile(home, monkeypatch):
    _use_profile(monkeypatch)
    target = home / "other.gguf"
    target.write_text("fine-tuned")
    (home / "other.gguf.bak").write_text("original")

    res = json.loads(
        server.jev_train_lora(action="restore", target_gguf_path=str(target))
    )

    assert res["status"] == "RESTORED"


# --- the launchd reader against a plist in the contract's shape ---


def _contract_plist(model_path, label=SMART_LABEL):
    """The shape section 5 of the contract gives the manager's plist."""
    arguments = [
        "/opt/homebrew/bin/llama-server",
        "-m",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        "8081",
        "--alias",
        "qwen2.5-7b",
        "--parallel",
        "2",
        "-c",
        "32768",
        "--log-verbosity",
        "3",
    ]
    return plistlib.dumps(
        {
            "Label": label,
            "ProgramArguments": arguments,
            "RunAtLoad": True,
            "KeepAlive": True,
            "StandardOutPath": "/tmp/jev.log",
            "StandardErrorPath": "/tmp/jev.log",
            "WorkingDirectory": "/tmp",
        }
    )


def _install_plist(home, payload, label=SMART_LABEL):
    agents = home / "Library" / "LaunchAgents"
    agents.mkdir(parents=True, exist_ok=True)
    (agents / f"{label}.plist").write_bytes(payload)


def test_the_label_reader_finds_the_agent_in_a_contract_shaped_plist(home):
    _install_plist(home, _contract_plist("/models/qwen.gguf"))

    assert lora_training.find_launchd_label("/models/qwen.gguf") == SMART_LABEL


def test_the_label_reader_ignores_a_contract_shaped_plist_for_another_model(home):
    _install_plist(home, _contract_plist("/models/kev.gguf"))

    assert lora_training.find_launchd_label("/models/qwen.gguf") is None


def test_the_label_reader_finds_the_agent_in_a_generated_plist(home):
    spec = _model("smart", "qwen.gguf")
    models_dir = home / "models"
    try:
        payload = engine_profiles.launchd_plist(
            spec,
            models_dir=str(models_dir),
            port=8081,
            label=SMART_LABEL,
            log_path=str(home / "jev.log"),
            llama_server="/opt/homebrew/bin/llama-server",
            workdir=str(home),
        )
    except NotImplementedError:
        pytest.skip("engine_profiles.launchd_plist is not implemented on this branch")

    _install_plist(home, payload)

    assert lora_training.find_launchd_label(spec.path(str(models_dir))) == SMART_LABEL
