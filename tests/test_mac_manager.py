"""Runs the real jev_mac_manager.sh against a temp HOME with stand-in launchctl/pkill/sleep.

Nothing here touches real launchd agents or processes: the stand-ins only record their arguments.
"""
import os
import subprocess
from pathlib import Path

import pytest

MANAGER = Path(__file__).resolve().parent.parent / "jev_mac_manager.sh"
PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0"><dict><key>ProgramArguments</key><array>
    <string>/opt/homebrew/bin/llama-server</string>
    <string>--port</string>
    <string>{port}</string>
</array></dict></plist>
"""


@pytest.fixture
def sandbox(tmp_path):
    home = tmp_path / "home"
    agents = home / "Library" / "LaunchAgents"
    agents.mkdir(parents=True)
    shims = tmp_path / "shims"
    shims.mkdir()
    log = tmp_path / "calls.log"
    log.write_text("")
    for name in ("launchctl", "pkill", "sleep"):
        shim = shims / name
        shim.write_text(f'#!/bin/bash\necho "{name} $*" >> "{log}"\n')
        shim.chmod(0o755)
    env = {**os.environ, "HOME": str(home), "PATH": f"{shims}:{os.environ['PATH']}"}

    def run(*args):
        workdir = tmp_path / "work"
        workdir.mkdir(exist_ok=True)
        subprocess.run(
            ["bash", str(MANAGER), *args], env=env, cwd=workdir, capture_output=True, text=True, timeout=60, check=True
        )
        return log.read_text().splitlines()

    return agents, run


def test_stop_only_kills_processes_on_each_agents_own_port(sandbox):
    # Arrange
    agents, run = sandbox
    (agents / "com.jev.mlx_server_fast.plist").write_text(PLIST.format(port=8080))
    (agents / "com.jev.mlx_server_smart.plist").write_text(PLIST.format(port=8081))

    # Act
    calls = run("stop")

    # Assert
    kills = [c for c in calls if c.startswith("pkill")]
    assert any("llama-server" in k and "--port 8080" in k for k in kills)
    assert any("llama-server" in k and "--port 8081" in k for k in kills)
    assert "pkill -9 -f llama-server" not in kills
    assert "pkill -9 -f mlx_lm.server" not in kills


def test_stop_kills_nothing_when_no_agent_plist_exists(sandbox):
    _, run = sandbox

    calls = run("stop")

    assert [c for c in calls if c.startswith("pkill")] == []


def test_uninstall_unloads_every_agent_before_removing_its_plist(sandbox):
    # Arrange
    agents, run = sandbox
    fast = agents / "com.jev.mlx_server_fast.plist"
    smart = agents / "com.jev.mlx_server_smart.plist"
    fast.write_text(PLIST.format(port=8080))
    smart.write_text(PLIST.format(port=8081))

    # Act
    calls = run("uninstall", "--headless")

    # Assert
    bootouts = [c for c in calls if c.startswith("launchctl bootout")]
    assert any(str(fast) in b for b in bootouts)
    assert any(str(smart) in b for b in bootouts)
    assert not fast.exists() and not smart.exists()
