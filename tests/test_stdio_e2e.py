"""End-to-end stdio test: spawns the real server and speaks MCP JSON-RPC.

Unlike the unit suite (whose conftest stubs the MCP SDK), this exercises the
real transport in a fresh interpreter: the initialize handshake, the
tools/list registry, and prompts/list. The subprocess is unaffected by the
parent process's conftest stubs.
"""
import json
import os
import queue
import subprocess
import sys
import threading
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXPECTED_TOOLS = {
    "evaluate", "optimize-prompt", "calibrate", "explain-decision",
    "generate-data", "manage-router", "model-router", "handoff", "train",
    "temperature", "read-file", "compact", "scan-repo",
}


def _start_server():
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen(
        [sys.executable, "-m", "jev_mcp.server"],
        cwd=REPO_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


def _reader(pipe, q):
    for line in iter(pipe.readline, b""):
        q.put(line)
    q.put(None)  # EOF sentinel


def test_stdio_handshake_and_tool_registry():
    proc = _start_server()
    q = queue.Queue()
    threading.Thread(target=_reader, args=(proc.stdout, q), daemon=True).start()

    def send(obj):
        proc.stdin.write((json.dumps(obj) + "\n").encode())
        proc.stdin.flush()

    def read_rpc(timeout=30.0):
        """Return the next JSON-RPC response (messages with an id).

        Skips notifications and any stray non-JSON stdout chatter; fails with
        the server's stderr if the process dies or the pipe closes.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                line = q.get(timeout=1.0)
            except queue.Empty:
                if proc.poll() is not None:
                    stderr = proc.stderr.read().decode(errors="replace")
                    raise AssertionError(
                        f"server exited early (rc={proc.returncode}):\n{stderr}"
                    )
                continue
            if line is None:
                stderr = proc.stderr.read().decode(errors="replace")
                raise AssertionError(f"server stdout closed:\n{stderr}")
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in msg:
                return msg
        raise AssertionError("timed out waiting for a JSON-RPC response")

    try:
        send({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "stdio-e2e", "version": "0.0.1"},
            },
        })
        init = read_rpc(timeout=30)
        assert init.get("id") == 1, init
        result = init.get("result") or {}
        assert result.get("serverInfo", {}).get("name") == "jev-mcp", init

        send({"jsonrpc": "2.0", "method": "notifications/initialized"})

        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        tools = read_rpc()
        names = {t["name"] for t in tools["result"]["tools"]}
        assert names == EXPECTED_TOOLS, (
            f"missing={EXPECTED_TOOLS - names} extra={names - EXPECTED_TOOLS}"
        )

        send({"jsonrpc": "2.0", "id": 3, "method": "prompts/list", "params": {}})
        prompts = read_rpc()
        prompt_names = {p["name"] for p in prompts["result"]["prompts"]}
        assert {"calibrate", "temperature", "scan-repo"} <= prompt_names
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
