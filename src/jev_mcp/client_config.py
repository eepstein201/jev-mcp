"""Registers this server in MCP client config files (`python -m jev_mcp.client_config`).

The config belongs to the user: every edit returns a new dict that differs only
in this server's entry, and a file that cannot be understood is never written.
"""

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from jev_mcp.router_config import write_json_atomic

SERVER_NAME = "jev-mcp"
OPENCODE_SCHEMA = "https://opencode.ai/config.json"
OPENCODE_CONFIG_PATH = "~/.config/opencode/opencode.json"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_PYTHON = os.path.join(REPO_ROOT, ".venv", "bin", "python")


def opencode_entry(python: str) -> dict[str, Any]:
    return {"type": "local", "command": [python, "-m", "jev_mcp.server"]}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} is not a JSON object")
    return value


def _without_server(section: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in section.items() if name != SERVER_NAME}


def link_opencode(config: Mapping[str, Any], python: str) -> dict[str, Any]:
    mcp = _mapping(config.get("mcp", {}), "mcp")
    servers = _mapping(mcp.get("servers", {}), "mcp.servers")
    linked = {
        **config,
        "mcp": {
            **_without_server(mcp),
            "servers": {**servers, SERVER_NAME: opencode_entry(python)},
        },
    }
    if "$schema" in linked:
        return linked
    return {"$schema": OPENCODE_SCHEMA, **linked}


def unlink_opencode(config: Mapping[str, Any]) -> dict[str, Any]:
    if "mcp" not in config:
        return dict(config)
    mcp = _mapping(config["mcp"], "mcp")
    unlinked = _without_server(mcp)
    if "servers" in mcp:
        unlinked["servers"] = _without_server(_mapping(mcp["servers"], "mcp.servers"))
    return {**config, "mcp": unlinked}


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m jev_mcp.client_config")
    parser.add_argument("action", choices=("link-opencode", "unlink-opencode"))
    parser.add_argument("--path", default=OPENCODE_CONFIG_PATH)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    args = parser.parse_args(argv)

    # Follow a symlink to the real file, so replacing it keeps the link in place.
    path = os.path.realpath(os.path.expanduser(args.path))
    if not os.path.exists(path):
        return 0
    try:
        with open(path, encoding="utf-8") as f:
            config = _mapping(json.load(f), "the top level")
        if args.action == "link-opencode":
            updated = link_opencode(config, args.python)
        else:
            updated = unlink_opencode(config)
    except (OSError, ValueError) as e:
        print(f"Left {path} untouched: {e}", file=sys.stderr)
        return 1
    if updated == config:
        return 0
    write_json_atomic(path, updated)
    print(f"Updated {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
