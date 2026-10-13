"""The `jev-mcp doctor` command.

Reports engine health, router-config validity, path security, webhook auth
setup, triage thresholds and versions. Hard rule: an auth variable's VALUE is
never placed in a report, a rendered line or the JSON output - only its NAME.
Printing is confined to main() so the module stays safe to import from server
code, where stdout is the JSON-RPC stream.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

from jev_mcp import router_config, security
from jev_mcp.email_triage.core import load_mail_config

HEALTH_TIMEOUT_SECONDS = 2

# engine name -> the port it is configured through, and its default.
ENGINES = (
    ("fast", "JEV_FAST_PORT", "8080"),
    ("smart", "JEV_SMART_PORT", "8081"),
)

# Auth variables reported by name only, in this order.
AUTH_VARIABLES = (
    "JEV_MCP_API_KEY",
    "JEV_MCP_JWT_SECRET",
    "JEV_MCP_JWKS_URL",
    "JEV_MCP_JWT_AUDIENCE",
)


def _port(raw: str, default: str) -> int:
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _engine_report(
    port_env: str, default_port: str, env: Mapping[str, str]
) -> dict[str, Any]:
    port = _port(env.get(port_env, default_port), default_port)
    try:
        request = urllib.request.Request(f"http://127.0.0.1:{port}/health")
        with urllib.request.urlopen(
            request, timeout=HEALTH_TIMEOUT_SECONDS
        ) as response:
            response.read()
        status = "up"
    except Exception:
        status = "down"
    return {"port": port, "status": status}


def _router_config_report() -> dict[str, Any]:
    path = router_config.ROUTER_CONFIG_PATH
    exists = os.path.exists(path)
    valid_json = False
    if exists:
        try:
            with open(path) as config_file:
                json.load(config_file)
            valid_json = True
        except (OSError, json.JSONDecodeError):
            valid_json = False
    return {"path": path, "exists": exists, "valid_json": valid_json}


def _versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for distribution in ("mcp", "torch"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def collect_report(env: Mapping[str, str] = os.environ) -> dict[str, Any]:
    """Everything `doctor` reports, as plain data. Reads no secret value:
    webhook auth is summarised as the NAMES of the variables that are set."""
    thresholds = load_mail_config()
    return {
        "engines": {
            name: _engine_report(port_env, default_port, env)
            for name, port_env, default_port in ENGINES
        },
        "router_config": _router_config_report(),
        "paths": security.allowed_roots_report(env),
        "webhook_auth": {"set": [name for name in AUTH_VARIABLES if env.get(name)]},
        "triage_thresholds": {
            "action_threshold": thresholds.action_threshold,
            "important_threshold": thresholds.important_threshold,
        },
        "versions": _versions(),
    }


def problems(report: Mapping[str, Any]) -> list[str]:
    """The report's actionable findings, in a stable order."""
    found: list[str] = []
    for name in ("fast", "smart"):
        engine = report["engines"][name]
        if engine["status"] != "up":
            found.append(
                f"{name} engine on port {engine['port']} is down - start the "
                "daemons with 'make start'"
            )
    config = report["router_config"]
    if config["exists"] and not config["valid_json"]:
        found.append(
            f"router config at {config['path']} is not valid JSON - fix or remove it"
        )
    auth_set = report["webhook_auth"]["set"]
    if "JEV_MCP_JWKS_URL" in auth_set and "JEV_MCP_JWT_AUDIENCE" not in auth_set:
        found.append(
            "JEV_MCP_JWKS_URL is set but JEV_MCP_JWT_AUDIENCE is not - the "
            "webhook cannot verify JWTs without an audience"
        )
    paths = report["paths"]
    if paths["working_directory_ignored"]:
        # Roots beyond the always-present defaults can only be explicit ones.
        defaults = set(security._default_roots())
        if all(root in defaults for root in paths["roots"]):
            found.append(
                "the server was started in its home directory, so the working "
                "directory is not an allowed root and JEV_ALLOWED_ROOTS lists "
                "no other - start it in a project directory or set "
                "JEV_ALLOWED_ROOTS"
            )
    return found


def render(report: Mapping[str, Any]) -> str:
    """The plain-text report. Auth variables appear by name only."""
    lines: list[str] = ["jev-mcp doctor", ""]
    lines.append("engines")
    for name in ("fast", "smart"):
        engine = report["engines"][name]
        lines.append(f"  {name} : {engine['status']} (port {engine['port']})")
    lines.append("")
    config = report["router_config"]
    lines.append("router config")
    lines.append(f"  path : {config['path']}")
    lines.append(f"  exists : {config['exists']}")
    lines.append(f"  valid json : {config['valid_json']}")
    lines.append("")
    paths = report["paths"]
    lines.append("paths")
    lines.append(f"  working directory : {paths['working_directory']}")
    lines.append(f"  working directory ignored : {paths['working_directory_ignored']}")
    for root in paths["roots"]:
        lines.append(f"  root : {root}")
    lines.append("")
    auth_set = report["webhook_auth"]["set"]
    lines.append("webhook auth (names only; values are never shown)")
    lines.append(f"  set : {', '.join(auth_set) if auth_set else 'none'}")
    lines.append("")
    thresholds = report["triage_thresholds"]
    lines.append("triage thresholds")
    lines.append(f"  action : {thresholds['action_threshold']}")
    lines.append(f"  important : {thresholds['important_threshold']}")
    lines.append("")
    lines.append("versions")
    for name, version in report["versions"].items():
        lines.append(f"  {name} : {version}")
    lines.append("")
    found = problems(report)
    lines.append("problems")
    if found:
        lines.extend(f"  - {problem}" for problem in found)
    else:
        lines.append("  none")
    return "\n".join(lines)


def main(argv: Sequence[str]) -> int:
    """The CLI entry point; the only function in this module that prints."""
    parser = argparse.ArgumentParser(
        prog="jev-mcp doctor",
        description=(
            "Report engine health, router-config validity, path security, "
            "webhook auth setup, triage thresholds and versions."
        ),
    )
    parser.add_argument(
        "--json", action="store_true", help="print the report as JSON instead of text"
    )
    args = parser.parse_args(argv)

    report = collect_report()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render(report))
    return 1 if problems(report) else 0
