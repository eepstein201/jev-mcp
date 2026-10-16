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
import math
import os
import platform
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

from jev_mcp import daemon_provider, engine_profiles, router_config, security
from jev_mcp.email_triage.core import load_mail_config
from jev_mcp.kev_provider import KevProvider
from jev_mcp.provider import NoulQuestion, post_json

HEALTH_TIMEOUT_SECONDS = 2
# llama-server answers 503 on /health and /props while a model loads.
LOADING_HTTP_STATUS = 503
PROBE_TIMEOUT_SECONDS = 10
# A probe that times out is attempted this many more times (a model paged in from
# swap, or an engine busy with other requests, can miss one window); other failures are not.
PROBE_TIMEOUT_RETRIES = 1
# The probe's reply must put at least this much probability on its two targets.
PROBE_MIN_TARGET_MASS = 0.9
PROBE_TARGETS = ("true", "false")
PROBE_PROMPT = (
    "You are a strict classification model. Evaluate the question based only on the "
    "context.\nContext: Water boils at 100 degrees Celsius at sea level.\n"
    "Question: Does water boil at 100 degrees Celsius at sea level?\n"
    "Answer True or False.\nAnswer:"
)

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


def _health_status(port: int) -> str:
    """ "up", "loading" (a 503: the model is still loading) or "down"."""
    try:
        request = urllib.request.Request(f"http://127.0.0.1:{port}/health")
        with urllib.request.urlopen(
            request, timeout=HEALTH_TIMEOUT_SECONDS
        ) as response:
            response.read()
        return "up"
    except urllib.error.HTTPError as error:
        return "loading" if error.code == LOADING_HTTP_STATUS else "down"
    except Exception:
        return "down"


# The fast engine scores through its SystemOne endpoint, not through chat completions
# (a chat request to the Kev daemon returns near-uniform tokens with no True/False).
FAST_PROBE_KEY = "probe"
FAST_PROBE_STATE = "Water boils at 100 degrees Celsius at sea level."
FAST_PROBE_QUESTION = "Does water boil at 100 degrees Celsius at sea level?"
# A fast answer's probabilities must add up to one, within this slack.
FAST_PROBE_SUM_TOLERANCE = 0.05


_T = TypeVar("_T")


def _is_timeout(error: BaseException | None) -> bool:
    if isinstance(error, urllib.error.URLError):
        error = error.reason if isinstance(error.reason, BaseException) else error
    return isinstance(error, TimeoutError)


def _retry_on_timeout(attempt: Callable[[], _T]) -> _T:
    """Run attempt, retrying PROBE_TIMEOUT_RETRIES times if it raises a timeout; any
    other error, and a timeout on the last attempt, propagate."""
    for _ in range(PROBE_TIMEOUT_RETRIES):
        try:
            return attempt()
        except Exception as e:
            if not _is_timeout(e):
                raise
    return attempt()


def _probe_fast(port: int) -> dict[str, Any]:
    """Score one NoulQuestion through KevProvider, the way the server scores the fast tier.

    Passes when an answer comes back whose probabilities sum to about 1. Never raises."""
    provider = KevProvider(base_url=f"http://127.0.0.1:{port}/v1/systemone")

    def score() -> dict[str, Any]:
        answers = provider.evaluate_batch(
            FAST_PROBE_STATE,
            [NoulQuestion(key=FAST_PROBE_KEY, prompt=FAST_PROBE_QUESTION)],
        )
        # KevProvider swallows its transport errors; surface a timeout so it is retried.
        if not answers and _is_timeout(provider.last_error):
            raise provider.last_error  # type: ignore[misc]
        return answers

    try:
        try:
            answers = _retry_on_timeout(score)
        except Exception as e:
            if not _is_timeout(e):
                raise
            answers = {}  # a second timeout reads as "no answer", as before
        probabilities = answers.get(FAST_PROBE_KEY, {}).get("probabilities")
        total = sum(probabilities.values()) if probabilities else None
    except Exception as e:
        return {
            "ok": False,
            "first_token": None,
            "target_mass": None,
            "detail": f"no usable answer ({type(e).__name__}: {e})",
        }
    if total is None:
        return {
            "ok": False,
            "first_token": None,
            "target_mass": None,
            "detail": "the SystemOne endpoint returned no answer",
        }
    if abs(total - 1.0) > FAST_PROBE_SUM_TOLERANCE:
        return {
            "ok": False,
            "first_token": None,
            "target_mass": total,
            "detail": f"the answer's probabilities sum to {total:.2f}, not 1",
        }
    return {"ok": True, "first_token": None, "target_mass": total, "detail": "ok"}


def _probe(port: int) -> dict[str, Any]:
    """Ask the engine one True/False question and check it answers with a target.

    The first token must be a target and the targets must hold most of the
    probability; a model that starts with something else (a think tag, prose) cannot
    be scored. Never raises."""
    payload = {
        "messages": [{"role": "user", "content": PROBE_PROMPT}],
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": 11,
        "temperature": 0.0,
    }
    try:
        res = _retry_on_timeout(
            lambda: post_json(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                payload,
                timeout=PROBE_TIMEOUT_SECONDS,
            )
        )
        first = res["choices"][0]["logprobs"]["content"][0]
        token = str(first.get("token", "")).strip().lower()
        mass = sum(
            math.exp(item["logprob"])
            for item in first.get("top_logprobs", [])
            if str(item.get("token", "")).strip().lower() in PROBE_TARGETS
        )
    except Exception as e:
        return {
            "ok": False,
            "first_token": None,
            "target_mass": None,
            "detail": f"no usable reply ({type(e).__name__}: {e})",
        }
    if token not in PROBE_TARGETS:
        return {
            "ok": False,
            "first_token": token,
            "target_mass": mass,
            "detail": f"the first token was {token!r}, not True or False",
        }
    if mass <= PROBE_MIN_TARGET_MASS:
        return {
            "ok": False,
            "first_token": token,
            "target_mass": mass,
            "detail": (
                f"True/False hold only {mass:.2f} of the probability "
                f"(need more than {PROBE_MIN_TARGET_MASS})"
            ),
        }
    return {"ok": True, "first_token": token, "target_mass": mass, "detail": "ok"}


def _engine_report(
    name: str,
    port_env: str,
    default_port: str,
    env: Mapping[str, str],
    profile: Any,
) -> dict[str, Any]:
    port = _port(env.get(port_env, default_port), default_port)
    fast_is_kev = name == "fast" and env.get("JEV_FAST_ENGINE", "kev") == "kev"
    status = _health_status(port)
    served = None
    probe = None
    if status == "up":
        served_path = daemon_provider.fetch_served_model_path(
            f"http://127.0.0.1:{port}"
        )
        served = os.path.basename(served_path) if served_path else None
        probe = _probe_fast(port) if fast_is_kev else _probe(port)
    expected = None
    if profile is not None:
        expected = (profile.fast if name == "fast" else profile.smart).filename
    return {
        "port": port,
        "status": status,
        "served_model": served,
        "expected_model": expected,
        "probe": probe,
    }


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


def _active_profile() -> tuple[Any, str | None]:
    """(profile, error): the active profile, read through the resolver's strict API so a
    broken marker or table is reported (a ProfileError) instead of looking like "none".

    Any other failure (the resolver not usable at all) is "no profile" and no error."""
    try:
        name = engine_profiles.read_active_name()
        return engine_profiles.load_profiles().get(name), None
    except engine_profiles.ProfileError as e:
        return None, str(e)
    except Exception as e:
        daemon_provider.logger.info(f"No active engine profile could be read: {e!r}")
        return None, None


def collect_report(env: Mapping[str, str] = os.environ) -> dict[str, Any]:
    """Everything `doctor` reports, as plain data. Reads no secret value:
    webhook auth is summarised as the NAMES of the variables that are set."""
    thresholds = load_mail_config()
    profile, profile_error = _active_profile()
    # The engines are checked at the same time, so a wedged one cannot delay the other.
    with ThreadPoolExecutor(max_workers=len(ENGINES)) as pool:
        futures = {
            name: pool.submit(
                _engine_report, name, port_env, default_port, env, profile
            )
            for name, port_env, default_port in ENGINES
        }
        engines = {name: future.result() for name, future in futures.items()}
    return {
        "profile": profile.name if profile is not None else None,
        "profile_error": profile_error,
        "engines": engines,
        "router_config": _router_config_report(),
        "paths": security.allowed_roots_report(env),
        "webhook_auth": {"set": [name for name in AUTH_VARIABLES if env.get(name)]},
        "triage_thresholds": {
            "action_threshold": thresholds.action_threshold,
            "important_threshold": thresholds.important_threshold,
        },
        "versions": _versions(),
    }


def _engine_problems(name: str, report: Mapping[str, Any]) -> list[str]:
    """What is wrong with one engine: down, still loading, the wrong model or a bad probe."""
    engine = report["engines"][name]
    port = engine["port"]
    if engine["status"] == "loading":
        return [
            f"{name} engine on port {port} is still loading its model - wait "
            "for it to finish, then run doctor again"
        ]
    if engine["status"] != "up":
        return [
            f"{name} engine on port {port} is down - start the daemons with "
            "'make start'"
        ]
    found: list[str] = []
    expected = engine.get("expected_model")
    served = engine.get("served_model")
    profile = report.get("profile")
    if expected and served is None:
        found.append(
            f"{name} engine on port {port} did not name its model in /props; "
            f"profile {profile} expects {expected}"
        )
    elif expected and served != expected:
        found.append(
            f"{name} engine on port {port} serves {served} but profile {profile} "
            f"expects {expected} - run 'make start PROFILE={profile}'"
        )
    probe = engine.get("probe")
    if probe is not None and not probe["ok"]:
        found.append(
            f"{name} engine on port {port} failed its scoring probe: {probe['detail']}"
        )
    return found


def problems(report: Mapping[str, Any]) -> list[str]:
    """The report's actionable findings, in a stable order."""
    found: list[str] = []
    if report.get("profile_error"):
        found.append(
            f"the active engine profile cannot be read: {report['profile_error']}"
        )
    for name in ("fast", "smart"):
        found.extend(_engine_problems(name, report))
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


def _engine_detail_lines(engine: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    if engine.get("served_model") is not None:
        lines.append(f"    served : {engine['served_model']}")
    if engine.get("expected_model") is not None:
        lines.append(f"    expected : {engine['expected_model']}")
    probe = engine.get("probe")
    if probe is not None:
        lines.append(f"    probe : {'ok' if probe['ok'] else probe['detail']}")
    return lines


def render(report: Mapping[str, Any]) -> str:
    """The plain-text report. Auth variables appear by name only."""
    lines: list[str] = ["jev-mcp doctor", ""]
    lines.append("engines")
    lines.append(f"  profile : {report.get('profile') or 'none'}")
    if report.get("profile_error"):
        lines.append(f"  profile error : {report['profile_error']}")
    for name in ("fast", "smart"):
        engine = report["engines"][name]
        lines.append(f"  {name} : {engine['status']} (port {engine['port']})")
        lines.extend(_engine_detail_lines(engine))
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
