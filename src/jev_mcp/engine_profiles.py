"""Engine profiles: which fast and smart models the daemons serve (`python -m jev_mcp.engine_profiles`).

Interface frozen by docs/superpowers/specs/2026-10-14-engine-profiles-cli-contract.md;
lane P1-a implements the bodies without changing these signatures.

The profile table is validated against an allow-list: every name, repo, revision, digest and
filename is matched with `fullmatch`, and unknown keys are rejected at every level. Stdlib only.
"""

import argparse
import contextlib
import http.client
import json
import logging
import math
import os
import plistlib
import re
import sys
import tempfile
import time
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, NoReturn

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_PROFILE = "default"
DEFAULT_GATE = 0.85
DEFAULT_CONFIG_PATH = os.path.join(REPO_ROOT, "engine_profiles.toml")
DEFAULT_MODELS_DIR = os.path.join(REPO_ROOT, "models")
ACTIVE_PROFILE_PATH = os.path.expanduser("~/.jev/active_profile")

_NAME_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,31}")
_REPO_RE = re.compile(r"[\w.-]+/[\w.-]+", re.ASCII)
_FILENAME_RE = re.compile(r"[\w.-]+\.gguf", re.ASCII)
_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_LABEL_RE = re.compile(r"[A-Za-z0-9._-]+")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")  # C0, DEL and C1

_TIERS = ("fast", "smart")
_REASONING = ("auto", "off", "on")
_STYLES = ("plain", "sandbox")
_MODEL_KEYS = (
    "tier",
    "repo",
    "revision",
    "filename",
    "sha256",
    "parallel",
    "ctx_per_slot",
    "reasoning",
    "style",
)
_PROFILE_KEYS = ("fast", "smart")
_PROFILE_OPTIONAL_KEYS = ("gate", "eval_only")
_PARALLEL_RANGE = (1, 16)
_CTX_PER_SLOT_RANGE = (512, 131072)
_MAX_PORT = 65535
_MAX_MARKER_BYTES = 256
_MAX_PROPS_BYTES = 1 << 20
_MAX_SLEEP_IDLE_SECONDS = 7 * 24 * 3600
_LOG_VERBOSITY = (
    "3"  # puts the model, KV and compute buffer lines in the daemon startup log
)
_LOOPBACK = "127.0.0.1"
_DEFAULT_ACTIVATE_TIMEOUT_S = 120.0
_PROPS_REQUEST_TIMEOUT_S = 5
_POLL_INTERVAL_S = 1
_EXIT_RUNTIME = 1
_EXIT_INVALID = 2

_sleep: Callable[[float], None] = time.sleep
_monotonic: Callable[[], float] = time.monotonic


class ProfileError(ValueError):
    """The profile table, a profile name or the active marker is invalid."""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    tier: Literal["fast", "smart"]
    repo: str
    revision: str
    filename: str
    sha256: str
    parallel: int
    ctx_per_slot: int
    reasoning: Literal["auto", "off", "on"]
    style: Literal["plain", "sandbox"]

    def path(self, models_dir: str) -> str:
        """`<models_dir>/<repo name>/<filename>`, which must stay inside `realpath(models_dir)`."""
        _, slash, repo_name = self.repo.partition("/")
        if not slash or not repo_name:
            raise ProfileError(
                f"models.{self.name}.repo: expected 'owner/name', got {_show(self.repo)}"
            )
        _check_no_control(models_dir, "models dir")
        root = os.path.realpath(models_dir)
        candidate = os.path.join(os.path.abspath(models_dir), repo_name, self.filename)
        resolved = os.path.realpath(candidate)
        if resolved == root or os.path.commonpath([root, resolved]) != root:
            raise ProfileError(
                f"models.{self.name}: model path resolves outside the models directory"
            )
        return candidate

    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{self.filename}"


@dataclass(frozen=True)
class Profile:
    name: str
    fast: ModelSpec
    smart: ModelSpec
    gate: float
    eval_only: bool


@dataclass(frozen=True)
class ProfileTable:
    models: Mapping[str, ModelSpec]
    profiles: Mapping[str, Profile]

    def get(self, name: str) -> Profile:
        _check_name(name, "profile name")
        profile = self.profiles.get(name)
        if profile is None:
            raise ProfileError(f"unknown profile {name!r}")
        return profile


# --- validation helpers ------------------------------------------------------------------


def _show(value: object) -> str:
    return repr(value)[:60]


def _join(where: str, key: str) -> str:
    return f"{where}.{key}" if where else key


def _check_name(value: object, where: str) -> str:
    if not isinstance(value, str) or not _NAME_RE.fullmatch(value):
        raise ProfileError(f"{where}: invalid name {_show(value)}")
    return value


def _check_match(value: object, regex: re.Pattern[str], where: str) -> str:
    if not isinstance(value, str) or not regex.fullmatch(value):
        raise ProfileError(f"{where}: invalid value {_show(value)}")
    return value


def _check_enum(value: object, allowed: Sequence[str], where: str) -> Any:
    if not isinstance(value, str) or value not in allowed:
        raise ProfileError(
            f"{where}: expected one of {', '.join(allowed)}, got {_show(value)}"
        )
    return value


def _check_int(value: object, bounds: tuple[int, int], where: str) -> int:
    low, high = bounds
    if type(value) is not int or not low <= value <= high:
        raise ProfileError(
            f"{where}: expected an integer from {low} to {high}, got {_show(value)}"
        )
    return value


def _check_table(value: object, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ProfileError(f"{where}: expected a table")
    return value


def _check_keys(
    table: Mapping[str, Any],
    where: str,
    required: Sequence[str],
    optional: Sequence[str] = (),
) -> None:
    unknown = sorted(set(table) - set(required) - set(optional))
    if unknown:
        raise ProfileError(f"{where or 'top level'}: unknown key {_show(unknown[0])}")
    for key in required:
        if key not in table:
            raise ProfileError(f"{_join(where, key)}: missing required key")


def _check_no_control(value: str, where: str) -> str:
    if _CONTROL_RE.search(value):
        raise ProfileError(f"{where}: contains a control character")
    return value


def _check_repo(value: object, where: str) -> str:
    repo = _check_match(value, _REPO_RE, where)
    if any(set(part) == {"."} for part in repo.split("/")):
        raise ProfileError(f"{where}: invalid value {_show(repo)}")
    return repo


def _validate_model(model: ModelSpec) -> ModelSpec:
    """Check every field of a spec (parsed or hand-built); returns it unchanged."""
    where = f"models.{_check_name(model.name, 'models')}"
    _check_enum(model.tier, _TIERS, f"{where}.tier")
    _check_repo(model.repo, f"{where}.repo")
    _check_match(model.revision, _REVISION_RE, f"{where}.revision")
    _check_match(model.filename, _FILENAME_RE, f"{where}.filename")
    _check_match(model.sha256, _SHA256_RE, f"{where}.sha256")
    _check_int(model.parallel, _PARALLEL_RANGE, f"{where}.parallel")
    _check_int(model.ctx_per_slot, _CTX_PER_SLOT_RANGE, f"{where}.ctx_per_slot")
    _check_enum(model.reasoning, _REASONING, f"{where}.reasoning")
    _check_enum(model.style, _STYLES, f"{where}.style")
    return model


def _parse_model(name: object, raw: object) -> ModelSpec:
    model_name = _check_name(name, "models")
    where = f"models.{model_name}"
    table = _check_table(raw, where)
    _check_keys(table, where, _MODEL_KEYS)
    return _validate_model(
        ModelSpec(name=model_name, **{key: table[key] for key in _MODEL_KEYS})
    )


def _model_for_tier(
    models: Mapping[str, ModelSpec], table: Mapping[str, Any], tier: str, where: str
) -> ModelSpec:
    name = _check_name(table[tier], f"{where}.{tier}")
    model = models.get(name)
    if model is None:
        raise ProfileError(f"{where}.{tier}: unknown model {name!r}")
    if model.tier != tier:
        raise ProfileError(
            f"{where}.{tier}: model {name!r} has tier {model.tier!r}, expected {tier!r}"
        )
    return model


def _parse_profile(
    name: object, raw: object, models: Mapping[str, ModelSpec]
) -> Profile:
    profile_name = _check_name(name, "profiles")
    where = f"profiles.{profile_name}"
    table = _check_table(raw, where)
    _check_keys(table, where, _PROFILE_KEYS, _PROFILE_OPTIONAL_KEYS)
    gate = table.get("gate", DEFAULT_GATE)
    if type(gate) is not float or not 0 < gate < 1:
        raise ProfileError(
            f"{where}.gate: expected a float between 0 and 1 (exclusive), got {_show(gate)}"
        )
    eval_only = table.get("eval_only", False)
    if type(eval_only) is not bool:
        raise ProfileError(
            f"{where}.eval_only: expected true or false, got {_show(eval_only)}"
        )
    return Profile(
        name=profile_name,
        fast=_model_for_tier(models, table, "fast", where),
        smart=_model_for_tier(models, table, "smart", where),
        gate=gate,
        eval_only=eval_only,
    )


def load_profiles(path: str = DEFAULT_CONFIG_PATH) -> ProfileTable:
    try:
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
    except OSError as exc:
        raise ProfileError(f"cannot read {path!r}: {exc.strerror or exc}") from exc
    except (
        ValueError,
        RecursionError,
    ) as exc:  # TOMLDecodeError, bad UTF-8, absurd nesting
        raise ProfileError(f"invalid TOML in {path!r}: {_oneline(exc)}") from exc
    _check_keys(data, "", ("models", "profiles"))
    raw_models = _check_table(data["models"], "models")
    models = {name: _parse_model(name, raw) for name, raw in raw_models.items()}
    raw_profiles = _check_table(data["profiles"], "profiles")
    profiles = {
        name: _parse_profile(name, raw, models) for name, raw in raw_profiles.items()
    }
    return ProfileTable(MappingProxyType(models), MappingProxyType(profiles))


# --- active marker -----------------------------------------------------------------------


def read_active_name(path: str = ACTIVE_PROFILE_PATH) -> str:
    try:
        with open(path, encoding="utf-8") as handle:
            content = handle.read(_MAX_MARKER_BYTES)
    except FileNotFoundError:
        return DEFAULT_PROFILE
    except (OSError, UnicodeDecodeError) as exc:
        raise ProfileError(
            f"cannot read active profile marker {path!r}: {exc}"
        ) from exc
    return _check_name(content.strip(), "active profile marker")


def active_profile(
    config_path: str = DEFAULT_CONFIG_PATH, marker_path: str = ACTIVE_PROFILE_PATH
) -> Profile | None:
    """The marked profile, or None (logged) on any problem: the server never fails over a marker."""
    try:
        return load_profiles(config_path).get(read_active_name(marker_path))
    except (
        Exception
    ) as exc:  # the contract: never raises, whatever the table or marker holds
        logger.warning("no active engine profile: %s", exc)
        return None


# --- launchd plist -----------------------------------------------------------------------


def _check_port(value: object, where: str) -> int:
    return _check_int(value, (1, _MAX_PORT), where)


def _check_abs_path(value: object, where: str) -> str:
    if not isinstance(value, str) or not os.path.isabs(value):
        raise ProfileError(f"{where}: expected an absolute path, got {_show(value)}")
    return _check_no_control(value, where)


def launchd_plist(
    model: ModelSpec,
    *,
    models_dir: str,
    port: int,
    label: str,
    log_path: str,
    llama_server: str,
    workdir: str,
    sleep_idle_seconds: int | None = None,
) -> bytes:
    _validate_model(model)
    _check_port(port, "port")
    _check_match(label, _LABEL_RE, "label")
    log_path = _check_abs_path(log_path, "log path")
    llama_server = _check_abs_path(llama_server, "llama-server path")
    workdir = _check_abs_path(workdir, "workdir")
    model_path = _check_no_control(model.path(models_dir), "model path")
    arguments = [
        llama_server,
        "-m",
        model_path,
        "--host",
        _LOOPBACK,
        "--port",
        str(port),
        "--alias",
        model.name,
        "--parallel",
        str(model.parallel),
        "-c",
        str(model.parallel * model.ctx_per_slot),
        "--log-verbosity",
        _LOG_VERBOSITY,
    ]
    if model.reasoning != "auto":
        arguments += ["--reasoning", model.reasoning]
    if sleep_idle_seconds is not None:
        arguments += [
            "--sleep-idle-seconds",
            str(
                _check_int(
                    sleep_idle_seconds,
                    (1, _MAX_SLEEP_IDLE_SECONDS),
                    "sleep-idle-seconds",
                )
            ),
        ]
    document = {
        "Label": label,
        "ProgramArguments": arguments,
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": log_path,
        "StandardErrorPath": log_path,
        "WorkingDirectory": workdir,
    }
    return plistlib.dumps(document, fmt=plistlib.FMT_XML, sort_keys=False)


# --- CLI ---------------------------------------------------------------------------------


class _UsageError(Exception):
    """argparse rejected the command line."""


class _RuntimeFailure(Exception):
    """A runtime failure (exit 1); each entry is one stderr line."""

    def __init__(self, lines: Sequence[str]):
        super().__init__("; ".join(lines))
        self.lines = list(lines)


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _UsageError(message)


def _port_arg(text: str) -> int:
    try:
        return _check_port(int(text), "port")
    except (ValueError, ProfileError) as exc:
        raise argparse.ArgumentTypeError(f"invalid port {text!r}") from exc


def _seconds_arg(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid number {text!r}") from exc
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(f"invalid number {text!r}")
    return value


def _positive_int_arg(text: str) -> int:
    try:
        return int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer {text!r}") from exc


def _oneline(value: object) -> str:
    return _CONTROL_RE.sub("?", str(value))


def _select_profile(args: argparse.Namespace, name: str) -> tuple[Profile, str]:
    config = DEFAULT_CONFIG_PATH if args.config is None else args.config
    if args.models_dir == "":
        raise ProfileError("--models-dir: must not be empty")
    models_dir = os.path.abspath(
        DEFAULT_MODELS_DIR if args.models_dir is None else args.models_dir
    )
    profile = load_profiles(config).get(name)
    if profile.eval_only and not args.allow_eval_only:
        raise ProfileError(f"profile {name!r} is eval-only; pass --allow-eval-only")
    return profile, models_dir


def _cmd_resolve(args: argparse.Namespace) -> list[str]:
    name = (
        args.profile
        if args.profile is not None
        else read_active_name(ACTIVE_PROFILE_PATH)
    )
    profile, models_dir = _select_profile(args, name)
    fields: list[tuple[str, str]] = [("profile", profile.name)]
    for tier, spec in (("fast", profile.fast), ("smart", profile.smart)):
        fields += [
            (f"{tier}_model", spec.name),
            (f"{tier}_path", spec.path(models_dir)),
            (f"{tier}_url", spec.url()),
            (f"{tier}_sha256", spec.sha256),
        ]
    return [f"{key}={_check_no_control(value, key)}" for key, value in fields]


def _atomic_write(path: str, data: bytes) -> None:
    path = os.path.abspath(path)
    handle, temp_path = tempfile.mkstemp(
        dir=os.path.dirname(path), prefix=".tmp-", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
        os.chmod(temp_path, 0o644)
        os.replace(temp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temp_path)
        raise


def _cmd_plist(args: argparse.Namespace) -> list[str]:
    profile, models_dir = _select_profile(args, args.profile)
    out = _check_abs_path(args.out, "--out")
    document = launchd_plist(
        profile.fast if args.tier == "fast" else profile.smart,
        models_dir=models_dir,
        port=args.port,
        label=args.label,
        log_path=args.log,
        llama_server=args.llama_server,
        workdir=args.workdir,
        sleep_idle_seconds=args.sleep_idle_seconds,
    )
    _atomic_write(out, document)
    return []


def _parse_props(tier: str, port: int, body: bytes) -> dict[str, Any]:
    try:
        props = json.loads(body)
    except ValueError as exc:
        raise _RuntimeFailure(
            [f"{tier} /props on port {port}: response is not JSON"]
        ) from exc
    if not isinstance(props, dict):
        raise _RuntimeFailure(
            [f"{tier} /props on port {port}: response is not a JSON object"]
        )
    return props


def _poll_props(tier: str, port: int, timeout: float) -> dict[str, Any]:
    """GET /props once a second until 200; a 503 means the model is still loading."""
    url = f"http://{_LOOPBACK}:{port}/props"
    deadline = _monotonic() + timeout
    while True:
        try:
            with urllib.request.urlopen(
                url, timeout=_PROPS_REQUEST_TIMEOUT_S
            ) as response:
                if response.status != 200:
                    reason = f"HTTP {response.status}"
                else:
                    body = response.read(_MAX_PROPS_BYTES + 1)
                    if len(body) <= _MAX_PROPS_BYTES:
                        return _parse_props(tier, port, body)
                    reason = f"response larger than {_MAX_PROPS_BYTES} bytes"
        except urllib.error.HTTPError as exc:
            if exc.code != 503:
                raise _RuntimeFailure(
                    [f"{tier} /props on port {port}: HTTP {exc.code}"]
                ) from exc
            reason = "model still loading (HTTP 503)"
        except (OSError, http.client.HTTPException) as exc:
            reason = f"unreachable ({_oneline(exc)})"
        if _monotonic() >= deadline:
            raise _RuntimeFailure(
                [
                    f"{tier} /props on port {port}: not ready after {timeout:g}s: {reason}"
                ]
            )
        _sleep(_POLL_INTERVAL_S)


def _served_n_ctx(props: Mapping[str, Any]) -> object:
    settings = props.get("default_generation_settings")
    if isinstance(settings, dict) and "n_ctx" in settings:
        return settings["n_ctx"]
    return props.get("n_ctx")


def _mismatches(
    tier: str, spec: ModelSpec, expected_path: str, props: Mapping[str, Any]
) -> list[str]:
    served_path = props.get("model_path")
    served_real = (
        os.path.realpath(served_path)
        if isinstance(served_path, str) and served_path
        else None
    )
    checks = [
        ("model_path", served_real == expected_path, expected_path, served_path),
        (
            "total_slots",
            _same_int(props.get("total_slots"), spec.parallel),
            spec.parallel,
            props.get("total_slots"),
        ),
        (
            "n_ctx",
            _same_int(_served_n_ctx(props), spec.ctx_per_slot),
            spec.ctx_per_slot,
            _served_n_ctx(props),
        ),
    ]
    return [
        f"{tier} {field}: expected {expected}, served {_oneline(served)}"
        for field, ok, expected, served in checks
        if not ok
    ]


def _same_int(served: object, expected: int) -> bool:
    return type(served) is int and served == expected


def _check_daemon(
    tier: str, spec: ModelSpec, expected_path: str, port: int, timeout: float
) -> list[str]:
    try:
        props = _poll_props(tier, port, timeout)
    except _RuntimeFailure as failure:
        return failure.lines
    return _mismatches(tier, spec, expected_path, props)


def _cmd_activate(args: argparse.Namespace) -> list[str]:
    profile, models_dir = _select_profile(args, args.profile)
    marker = ACTIVE_PROFILE_PATH if args.marker is None else args.marker
    daemons = (
        ("fast", profile.fast, args.fast_port),
        ("smart", profile.smart, args.smart_port),
    )
    expected = {
        tier: os.path.realpath(spec.path(models_dir)) for tier, spec, _ in daemons
    }
    failures: list[str] = []
    for tier, spec, port in daemons:
        failures += _check_daemon(tier, spec, expected[tier], port, args.timeout)
    if failures:
        raise _RuntimeFailure(failures)
    os.makedirs(os.path.dirname(os.path.abspath(marker)), exist_ok=True)
    _atomic_write(marker, f"{profile.name}\n".encode())
    return [f"active profile: {profile.name}"]


def _build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="python -m jev_mcp.engine_profiles",
        description="Resolve and activate engine profiles.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def command(
        name: str,
        handler: Callable[[argparse.Namespace], list[str]],
        profile_optional: bool = False,
    ) -> argparse.ArgumentParser:
        sub = commands.add_parser(name)
        sub.add_argument(
            "profile", nargs="?" if profile_optional else None, default=None
        )
        sub.add_argument("--config", default=None)
        sub.add_argument("--models-dir", default=None)
        sub.add_argument("--allow-eval-only", action="store_true")
        sub.set_defaults(handler=handler)
        return sub

    command("resolve", _cmd_resolve, profile_optional=True)
    plist = command("plist", _cmd_plist)
    plist.add_argument("--tier", required=True, choices=_TIERS)
    plist.add_argument("--port", required=True, type=_port_arg)
    for flag in ("--label", "--log", "--llama-server", "--workdir", "--out"):
        plist.add_argument(flag, required=True)
    plist.add_argument("--sleep-idle-seconds", type=_positive_int_arg, default=None)
    activate = command("activate", _cmd_activate)
    activate.add_argument("--fast-port", required=True, type=_port_arg)
    activate.add_argument("--smart-port", required=True, type=_port_arg)
    activate.add_argument(
        "--timeout", type=_seconds_arg, default=_DEFAULT_ACTIVATE_TIMEOUT_S
    )
    activate.add_argument("--marker", default=None)
    return parser


def main(argv: Sequence[str]) -> int:
    try:
        args = _build_parser().parse_args(list(argv))
        lines = args.handler(args)
    except SystemExit as exit_request:  # --help
        return exit_request.code if isinstance(exit_request.code, int) else 0
    except (_UsageError, ProfileError) as exc:
        print(f"error: {_oneline(exc)}", file=sys.stderr)
        return _EXIT_INVALID
    except _RuntimeFailure as failure:
        for line in failure.lines:
            print(_oneline(line), file=sys.stderr)
        return _EXIT_RUNTIME
    except OSError as exc:
        print(f"error: {_oneline(exc)}", file=sys.stderr)
        return _EXIT_RUNTIME
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
