"""Engine profiles: which fast and smart models the daemons serve (`python -m jev_mcp.engine_profiles`).

Interface frozen by docs/superpowers/specs/2026-10-14-engine-profiles-cli-contract.md;
lane P1-a implements the bodies without changing these signatures.
"""

import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_PROFILE = "default"
DEFAULT_GATE = 0.85
DEFAULT_CONFIG_PATH = os.path.join(REPO_ROOT, "engine_profiles.toml")
DEFAULT_MODELS_DIR = os.path.join(REPO_ROOT, "models")
ACTIVE_PROFILE_PATH = os.path.expanduser("~/.jev/active_profile")


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
        raise NotImplementedError

    def url(self) -> str:
        raise NotImplementedError


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
        raise NotImplementedError


def load_profiles(path: str = DEFAULT_CONFIG_PATH) -> ProfileTable:
    raise NotImplementedError


def read_active_name(path: str = ACTIVE_PROFILE_PATH) -> str:
    raise NotImplementedError


def active_profile(
    config_path: str = DEFAULT_CONFIG_PATH, marker_path: str = ACTIVE_PROFILE_PATH
) -> Profile | None:
    raise NotImplementedError


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
    raise NotImplementedError


def main(argv: Sequence[str]) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
