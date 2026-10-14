"""Contract shape of jev_mcp.engine_profiles (the frozen interface, before lane P1-a)."""

import dataclasses
import os

import pytest

from jev_mcp import engine_profiles as ep

MODEL = ep.ModelSpec(
    name="kev-0.8b",
    tier="fast",
    repo="ggml-org/Kev-0.8B-GGUF",
    revision="e551e319d483ff57e1ff208924b349d397cffc1c",
    filename="Kev-0.8B-Q8_0.gguf",
    sha256="27278f34eb3273bceea4c053dc50dd61a5161da21a718c4aacdf8fd5830771d0",
    parallel=4,
    ctx_per_slot=16384,
    reasoning="auto",
    style="plain",
)


def test_defaults_match_the_contract():
    assert ep.DEFAULT_PROFILE == "default"
    assert ep.DEFAULT_GATE == 0.85
    assert ep.DEFAULT_CONFIG_PATH == os.path.join(ep.REPO_ROOT, "engine_profiles.toml")
    assert ep.DEFAULT_MODELS_DIR == os.path.join(ep.REPO_ROOT, "models")
    assert ep.ACTIVE_PROFILE_PATH.endswith(os.path.join(".jev", "active_profile"))
    assert issubclass(ep.ProfileError, ValueError)


def test_specs_are_frozen():
    profile = ep.Profile("default", MODEL, MODEL, ep.DEFAULT_GATE, False)
    table = ep.ProfileTable({"kev-0.8b": MODEL}, {"default": profile})
    for value in (MODEL, profile, table):
        with pytest.raises(dataclasses.FrozenInstanceError):
            value.name = "x"  # type: ignore[misc, attr-defined]


@pytest.mark.parametrize(
    "call",
    [
        lambda: MODEL.path("/models"),
        lambda: MODEL.url(),
        lambda: ep.ProfileTable({}, {}).get("default"),
        lambda: ep.load_profiles(),
        lambda: ep.read_active_name(),
        lambda: ep.active_profile(),
        lambda: ep.launchd_plist(
            MODEL,
            models_dir="/m",
            port=8080,
            label="l",
            log_path="/l",
            llama_server="/b",
            workdir="/w",
        ),
        lambda: ep.main([]),
    ],
)
def test_skeleton_is_unimplemented_until_p1a(call):
    with pytest.raises(NotImplementedError):
        call()
