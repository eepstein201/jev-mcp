"""Tests for jev_mcp.engine_profiles: validation, resolver, plist generator and CLI."""

import dataclasses
import http.client
import io
import json
import os
import plistlib
import subprocess
import sys
import urllib.error
from pathlib import Path

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
SMART = dataclasses.replace(
    MODEL,
    name="qwen3.5-4b",
    tier="smart",
    repo="bartowski/Qwen_Qwen3.5-4B-GGUF",
    filename="Qwen_Qwen3.5-4B-Q4_K_M.gguf",
    parallel=2,
    reasoning="off",
    style="sandbox",
)
SHA_A = "a" * 64
REV_A = "b" * 40

# Independent expectation of the pinned table (repo, revision, filename, sha256) per model.
PINS = {
    "kev-0.8b": (
        "ggml-org/Kev-0.8B-GGUF",
        "e551e319d483ff57e1ff208924b349d397cffc1c",
        "Kev-0.8B-Q8_0.gguf",
        "27278f34eb3273bceea4c053dc50dd61a5161da21a718c4aacdf8fd5830771d0",
    ),
    "kev-4b": (
        "ggml-org/Kev-4B-GGUF",
        "d924f2e2c3872da8b8aaf3eb4453b4126deceb79",
        "Kev-4B-Q4_K_M.gguf",
        "33ae6b18926502b2209a1bf7d3b61a350d65938441515c686bb87d226a19eff9",
    ),
    "kev-4b-q8": (
        "ggml-org/Kev-4B-GGUF",
        "d924f2e2c3872da8b8aaf3eb4453b4126deceb79",
        "Kev-4B-Q8_0.gguf",
        "7c2ebed90560522c2801389db482ac1dc4c36d828f201f6074c1d60e433948da",
    ),
    "qwen2.5-7b": (
        "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "8911e8a47f92bac19d6f5c64a2e2095bd2f7d031",
        "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423",
    ),
    "qwen3.5-4b": (
        "bartowski/Qwen_Qwen3.5-4B-GGUF",
        "4168f45a16a1290d65a4ec0fa312ae917a4c15d6",
        "Qwen_Qwen3.5-4B-Q4_K_M.gguf",
        "13c16f426047e2de38cd075bdade4a7bcbc8c774384876f677740cda65f8a983",
    ),
    "qwen3.5-4b-q6": (
        "bartowski/Qwen_Qwen3.5-4B-GGUF",
        "4168f45a16a1290d65a4ec0fa312ae917a4c15d6",
        "Qwen_Qwen3.5-4B-Q6_K.gguf",
        "5a5bc7a3f9375b395e9f81fb69109038cf160c84606b8192f13cbb8355ca59f2",
    ),
    "qwen3.5-4b-q8": (
        "bartowski/Qwen_Qwen3.5-4B-GGUF",
        "4168f45a16a1290d65a4ec0fa312ae917a4c15d6",
        "Qwen_Qwen3.5-4B-Q8_0.gguf",
        "5c74c0ede371924357dff0cb6ba145bd67208b9b2389ded681adfff3f7608db7",
    ),
    "qwen3.5-9b": (
        "bartowski/Qwen_Qwen3.5-9B-GGUF",
        "182be2fd6c7bc44887d88a91cb03ff009cc9f549",
        "Qwen_Qwen3.5-9B-Q4_K_M.gguf",
        "d784ce9eda1a5a7b51e8f705a9e6310844bf4f173654d115823c775fdea56d43",
    ),
}
EXPECTED_PROFILES = {
    "default": ("kev-0.8b", "qwen2.5-7b", False),
    "lean": ("kev-0.8b", "qwen3.5-4b", False),
    "balanced": ("kev-4b", "qwen3.5-4b", False),
    "accuracy": ("kev-4b", "qwen3.5-9b", True),
    "probe-kev4b-q8": ("kev-4b-q8", "qwen3.5-4b", True),
    "probe-q35-q6": ("kev-4b", "qwen3.5-4b-q6", True),
    "probe-q35-q8": ("kev-4b", "qwen3.5-4b-q8", True),
}

BASE_TOML = f"""
[models."m-fast"]
tier = "fast"
repo = "org/Fast-GGUF"
revision = "{REV_A}"
filename = "Fast-Q8.gguf"
sha256 = "{SHA_A}"
parallel = 4
ctx_per_slot = 16384
reasoning = "auto"
style = "plain"

[models."m-smart"]
tier = "smart"
repo = "org/Smart-GGUF"
revision = "{REV_A}"
filename = "Smart-Q4.gguf"
sha256 = "{SHA_A}"
parallel = 2
ctx_per_slot = 8192
reasoning = "off"
style = "sandbox"

[profiles.default]
fast = "m-fast"
smart = "m-smart"
"""


@pytest.fixture
def scratch_marker(tmp_path, monkeypatch):
    """Point the module-level marker default at a tmp path so the real ~/.jev is never touched."""
    marker = tmp_path / "home" / ".jev" / "active_profile"
    monkeypatch.setattr(ep, "ACTIVE_PROFILE_PATH", str(marker))
    return marker


def write_config(tmp_path: Path, text: str = BASE_TOML) -> str:
    path = tmp_path / "profiles.toml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def mutated(old: str, new: str) -> str:
    assert old in BASE_TOML
    return BASE_TOML.replace(old, new, 1)


# --- frozen contract ---------------------------------------------------------------------


def test_defaults_match_the_contract():
    assert ep.DEFAULT_PROFILE == "default"
    assert ep.DEFAULT_GATE == 0.85
    assert ep.DEFAULT_CONFIG_PATH == os.path.join(ep.REPO_ROOT, "engine_profiles.toml")
    assert ep.DEFAULT_MODELS_DIR == os.path.join(ep.REPO_ROOT, "models")
    assert ep.ACTIVE_PROFILE_PATH.endswith(os.path.join(".jev", "active_profile"))
    assert issubclass(ep.ProfileError, ValueError)


def test_specs_are_frozen():
    profile = ep.Profile("default", MODEL, SMART, ep.DEFAULT_GATE, False)
    table = ep.ProfileTable({"kev-0.8b": MODEL}, {"default": profile})
    for value in (MODEL, profile, table):
        with pytest.raises(dataclasses.FrozenInstanceError):
            value.name = "x"  # type: ignore[misc, attr-defined]


# --- ModelSpec ---------------------------------------------------------------------------


def test_model_url_pins_the_revision():
    assert MODEL.url() == (
        "https://huggingface.co/ggml-org/Kev-0.8B-GGUF/resolve/"
        "e551e319d483ff57e1ff208924b349d397cffc1c/Kev-0.8B-Q8_0.gguf"
    )


def test_model_path_uses_repo_name_and_filename(tmp_path):
    expected = os.path.join(str(tmp_path), "Kev-0.8B-GGUF", "Kev-0.8B-Q8_0.gguf")
    assert MODEL.path(str(tmp_path)) == expected


def test_model_path_of_relative_models_dir_is_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert os.path.isabs(MODEL.path("models"))


def test_model_path_rejects_dotdot_escape(tmp_path):
    evil = dataclasses.replace(MODEL, repo="org/..", filename="x.gguf")
    with pytest.raises(ep.ProfileError, match="outside"):
        evil.path(str(tmp_path / "models"))


def test_model_path_rejects_symlinked_escape(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (models / "Kev-0.8B-GGUF").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ep.ProfileError, match="outside"):
        MODEL.path(str(models))


def test_model_path_allows_symlinked_models_dir(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    assert MODEL.path(str(link)).startswith(str(link))


def test_model_path_requires_a_repo_with_a_slash(tmp_path):
    with pytest.raises(ep.ProfileError, match="repo"):
        dataclasses.replace(MODEL, repo="noslash").path(str(tmp_path))


# --- load_profiles: the real table -------------------------------------------------------


def test_real_table_loads_and_every_profile_resolves(tmp_path):
    table = ep.load_profiles(ep.DEFAULT_CONFIG_PATH)
    assert set(table.models) == set(PINS)
    assert set(table.profiles) == set(EXPECTED_PROFILES)
    for name, (fast, smart, eval_only) in EXPECTED_PROFILES.items():
        profile = table.get(name)
        assert (profile.fast.name, profile.smart.name, profile.eval_only) == (
            fast,
            smart,
            eval_only,
        )
        assert profile.gate == ep.DEFAULT_GATE
        assert profile.fast.path(str(tmp_path)).endswith(profile.fast.filename)
        assert profile.smart.url().startswith("https://huggingface.co/")


@pytest.mark.parametrize("name", sorted(PINS))
def test_real_table_matches_pinned_repo_revision_filename_sha(name):
    spec = ep.load_profiles(ep.DEFAULT_CONFIG_PATH).models[name]
    assert (spec.repo, spec.revision, spec.filename, spec.sha256) == PINS[name]


def test_real_table_settings_follow_the_contract():
    for name, spec in ep.load_profiles(ep.DEFAULT_CONFIG_PATH).models.items():
        assert spec.ctx_per_slot == 16384
        assert spec.parallel == (4 if spec.tier == "fast" else 2)
        if name.startswith("kev"):
            assert (spec.reasoning, spec.style) == ("auto", "plain")
        elif name == "qwen2.5-7b":
            assert (spec.reasoning, spec.style) == ("auto", "sandbox")
        else:
            assert (spec.reasoning, spec.style) == ("off", "sandbox")


# --- load_profiles: validation -----------------------------------------------------------


def test_minimal_table_loads_with_defaults(tmp_path):
    profile = ep.load_profiles(write_config(tmp_path)).get("default")
    assert (profile.gate, profile.eval_only) == (0.85, False)
    assert profile.fast.name == "m-fast" and profile.smart.ctx_per_slot == 8192


def test_optional_profile_keys_are_honoured(tmp_path):
    text = mutated(
        'smart = "m-smart"', 'smart = "m-smart"\ngate = 0.7\neval_only = true'
    )
    profile = ep.load_profiles(write_config(tmp_path, text)).get("default")
    assert (profile.gate, profile.eval_only) == (0.7, True)


@pytest.mark.parametrize(
    "bad",
    [
        "../x",
        "a;b",
        "a\nb",
        "a&b",
        "a<b",
        "",
        "a" * 33,
        "Upper",
        "-lead",
        ".lead",
        "a b",
    ],
)
def test_hostile_model_and_profile_names_rejected(tmp_path, bad):
    quoted = json.dumps(bad)
    for text in (
        BASE_TOML.replace('[models."m-fast"]', f"[models.{quoted}]").replace(
            'fast = "m-fast"', f"fast = {quoted}"
        ),
        BASE_TOML.replace("[profiles.default]", f"[profiles.{quoted}]"),
    ):
        with pytest.raises(ep.ProfileError) as info:
            ep.load_profiles(write_config(tmp_path, text))
        assert "\n" not in str(info.value)


def test_name_of_32_chars_is_accepted(tmp_path):
    name = "a" * 32
    text = BASE_TOML.replace("[profiles.default]", f"[profiles.{name}]")
    assert name in ep.load_profiles(write_config(tmp_path, text)).profiles


@pytest.mark.parametrize(
    "text, key",
    [
        (BASE_TOML + "\n[extra]\nx = 1\n", "extra"),
        (mutated('style = "plain"', 'style = "plain"\nmystery = 1'), "mystery"),
        (mutated('smart = "m-smart"', 'smart = "m-smart"\nmystery = 1'), "mystery"),
    ],
)
def test_unknown_keys_rejected_at_every_level(tmp_path, text, key):
    with pytest.raises(ep.ProfileError, match=key):
        ep.load_profiles(write_config(tmp_path, text))


@pytest.mark.parametrize(
    "key", ["revision", "sha256", "tier", "repo", "filename", "parallel", "style"]
)
def test_missing_model_key_rejected(tmp_path, key):
    first_line = next(
        line for line in BASE_TOML.splitlines() if line.startswith(f"{key} =")
    )
    text = BASE_TOML.replace(
        first_line + "\n", "", 1
    )  # drops it from the fast model only
    with pytest.raises(ep.ProfileError, match=f"m-fast.{key}"):
        ep.load_profiles(write_config(tmp_path, text))


@pytest.mark.parametrize("key", ["fast", "smart"])
def test_missing_profile_key_rejected(tmp_path, key):
    text = BASE_TOML.replace(f'{key} = "m-{key}"\n', "")
    with pytest.raises(ep.ProfileError, match=key):
        ep.load_profiles(write_config(tmp_path, text))


@pytest.mark.parametrize("table", ["models", "profiles"])
def test_missing_top_level_table_rejected(tmp_path, table):
    text = (
        "[profiles.p]\n" if table == "models" else '[models."m-fast"]\ntier = "fast"\n'
    )
    with pytest.raises(ep.ProfileError, match=table):
        ep.load_profiles(write_config(tmp_path, text))


def test_profile_naming_wrong_tier_rejected(tmp_path):
    text = mutated('fast = "m-fast"', 'fast = "m-smart"')
    with pytest.raises(ep.ProfileError, match="fast"):
        ep.load_profiles(write_config(tmp_path, text))


def test_profile_naming_unknown_model_rejected(tmp_path):
    text = mutated('smart = "m-smart"', 'smart = "ghost"')
    with pytest.raises(ep.ProfileError, match="ghost"):
        ep.load_profiles(write_config(tmp_path, text))


@pytest.mark.parametrize(
    "old, new, key",
    [
        (f'revision = "{REV_A}"', 'revision = "main"', "revision"),
        (f'revision = "{REV_A}"', f'revision = "{"B" * 40}"', "revision"),
        (f'revision = "{REV_A}"', f'revision = "{"b" * 39}"', "revision"),
        (f'sha256 = "{SHA_A}"', f'sha256 = "{"a" * 63}"', "sha256"),
        (f'sha256 = "{SHA_A}"', f'sha256 = "{"g" * 64}"', "sha256"),
        ('filename = "Fast-Q8.gguf"', 'filename = "Fast-Q8.bin"', "filename"),
        ('filename = "Fast-Q8.gguf"', 'filename = "../Fast.gguf"', "filename"),
        ('filename = "Fast-Q8.gguf"', 'filename = "a\\nb.gguf"', "filename"),
        ('repo = "org/Fast-GGUF"', 'repo = "orgFast"', "repo"),
        ('repo = "org/Fast-GGUF"', 'repo = "a/b/c"', "repo"),
        ('repo = "org/Fast-GGUF"', 'repo = "../.."', "repo"),
        ('repo = "org/Fast-GGUF"', 'repo = "org/Fast GGUF"', "repo"),
        ('tier = "fast"', 'tier = "medium"', "tier"),
        ('reasoning = "auto"', 'reasoning = "maybe"', "reasoning"),
        ('style = "plain"', 'style = "fancy"', "style"),
        ("parallel = 4", "parallel = 0", "parallel"),
        ("parallel = 4", "parallel = 17", "parallel"),
        ("parallel = 4", "parallel = true", "parallel"),
        ("parallel = 4", "parallel = 4.0", "parallel"),
        ("parallel = 4", 'parallel = "4"', "parallel"),
        ("ctx_per_slot = 16384", "ctx_per_slot = 511", "ctx_per_slot"),
        ("ctx_per_slot = 16384", "ctx_per_slot = 131073", "ctx_per_slot"),
        ("ctx_per_slot = 16384", "ctx_per_slot = false", "ctx_per_slot"),
    ],
)
def test_invalid_model_values_rejected(tmp_path, old, new, key):
    with pytest.raises(ep.ProfileError, match=key):
        ep.load_profiles(write_config(tmp_path, mutated(old, new)))


@pytest.mark.parametrize(
    "gate", ["0", "1", "0.0", "1.0", "-0.5", "1.5", "true", '"0.5"', "nan", "1e9"]
)
def test_invalid_gate_rejected(tmp_path, gate):
    text = mutated('smart = "m-smart"', f'smart = "m-smart"\ngate = {gate}')
    with pytest.raises(ep.ProfileError, match="gate"):
        ep.load_profiles(write_config(tmp_path, text))


def test_non_bool_eval_only_rejected(tmp_path):
    text = mutated('smart = "m-smart"', 'smart = "m-smart"\neval_only = 1')
    with pytest.raises(ep.ProfileError, match="eval_only"):
        ep.load_profiles(write_config(tmp_path, text))


def test_models_must_be_tables(tmp_path):
    with pytest.raises(ep.ProfileError, match="models"):
        ep.load_profiles(
            write_config(tmp_path, 'models = 3\n[profiles.p]\nfast = "a"\n')
        )


def test_model_entry_must_be_a_table(tmp_path):
    with pytest.raises(ep.ProfileError, match="m"):
        ep.load_profiles(write_config(tmp_path, "[models]\nm = 3\n[profiles.p]\n"))


def test_invalid_toml_is_a_profile_error(tmp_path):
    with pytest.raises(ep.ProfileError, match="TOML"):
        ep.load_profiles(write_config(tmp_path, "[models\n"))


def test_deeply_nested_toml_is_a_profile_error(tmp_path):
    depth = 200_000
    with pytest.raises(ep.ProfileError, match="TOML") as info:
        ep.load_profiles(write_config(tmp_path, "a = " + "[" * depth + "]" * depth))
    assert "\n" not in str(info.value)


def test_any_decode_failure_is_a_profile_error(tmp_path, monkeypatch):
    def broken(_handle):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(ep.tomllib, "load", broken)
    with pytest.raises(ep.ProfileError, match="TOML"):
        ep.load_profiles(write_config(tmp_path))


def test_non_utf8_toml_is_a_profile_error(tmp_path):
    path = tmp_path / "profiles.toml"
    path.write_bytes(b"\xff\xfe = 1\n")
    with pytest.raises(ep.ProfileError, match="TOML"):
        ep.load_profiles(str(path))


def test_missing_config_file_is_a_profile_error(tmp_path):
    with pytest.raises(ep.ProfileError, match="cannot read"):
        ep.load_profiles(str(tmp_path / "nope.toml"))


def test_table_get_unknown_and_invalid_names(tmp_path):
    table = ep.load_profiles(write_config(tmp_path))
    with pytest.raises(ep.ProfileError, match="ghost"):
        table.get("ghost")
    with pytest.raises(ep.ProfileError):
        table.get("../x")
    with pytest.raises(ep.ProfileError) as info:
        table.get("a\nb")
    assert "\n" not in str(info.value)


# --- active marker -----------------------------------------------------------------------


def test_read_active_name_missing_file_is_default(tmp_path):
    assert ep.read_active_name(str(tmp_path / "none")) == "default"


def test_read_active_name_reads_one_line(tmp_path):
    marker = tmp_path / "active_profile"
    marker.write_text("lean\n")
    assert ep.read_active_name(str(marker)) == "lean"


@pytest.mark.parametrize(
    "content", ["", "\n", "  \n", "../x\n", "a\nb\n", "UPPER\n", "x" * 40]
)
def test_read_active_name_rejects_empty_or_invalid(tmp_path, content):
    marker = tmp_path / "active_profile"
    marker.write_text(content)
    with pytest.raises(ep.ProfileError):
        ep.read_active_name(str(marker))


def test_read_active_name_rejects_unreadable_marker(tmp_path):
    with pytest.raises(ep.ProfileError, match="cannot read"):
        ep.read_active_name(str(tmp_path))  # a directory


def test_read_active_name_rejects_binary_marker(tmp_path):
    marker = tmp_path / "active_profile"
    marker.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(ep.ProfileError):
        ep.read_active_name(str(marker))


def test_active_profile_returns_the_marked_profile(tmp_path):
    config = write_config(tmp_path)
    marker = tmp_path / "active_profile"
    marker.write_text("default\n")
    assert ep.active_profile(config, str(marker)).name == "default"


def test_active_profile_without_marker_is_default(tmp_path):
    assert (
        ep.active_profile(write_config(tmp_path), str(tmp_path / "none")).name
        == "default"
    )


def test_active_profile_never_raises(tmp_path, caplog):
    config = write_config(tmp_path)
    marker = tmp_path / "active_profile"
    marker.write_text("ghost\n")
    with caplog.at_level("WARNING"):
        assert ep.active_profile(config, str(marker)) is None
        assert ep.active_profile(str(tmp_path / "nope.toml"), str(marker)) is None
    assert "ghost" in caplog.text


# --- launchd plist -----------------------------------------------------------------------

PLIST_KW = dict(
    models_dir="/models",
    port=8080,
    label="com.jev.fast",
    log_path="/var/log/jev.log",
    llama_server="/opt/bin/llama-server",
    workdir="/work",
)


def load_plist(model=MODEL, **overrides) -> dict:
    return plistlib.loads(ep.launchd_plist(model, **{**PLIST_KW, **overrides}))


def test_plist_program_arguments_exact_order():
    plist = load_plist()
    assert plist["ProgramArguments"] == [
        "/opt/bin/llama-server",
        "-m",
        "/models/Kev-0.8B-GGUF/Kev-0.8B-Q8_0.gguf",
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
        "--alias",
        "kev-0.8b",
        "--parallel",
        "4",
        "-c",
        "65536",
        "--log-verbosity",
        "3",
    ]


def test_plist_static_keys():
    plist = load_plist()
    assert plist["Label"] == "com.jev.fast"
    assert plist["RunAtLoad"] is True and plist["KeepAlive"] is True
    assert plist["StandardOutPath"] == plist["StandardErrorPath"] == "/var/log/jev.log"
    assert plist["WorkingDirectory"] == "/work"


def test_plist_context_is_parallel_times_ctx_per_slot():
    plist = load_plist(SMART)
    args = plist["ProgramArguments"]
    assert args[args.index("-c") + 1] == str(2 * 16384)
    assert args[args.index("--parallel") + 1] == "2"
    assert args[args.index("--alias") + 1] == "qwen3.5-4b"


@pytest.mark.parametrize(
    "reasoning, expected",
    [("auto", None), ("off", "off"), ("on", "on")],
)
def test_plist_reasoning_flag(reasoning, expected):
    args = load_plist(dataclasses.replace(MODEL, reasoning=reasoning))[
        "ProgramArguments"
    ]
    if expected is None:
        assert "--reasoning" not in args
    else:
        assert args[-2:] == ["--reasoning", expected]


def test_plist_sleep_idle_seconds_only_when_given():
    assert "--sleep-idle-seconds" not in load_plist()["ProgramArguments"]
    args = load_plist(sleep_idle_seconds=300)["ProgramArguments"]
    assert args[-2:] == ["--sleep-idle-seconds", "300"]


def test_plist_reasoning_precedes_sleep_idle():
    args = load_plist(SMART, sleep_idle_seconds=60)["ProgramArguments"]
    assert args[-4:] == ["--reasoning", "off", "--sleep-idle-seconds", "60"]


def test_plist_special_characters_round_trip():
    odd = "/Users/o'brien & <co>/jev models"
    plist = load_plist(
        models_dir=odd,
        log_path=f"{odd}/a&b.log",
        workdir=odd,
        llama_server=f"{odd}/llama server",
    )
    args = plist["ProgramArguments"]
    assert args[0] == f"{odd}/llama server"
    assert args[2] == f"{odd}/Kev-0.8B-GGUF/Kev-0.8B-Q8_0.gguf"
    assert plist["StandardOutPath"] == f"{odd}/a&b.log"
    assert plist["WorkingDirectory"] == odd


def test_plist_works_with_the_managers_awk_port_reader(tmp_path):
    plist_path = tmp_path / "fast.plist"
    plist_path.write_bytes(ep.launchd_plist(MODEL, **{**PLIST_KW, "port": 8123}))
    awk = r"/<string>--port<\/string>/ { getline; gsub(/.*<string>|<\/string>.*/, \"\"); print; exit }"
    out = subprocess.run(
        ["awk", awk.replace('\\"', '"'), str(plist_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "8123"


@pytest.mark.parametrize(
    "field, value",
    [
        ("name", "--x y"),
        ("name", "Upper"),
        ("name", "a\nb"),
        ("tier", "medium"),
        ("reasoning", "bogus"),
        ("style", "fancy"),
        ("repo", "../.."),
        ("repo", "noslash"),
        ("revision", "main"),
        ("filename", "../x.gguf"),
        ("sha256", "z" * 64),
        ("parallel", 0),
        ("parallel", True),
        ("parallel", 99),
        ("ctx_per_slot", 1),
        ("ctx_per_slot", "16384"),
    ],
)
def test_plist_revalidates_a_hand_built_spec(field, value):
    hostile = dataclasses.replace(MODEL, **{field: value})
    with pytest.raises(ep.ProfileError):
        ep.launchd_plist(hostile, **PLIST_KW)


@pytest.mark.parametrize(
    "override",
    [
        {"port": 0},
        {"port": 65536},
        {"port": True},
        {"port": "8080"},
        {"label": ""},
        {"label": "a b"},
        {"label": "a\nb"},
        {"label": "a/b"},
        {"log_path": "relative.log"},
        {"llama_server": "llama-server"},
        {"workdir": "work"},
        {"sleep_idle_seconds": 0},
        {"sleep_idle_seconds": -5},
        {"sleep_idle_seconds": True},
        {"sleep_idle_seconds": 10**9},
    ],
)
def test_plist_rejects_invalid_inputs(override):
    with pytest.raises(ep.ProfileError):
        ep.launchd_plist(MODEL, **{**PLIST_KW, **override})


# --- CLI: resolve ------------------------------------------------------------------------


def run_main(capsys, *argv: str) -> tuple[int, str, str]:
    code = ep.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_resolve_prints_exact_lines(tmp_path, capsys):
    config = write_config(tmp_path)
    models = tmp_path / "models"
    code, out, err = run_main(
        capsys, "resolve", "default", "--config", config, "--models-dir", str(models)
    )
    assert (code, err) == (0, "")
    assert out.splitlines() == [
        "profile=default",
        "fast_model=m-fast",
        f"fast_path={models}/Fast-GGUF/Fast-Q8.gguf",
        f"fast_url=https://huggingface.co/org/Fast-GGUF/resolve/{REV_A}/Fast-Q8.gguf",
        f"fast_sha256={SHA_A}",
        "smart_model=m-smart",
        f"smart_path={models}/Smart-GGUF/Smart-Q4.gguf",
        f"smart_url=https://huggingface.co/org/Smart-GGUF/resolve/{REV_A}/Smart-Q4.gguf",
        f"smart_sha256={SHA_A}",
    ]
    assert out.endswith("\n")


def test_resolve_omitted_profile_reads_the_marker(tmp_path, capsys, scratch_marker):
    config = write_config(
        tmp_path, BASE_TOML + '\n[profiles.other]\nfast = "m-fast"\nsmart = "m-smart"\n'
    )
    scratch_marker.parent.mkdir(parents=True)
    scratch_marker.write_text("other\n")
    code, out, _ = run_main(
        capsys, "resolve", "--config", config, "--models-dir", str(tmp_path)
    )
    assert code == 0 and out.splitlines()[0] == "profile=other"


def test_resolve_omitted_profile_without_marker_is_default(
    tmp_path, capsys, scratch_marker
):
    config = write_config(tmp_path)
    code, out, _ = run_main(
        capsys, "resolve", "--config", config, "--models-dir", str(tmp_path)
    )
    assert code == 0 and out.splitlines()[0] == "profile=default"


def test_resolve_invalid_marker_is_exit_2(tmp_path, capsys, scratch_marker):
    scratch_marker.parent.mkdir(parents=True)
    scratch_marker.write_text("../x\n")
    code, out, err = run_main(capsys, "resolve", "--config", write_config(tmp_path))
    assert (code, out) == (2, "")
    assert err.count("\n") == 1


def test_resolve_eval_only_refused_without_flag(tmp_path, capsys):
    text = mutated('smart = "m-smart"', 'smart = "m-smart"\neval_only = true')
    config = write_config(tmp_path, text)
    code, out, err = run_main(
        capsys, "resolve", "default", "--config", config, "--models-dir", str(tmp_path)
    )
    assert (code, out) == (2, "")
    assert "eval-only" in err and err.count("\n") == 1
    code, out, _ = run_main(
        capsys,
        "resolve",
        "default",
        "--allow-eval-only",
        "--config",
        config,
        "--models-dir",
        str(tmp_path),
    )
    assert code == 0 and out.splitlines()[0] == "profile=default"


@pytest.mark.parametrize("name", ["ghost", "../x", "a;b", "UP"])
def test_resolve_unknown_or_hostile_profile_is_exit_2(tmp_path, capsys, name):
    code, out, err = run_main(
        capsys,
        "resolve",
        name,
        "--config",
        write_config(tmp_path),
        "--models-dir",
        str(tmp_path),
    )
    assert (code, out) == (2, "")
    assert err.startswith("error:") and err.count("\n") == 1


def test_resolve_deeply_nested_toml_is_exit_2_with_one_line(tmp_path, capsys):
    depth = 200_000
    config = write_config(tmp_path, "a = " + "[" * depth + "]" * depth)
    code, out, err = run_main(capsys, "resolve", "default", "--config", config)
    assert (code, out) == (2, "") and err.count("\n") == 1 and "Traceback" not in err


def test_resolve_bad_toml_is_exit_2(tmp_path, capsys):
    code, out, err = run_main(
        capsys, "resolve", "default", "--config", write_config(tmp_path, "[x")
    )
    assert (code, out) == (2, "") and err.count("\n") == 1


def test_resolve_rejects_control_characters_in_paths(tmp_path, capsys):
    code, out, err = run_main(
        capsys,
        "resolve",
        "default",
        "--config",
        write_config(tmp_path),
        "--models-dir",
        str(tmp_path / "bad\nmodels"),
    )
    assert (code, out) == (2, "") and err.count("\n") == 1


@pytest.mark.parametrize("char", ["\x85", "\x80", "\x9f", "\x7f", "\x00"])
def test_control_characters_incl_c1_are_rejected_in_paths(tmp_path, capsys, char):
    code, out, err = run_main(
        capsys,
        "resolve",
        "default",
        "--config",
        write_config(tmp_path),
        "--models-dir",
        str(tmp_path / f"bad{char}models"),
    )
    assert (code, out) == (2, "") and err.count("\n") == 1


def test_plist_rejects_c1_control_characters_in_paths():
    with pytest.raises(ep.ProfileError, match="control"):
        ep.launchd_plist(MODEL, **{**PLIST_KW, "workdir": "/work\x85dir"})


def test_non_control_unicode_in_paths_is_accepted(tmp_path, capsys):
    code, out, _ = run_main(
        capsys,
        "resolve",
        "default",
        "--config",
        write_config(tmp_path),
        "--models-dir",
        str(tmp_path / "modèles"),
    )
    assert code == 0 and "modèles" in out


@pytest.mark.parametrize("command", ["resolve", "plist", "activate"])
def test_empty_models_dir_is_exit_2(tmp_path, capsys, daemons, command):
    if command == "resolve":
        argv = ["resolve", "default", "--config", write_config(tmp_path)]
    elif command == "plist":
        argv = plist_argv(tmp_path, write_config(tmp_path))
    else:
        argv = activate_argv(tmp_path, tmp_path / "m")
    if "--models-dir" in argv:
        argv[argv.index("--models-dir") + 1] = ""
    else:
        argv += ["--models-dir", ""]
    code, out, err = run_main(capsys, *argv)
    assert (code, out) == (2, "") and "models-dir" in err and err.count("\n") == 1
    assert daemons.requests == []
    assert not (tmp_path / "out.plist").exists() and not (tmp_path / "m").exists()


def test_resolve_the_real_table_with_default_paths(capsys):
    code, out, _ = run_main(capsys, "resolve", "default")
    assert code == 0
    lines = dict(line.split("=", 1) for line in out.splitlines())
    assert lines["fast_path"] == os.path.join(
        ep.DEFAULT_MODELS_DIR, "Kev-0.8B-GGUF", "Kev-0.8B-Q8_0.gguf"
    )


def test_unknown_subcommand_and_missing_args_are_exit_2(capsys):
    for argv in (
        [],
        ["nope"],
        ["plist", "default"],
        ["activate"],
        ["resolve", "--bogus"],
    ):
        code, out, err = run_main(capsys, *argv)
        assert (code, out) == (2, ""), argv
        assert err.count("\n") == 1, argv


def test_help_exits_zero(capsys):
    code, out, _ = run_main(capsys, "--help")
    assert code == 0 and "resolve" in out


# --- CLI: plist --------------------------------------------------------------------------


def plist_argv(
    tmp_path: Path,
    config: str,
    *extra: str,
    profile: str = "default",
    tier: str = "fast",
):
    return [
        "plist",
        profile,
        "--tier",
        tier,
        "--port",
        "8080",
        "--label",
        "com.jev.fast",
        "--log",
        "/var/log/jev.log",
        "--llama-server",
        "/opt/bin/llama-server",
        "--workdir",
        "/work",
        "--out",
        str(tmp_path / "out.plist"),
        "--config",
        config,
        "--models-dir",
        "/models",
        *extra,
    ]


def test_plist_command_writes_file_atomically_and_prints_nothing(tmp_path, capsys):
    argv = plist_argv(tmp_path, write_config(tmp_path), "--sleep-idle-seconds", "120")
    code, out, err = run_main(capsys, *argv)
    assert (code, out, err) == (0, "", "")
    out_path = tmp_path / "out.plist"
    plist = plistlib.loads(out_path.read_bytes())
    assert plist["ProgramArguments"][2] == "/models/Fast-GGUF/Fast-Q8.gguf"
    assert plist["ProgramArguments"][-2:] == ["--sleep-idle-seconds", "120"]
    assert os.stat(out_path).st_mode & 0o777 == 0o644
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.plist", "profiles.toml"]


def test_plist_command_selects_the_smart_tier(tmp_path, capsys):
    code, *_ = run_main(
        capsys, *plist_argv(tmp_path, write_config(tmp_path), tier="smart")
    )
    assert code == 0
    args = plistlib.loads((tmp_path / "out.plist").read_bytes())["ProgramArguments"]
    assert args[args.index("--alias") + 1] == "m-smart"
    assert args[-2:] == ["--reasoning", "off"]


def test_plist_command_replaces_an_existing_file(tmp_path, capsys):
    (tmp_path / "out.plist").write_text("old")
    code, *_ = run_main(capsys, *plist_argv(tmp_path, write_config(tmp_path)))
    assert code == 0
    assert (
        plistlib.loads((tmp_path / "out.plist").read_bytes())["Label"] == "com.jev.fast"
    )


def test_plist_command_eval_only_needs_the_flag(tmp_path, capsys):
    config = write_config(
        tmp_path, mutated('smart = "m-smart"', 'smart = "m-smart"\neval_only = true')
    )
    code, _, err = run_main(capsys, *plist_argv(tmp_path, config))
    assert code == 2 and "eval-only" in err
    assert not (tmp_path / "out.plist").exists()
    code, *_ = run_main(capsys, *plist_argv(tmp_path, config, "--allow-eval-only"))
    assert code == 0 and (tmp_path / "out.plist").exists()


@pytest.mark.parametrize(
    "flag, value",
    [
        ("--label", "a b"),
        ("--label", "a\nb"),
        ("--log", "rel.log"),
        ("--llama-server", "llama-server"),
        ("--workdir", "work"),
        ("--out", "rel.plist"),
        ("--port", "0"),
        ("--port", "70000"),
        ("--port", "abc"),
        ("--tier", "medium"),
        ("--sleep-idle-seconds", "0"),
        ("--sleep-idle-seconds", "abc"),
    ],
)
def test_plist_command_invalid_inputs_are_exit_2(tmp_path, capsys, flag, value):
    argv = plist_argv(tmp_path, write_config(tmp_path))
    if flag in argv:
        argv[argv.index(flag) + 1] = value
    else:
        argv += [flag, value]
    code, out, err = run_main(capsys, *argv)
    assert (code, out) == (2, "") and err.count("\n") == 1
    assert not (tmp_path / "out.plist").exists()


def test_plist_command_cleans_up_the_temp_file_when_replace_fails(
    tmp_path, capsys, monkeypatch
):
    def broken_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(ep.os, "replace", broken_replace)
    code, out, err = run_main(capsys, *plist_argv(tmp_path, write_config(tmp_path)))
    assert (code, out) == (1, "") and "disk full" in err
    assert sorted(p.name for p in tmp_path.iterdir()) == ["profiles.toml"]


def test_plist_command_write_failure_is_exit_1(tmp_path, capsys):
    argv = plist_argv(tmp_path, write_config(tmp_path))
    argv[argv.index("--out") + 1] = str(tmp_path / "missing-dir" / "out.plist")
    code, out, err = run_main(capsys, *argv)
    assert (code, out) == (1, "") and err.count("\n") == 1


# --- CLI: activate -----------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status: int, body: dict | bytes):
        self.status = status
        self._body = body if isinstance(body, bytes) else json.dumps(body).encode()

    def read(self, size: int = -1) -> bytes:
        return self._body if size < 0 else self._body[:size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "http://127.0.0.1/props", code, "x", {}, io.BytesIO(b"")
    )  # type: ignore[arg-type]


def props_for(spec: ep.ModelSpec, models_dir: Path, **overrides) -> dict:
    body: dict = {
        "model_path": spec.path(str(models_dir)),
        "total_slots": spec.parallel,
        "default_generation_settings": {"n_ctx": spec.ctx_per_slot},
    }
    body.update(overrides)
    return body


class Daemons:
    """Scripted /props answers per port; records every request."""

    def __init__(self):
        self.script: dict[int, list] = {}
        self.requests: list[str] = []

    def urlopen(self, request, timeout=None):
        url = request if isinstance(request, str) else request.full_url
        self.requests.append(url)
        port = int(url.split(":")[2].split("/")[0])
        queue = self.script[port]
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, BaseException):
            raise item
        return item


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def daemons(monkeypatch):
    fake = Daemons()
    clock = Clock()
    monkeypatch.setattr(ep.urllib.request, "urlopen", fake.urlopen)
    monkeypatch.setattr(ep, "_sleep", clock.sleep)
    monkeypatch.setattr(ep, "_monotonic", clock.monotonic)
    fake.clock = clock  # type: ignore[attr-defined]
    return fake


def activate_argv(tmp_path: Path, marker: Path, *extra: str) -> list[str]:
    return [
        "activate",
        "default",
        "--fast-port",
        "8080",
        "--smart-port",
        "8081",
        "--config",
        write_config(tmp_path),
        "--models-dir",
        str(tmp_path / "models"),
        "--marker",
        str(marker),
        *extra,
    ]


def healthy(daemons: Daemons, tmp_path: Path) -> ep.Profile:
    profile = ep.load_profiles(write_config(tmp_path)).get("default")
    models = tmp_path / "models"
    daemons.script[8080] = [FakeResponse(200, props_for(profile.fast, models))]
    daemons.script[8081] = [FakeResponse(200, props_for(profile.smart, models))]
    return profile


def test_activate_success_writes_marker_and_prints(tmp_path, capsys, daemons):
    healthy(daemons, tmp_path)
    marker = tmp_path / "state" / "active_profile"
    code, out, err = run_main(capsys, *activate_argv(tmp_path, marker))
    assert (code, out, err) == (0, "active profile: default\n", "")
    assert marker.read_text() == "default\n"
    assert daemons.requests == [
        "http://127.0.0.1:8080/props",
        "http://127.0.0.1:8081/props",
    ]


def test_activate_default_marker_is_the_module_default(
    tmp_path, capsys, daemons, scratch_marker
):
    healthy(daemons, tmp_path)
    argv = activate_argv(tmp_path, tmp_path / "unused")
    del argv[argv.index("--marker") : argv.index("--marker") + 2]
    code, *_ = run_main(capsys, *argv)
    assert code == 0 and scratch_marker.read_text() == "default\n"


def test_activate_waits_through_503_then_succeeds(tmp_path, capsys, daemons):
    profile = healthy(daemons, tmp_path)
    good = FakeResponse(200, props_for(profile.fast, tmp_path / "models"))
    daemons.script[8080] = [http_error(503), http_error(503), good]
    marker = tmp_path / "active_profile"
    code, out, _ = run_main(capsys, *activate_argv(tmp_path, marker))
    assert code == 0 and marker.exists()
    assert daemons.clock.sleeps == [1, 1]  # type: ignore[attr-defined]


def test_activate_waits_for_a_daemon_that_is_not_up_yet(tmp_path, capsys, daemons):
    profile = healthy(daemons, tmp_path)
    good = FakeResponse(200, props_for(profile.fast, tmp_path / "models"))
    daemons.script[8080] = [
        urllib.error.URLError("refused"),
        ConnectionResetError(),
        good,
    ]
    code, *_ = run_main(capsys, *activate_argv(tmp_path, tmp_path / "active_profile"))
    assert code == 0


def test_activate_times_out_when_daemon_stays_loading(tmp_path, capsys, daemons):
    healthy(daemons, tmp_path)
    daemons.script[8081] = [http_error(503)]
    marker = tmp_path / "active_profile"
    code, out, err = run_main(
        capsys, *activate_argv(tmp_path, marker, "--timeout", "5")
    )
    assert (code, out) == (1, "")
    assert err.count("\n") == 1 and "smart" in err and "8081" in err
    assert not marker.exists()
    assert sum(daemons.clock.sleeps) >= 5  # type: ignore[attr-defined]


def test_activate_reports_both_tiers_when_both_time_out(tmp_path, capsys, daemons):
    daemons.script[8080] = [urllib.error.URLError("refused")]
    daemons.script[8081] = [http_error(503)]
    code, _, err = run_main(
        capsys, *activate_argv(tmp_path, tmp_path / "m", "--timeout", "2")
    )
    assert code == 1 and err.count("\n") == 2


def test_activate_keeps_polling_after_a_non_200_success_status(
    tmp_path, capsys, daemons
):
    profile = healthy(daemons, tmp_path)
    good = FakeResponse(200, props_for(profile.fast, tmp_path / "models"))
    daemons.script[8080] = [FakeResponse(204, b""), good]
    code, *_ = run_main(capsys, *activate_argv(tmp_path, tmp_path / "active_profile"))
    assert code == 0 and daemons.clock.sleeps == [1]  # type: ignore[attr-defined]


def test_activate_fails_fast_on_a_non_503_http_error(tmp_path, capsys, daemons):
    healthy(daemons, tmp_path)
    daemons.script[8080] = [http_error(404)]
    code, _, err = run_main(capsys, *activate_argv(tmp_path, tmp_path / "m"))
    assert code == 1 and "404" in err
    assert daemons.clock.sleeps == []  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "tier, field, override, expected_fragment",
    [
        (
            "fast",
            "model_path",
            {"model_path": "/elsewhere/other.gguf"},
            "fast model_path: expected ",
        ),
        (
            "smart",
            "total_slots",
            {"total_slots": 4},
            "smart total_slots: expected 2, served 4",
        ),
        (
            "fast",
            "n_ctx",
            {"default_generation_settings": {"n_ctx": 65536}},
            "fast n_ctx: expected 16384, served 65536",
        ),
        (
            "fast",
            "total_slots",
            {"total_slots": None},
            "fast total_slots: expected 4, served None",
        ),
        ("fast", "model_path", {"model_path": None}, "fast model_path: expected "),
    ],
)
def test_activate_each_mismatch_is_exit_1_and_marker_untouched(
    tmp_path, capsys, daemons, tier, field, override, expected_fragment
):
    profile = healthy(daemons, tmp_path)
    spec = profile.fast if tier == "fast" else profile.smart
    port = 8080 if tier == "fast" else 8081
    daemons.script[port] = [
        FakeResponse(200, props_for(spec, tmp_path / "models", **override))
    ]
    marker = tmp_path / "active_profile"
    marker.write_text("previous\n")
    code, out, err = run_main(capsys, *activate_argv(tmp_path, marker))
    assert (code, out) == (1, "")
    assert expected_fragment in err
    assert err.count("\n") == 1
    assert marker.read_text() == "previous\n"


def test_activate_reports_every_mismatching_field(tmp_path, capsys, daemons):
    profile = healthy(daemons, tmp_path)
    bad = props_for(profile.fast, tmp_path / "models", total_slots=1)
    bad["default_generation_settings"] = {"n_ctx": 1}
    daemons.script[8080] = [FakeResponse(200, bad)]
    daemons.script[8081] = [
        FakeResponse(200, props_for(profile.smart, tmp_path / "models", total_slots=9))
    ]
    code, _, err = run_main(capsys, *activate_argv(tmp_path, tmp_path / "m"))
    assert code == 1
    assert err.splitlines() == [
        "fast total_slots: expected 4, served 1",
        "fast n_ctx: expected 16384, served 1",
        "smart total_slots: expected 2, served 9",
    ]


def test_activate_accepts_top_level_n_ctx_fallback(tmp_path, capsys, daemons):
    profile = healthy(daemons, tmp_path)
    body = props_for(profile.fast, tmp_path / "models")
    del body["default_generation_settings"]
    body["n_ctx"] = 16384
    daemons.script[8080] = [FakeResponse(200, body)]
    code, *_ = run_main(capsys, *activate_argv(tmp_path, tmp_path / "active_profile"))
    assert code == 0


def test_activate_nested_n_ctx_wins_over_top_level(tmp_path, capsys, daemons):
    profile = healthy(daemons, tmp_path)
    body = props_for(profile.fast, tmp_path / "models")
    body["n_ctx"] = (
        65536  # the total context: must not be compared when nested is present
    )
    daemons.script[8080] = [FakeResponse(200, body)]
    code, *_ = run_main(capsys, *activate_argv(tmp_path, tmp_path / "active_profile"))
    assert code == 0


def test_activate_top_level_n_ctx_mismatch_is_reported(tmp_path, capsys, daemons):
    profile = healthy(daemons, tmp_path)
    body = props_for(profile.fast, tmp_path / "models")
    del body["default_generation_settings"]
    body["n_ctx"] = 65536
    daemons.script[8080] = [FakeResponse(200, body)]
    code, _, err = run_main(capsys, *activate_argv(tmp_path, tmp_path / "m"))
    assert code == 1 and "fast n_ctx: expected 16384, served 65536" in err


def test_activate_refuses_a_models_dir_that_escapes_via_symlink(
    tmp_path, capsys, daemons
):
    models = tmp_path / "models"
    models.mkdir()
    (models / "Fast-GGUF").symlink_to(tmp_path, target_is_directory=True)
    code, _, err = run_main(capsys, *activate_argv(tmp_path, tmp_path / "m"))
    assert code == 2 and "outside" in err
    assert daemons.requests == []


def test_activate_matches_a_served_path_through_a_symlinked_models_dir(
    tmp_path, capsys, daemons
):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "models"
    link.symlink_to(real, target_is_directory=True)
    profile = ep.load_profiles(write_config(tmp_path)).get("default")
    served_fast = str(real / "Fast-GGUF" / "Fast-Q8.gguf")
    served_smart = str(real / "Smart-GGUF" / "Smart-Q4.gguf")
    daemons.script[8080] = [
        FakeResponse(
            200, props_for(profile.fast, tmp_path / "models", model_path=served_fast)
        )
    ]
    daemons.script[8081] = [
        FakeResponse(
            200, props_for(profile.smart, tmp_path / "models", model_path=served_smart)
        )
    ]
    code, *_ = run_main(capsys, *activate_argv(tmp_path, tmp_path / "active_profile"))
    assert code == 0


@pytest.mark.parametrize(
    "failure",
    [
        http.client.IncompleteRead(b"par"),
        http.client.BadStatusLine("garbage"),
        http.client.RemoteDisconnected("closed"),
    ],
)
def test_activate_retries_http_client_errors(tmp_path, capsys, daemons, failure):
    profile = healthy(daemons, tmp_path)
    good = FakeResponse(200, props_for(profile.fast, tmp_path / "models"))
    daemons.script[8080] = [failure, good]
    code, *_ = run_main(capsys, *activate_argv(tmp_path, tmp_path / "active_profile"))
    assert code == 0 and daemons.clock.sleeps == [1]  # type: ignore[attr-defined]


def test_activate_persistent_http_client_error_is_one_line_exit_1(
    tmp_path, capsys, daemons
):
    healthy(daemons, tmp_path)
    daemons.script[8081] = [http.client.IncompleteRead(b"x")]
    marker = tmp_path / "active_profile"
    code, out, err = run_main(
        capsys, *activate_argv(tmp_path, marker, "--timeout", "3")
    )
    assert (code, out) == (1, "") and err.count("\n") == 1 and "smart" in err
    assert "Traceback" not in err and not marker.exists()


def test_activate_oversized_props_body_counts_as_a_failed_read(
    tmp_path, capsys, daemons
):
    healthy(daemons, tmp_path)
    huge = b" " * (ep._MAX_PROPS_BYTES + 1)
    daemons.script[8080] = [FakeResponse(200, huge)]
    marker = tmp_path / "active_profile"
    code, out, err = run_main(
        capsys, *activate_argv(tmp_path, marker, "--timeout", "2")
    )
    assert (code, out) == (1, "") and err.count("\n") == 1 and "larger than" in err
    assert not marker.exists()


def test_activate_props_body_at_the_cap_is_still_parsed(tmp_path, capsys, daemons):
    profile = healthy(daemons, tmp_path)
    body = json.dumps(props_for(profile.fast, tmp_path / "models")).encode()
    padded = body[:-1] + b" " * (ep._MAX_PROPS_BYTES - len(body)) + b"}"
    assert len(padded) == ep._MAX_PROPS_BYTES
    daemons.script[8080] = [FakeResponse(200, padded)]
    code, *_ = run_main(capsys, *activate_argv(tmp_path, tmp_path / "active_profile"))
    assert code == 0


def test_activate_invalid_json_is_exit_1(tmp_path, capsys, daemons):
    healthy(daemons, tmp_path)
    daemons.script[8080] = [FakeResponse(200, b"not json")]
    marker = tmp_path / "active_profile"
    code, _, err = run_main(capsys, *activate_argv(tmp_path, marker))
    assert code == 1 and "fast" in err and err.count("\n") == 1
    assert not marker.exists()


def test_activate_non_object_json_is_exit_1(tmp_path, capsys, daemons):
    healthy(daemons, tmp_path)
    daemons.script[8081] = [FakeResponse(200, b"[1]")]
    code, _, err = run_main(capsys, *activate_argv(tmp_path, tmp_path / "m"))
    assert code == 1 and "smart" in err


def test_activate_eval_only_needs_the_flag(tmp_path, capsys, daemons):
    argv = activate_argv(tmp_path, tmp_path / "m")  # writes the default config first
    write_config(
        tmp_path, mutated('smart = "m-smart"', 'smart = "m-smart"\neval_only = true')
    )
    code, out, err = run_main(capsys, *argv)
    assert (code, out) == (2, "") and "eval-only" in err
    assert daemons.requests == []


def test_activate_unknown_profile_is_exit_2(tmp_path, capsys, daemons):
    argv = activate_argv(tmp_path, tmp_path / "m")
    argv[1] = "ghost"
    code, *_ = run_main(capsys, *argv)
    assert code == 2 and daemons.requests == []


@pytest.mark.parametrize(
    "flag, value",
    [
        ("--fast-port", "0"),
        ("--smart-port", "x"),
        ("--timeout", "-1"),
        ("--timeout", "nan"),
        ("--timeout", "abc"),
    ],
)
def test_activate_invalid_numbers_are_exit_2(tmp_path, capsys, daemons, flag, value):
    argv = activate_argv(tmp_path, tmp_path / "m")
    if flag in argv:
        argv[argv.index(flag) + 1] = value
    else:
        argv += [flag, value]
    code, out, err = run_main(capsys, *argv)
    assert (code, out) == (2, "") and err.count("\n") == 1


def test_activate_marker_write_failure_is_exit_1(tmp_path, capsys, daemons):
    healthy(daemons, tmp_path)
    blocker = tmp_path / "blocker"
    blocker.write_text("file, not a directory")
    code, out, err = run_main(
        capsys, *activate_argv(tmp_path, blocker / "active_profile")
    )
    assert (code, out) == (1, "") and err.count("\n") == 1


def test_activate_default_timeout_is_120_seconds(tmp_path, capsys, daemons):
    daemons.script[8080] = [http_error(503)]
    daemons.script[8081] = [http_error(503)]
    code, *_ = run_main(capsys, *activate_argv(tmp_path, tmp_path / "m"))
    assert code == 1
    assert 120 <= sum(daemons.clock.sleeps) <= 121 * 2  # type: ignore[attr-defined]


# --- module entry point ------------------------------------------------------------------


def test_module_runs_as_a_script(tmp_path):
    src = os.path.join(ep.REPO_ROOT, "src")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jev_mcp.engine_profiles",
            "resolve",
            "default",
            "--models-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": src,
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(tmp_path),
        },
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == "profile=default"
