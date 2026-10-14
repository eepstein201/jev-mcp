"""Runs the real jev_mac_manager.sh against a temp HOME with stand-in launchctl/pkill/sleep.

Nothing here touches real launchd agents or processes: the stand-ins only record their arguments.
"""

import hashlib
import json
import os
import plistlib
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
MANAGER = REPO / "jev_mac_manager.sh"
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
            ["bash", str(MANAGER), *args],
            env=env,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
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


def test_uninstall_unlinks_opencode_while_the_virtualenv_still_exists(
    sandbox, tmp_path
):
    # Arrange
    _, run = sandbox
    venv = tmp_path / "work" / ".venv"
    (venv / "bin").mkdir(parents=True)
    python = venv / "bin" / "python"
    python.write_text(f'#!/bin/bash\necho "python $*" >> "{tmp_path / "calls.log"}"\n')
    python.chmod(0o755)

    # Act
    calls = run("uninstall", "--headless")

    # Assert
    assert "python -m jev_mcp.client_config unlink-opencode" in calls
    assert not venv.exists()


# --- hybrid [profile]: resolver-driven start (contract section 6) -------------------------

PAYLOAD = "weights"
PAYLOAD_SHA = hashlib.sha256(PAYLOAD.encode()).hexdigest()
OTHER_SHA = "0" * 64
CURL_FLAGS = [
    "--fail", "--location", "--proto", "=https", "--proto-redir", "=https",
    "--globoff", "--connect-timeout", "30", "--speed-limit", "10240", "--speed-time", "60",
]  # fmt: skip

# Stands in for `python -m jev_mcp.engine_profiles`: logs its argv, answers from env settings.
FAKE_PROFILES_CLI = """#!{python}
import json, os, plistlib, sys

argv = sys.argv[1:]
assert argv[:2] == ["-m", "jev_mcp.engine_profiles"], argv
sub, rest = argv[2], argv[3:]
with open(os.environ["FAKE_CALLS_LOG"], "a") as log:
    log.write("py " + json.dumps({{"sub": sub, "args": rest,
                                   "pythonpath": os.environ.get("PYTHONPATH", ""),
                                   "NO_PROXY": os.environ.get("NO_PROXY"),
                                   "no_proxy": os.environ.get("no_proxy")}}) + "\\n")
rc = int(os.environ.get("FAKE_" + sub.upper() + "_RC", "0"))
only_tier = os.environ.get("FAKE_PLIST_FAIL_TIER")  # fail just this tier's plist
if sub == "plist" and only_tier and rest[rest.index("--tier") + 1] != only_tier:
    rc = 0
if rc:
    print("error: " + sub + " failed", file=sys.stderr)
    sys.exit(rc)
if sub == "resolve":
    sys.stdout.write(open(os.environ["FAKE_RESOLVE_OUT"]).read())
elif sub == "plist":
    opts = {{rest[i]: rest[i + 1] for i in range(1, len(rest) - 1) if rest[i].startswith("--")}}
    program = [opts["--llama-server"], "-m", "model.gguf", "--host", "127.0.0.1",
               "--port", opts["--port"], "--alias", "x"]
    with open(opts["--out"], "wb") as out:
        plistlib.dump({{"Label": opts["--label"], "ProgramArguments": program}}, out)
elif sub == "activate":
    print("active profile: " + rest[0])
"""

CURL_SHIM = (
    """#!/bin/bash
echo "curl $*" >> "{log}"
out=""
signal="${{CURL_SIGNAL:-}}"
while [ $# -gt 0 ]; do
    if [ "$1" = "-o" ]; then out=$2; fi
    shift
done
printf '%s' "${{CURL_PAYLOAD:-"""
    + PAYLOAD
    + """}}" > "$out"
if [ -n "$signal" ]; then kill -"$signal" "$PPID"; fi
if [ -n "${{CURL_HANG:-}}" ]; then echo $$ > "$CURL_HANG"; exec /bin/sleep 30; fi
exit "${{CURL_RC:-0}}"
"""
)


def _default_sigint():
    """Child setup: a manager started from a runner that ignores SIGINT could not trap it."""
    signal.signal(signal.SIGINT, signal.SIG_DFL)


class Hybrid:
    """A temp HOME, a shimmed PATH and a fake resolver CLI for running `hybrid`."""

    def __init__(self, tmp_path):
        self.work = tmp_path / "work"
        self.work.mkdir(exist_ok=True)
        self.agents = tmp_path / "home" / "Library" / "LaunchAgents"
        self.log = tmp_path / "calls.log"
        self.resolve_out = tmp_path / "resolve.out"
        self.shims = tmp_path / "shims"
        self.llama = self.shims / "llama-server"
        self._shim("curl", CURL_SHIM.format(log=self.log))
        self._shim("llama-server", "#!/bin/bash\nexit 0\n")
        cli = tmp_path / "fake_cli"
        cli.write_text(FAKE_PROFILES_CLI.format(python=sys.executable))
        cli.chmod(0o755)
        self.env = {
            **{
                k: v
                for k, v in os.environ.items()
                if not k.startswith(("JEV_", "FAKE_"))
            },
            "HOME": str(tmp_path / "home"),
            "PATH": f"{self.shims}:{os.environ['PATH']}",
            "JEV_PROFILES_PYTHON": str(cli),
            "FAKE_CALLS_LOG": str(self.log),
            "FAKE_RESOLVE_OUT": str(self.resolve_out),
        }
        self.models = self.work / "models"
        self.fast_path = self.models / "Kev-0.8B-GGUF" / "Kev-0.8B-Q8_0.gguf"
        self.smart_path = self.models / "Qwen3.5-4B-GGUF" / "Qwen3.5-4B-Q4_K_M.gguf"
        self.set_resolve()

    def _shim(self, name, body):
        shim = self.shims / name
        shim.write_text(body)
        shim.chmod(0o755)

    def set_resolve(
        self, *, profile="lean", fast_sha=PAYLOAD_SHA, smart_sha=PAYLOAD_SHA, **over
    ):
        fields = {
            "profile": profile,
            "fast_model": "kev-0.8b",
            "fast_path": str(self.fast_path),
            "fast_url": "https://huggingface.co/ggml-org/Kev-0.8B-GGUF/resolve/"
            + "a" * 40
            + "/k.gguf",
            "fast_sha256": fast_sha,
            "smart_model": "qwen3.5-4b",
            "smart_path": str(self.smart_path),
            "smart_url": "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/"
            + "b" * 40
            + "/q.gguf",
            "smart_sha256": smart_sha,
            **over,
        }
        self.write_resolve("".join(f"{k}={v}\n" for k, v in fields.items()))

    def write_resolve(self, text):
        self.resolve_out.write_text(text)

    def have_models(self):
        for path in (self.fast_path, self.smart_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("existing")

    def run(self, *args, **env):
        merged = {
            k: v for k, v in {**self.env, **env}.items() if v is not None
        }  # None unsets
        return subprocess.run(
            ["bash", str(MANAGER), *args],
            env=merged,
            cwd=self.work,
            capture_output=True,
            text=True,
            timeout=60,
            preexec_fn=_default_sigint,
        )

    def calls(self):
        return self.log.read_text().splitlines()

    def py(self, sub=None):
        found = [json.loads(c[3:]) for c in self.calls() if c.startswith("py ")]
        return [c for c in found if sub is None or c["sub"] == sub]

    def curls(self):
        return [c for c in self.calls() if c.startswith("curl ")]


@pytest.fixture
def hybrid(sandbox, tmp_path):
    # `sandbox` supplies the HOME, LaunchAgents dir, launchctl/pkill/sleep shims and calls log.
    return Hybrid(tmp_path)


def option(call, name):
    return call["args"][call["args"].index(name) + 1]


def test_hybrid_without_a_name_resolves_the_active_profile_and_prints_it(hybrid):
    # Arrange
    hybrid.have_models()

    # Act
    result = hybrid.run("hybrid")

    # Assert
    assert result.returncode == 0, result.stderr
    assert "Profile: lean" in result.stdout
    (resolve,) = hybrid.py("resolve")
    assert resolve["args"] == [
        "--config", f"{hybrid.work}/engine_profiles.toml",
        "--models-dir", f"{hybrid.work}/models",
    ]  # fmt: skip
    assert resolve["pythonpath"].split(":")[0] == f"{hybrid.work}/src"


def test_hybrid_prepends_src_to_an_existing_pythonpath(hybrid):
    hybrid.have_models()

    hybrid.run("hybrid", PYTHONPATH="/elsewhere")

    assert hybrid.py("resolve")[0]["pythonpath"] == f"{hybrid.work}/src:/elsewhere"


def test_hybrid_forwards_an_explicit_profile_name(hybrid):
    hybrid.have_models()
    hybrid.set_resolve(profile="balanced")

    result = hybrid.run("hybrid", "balanced")

    assert result.returncode == 0, result.stderr
    assert hybrid.py("resolve")[0]["args"][0] == "balanced"
    assert "Profile: balanced" in result.stdout
    assert {c["args"][0] for c in hybrid.py("plist")} == {"balanced"}
    assert hybrid.py("activate")[0]["args"][0] == "balanced"


@pytest.mark.parametrize(
    "name",
    [
        "../x",
        "a;b",
        "Upper",
        "",
        "-x",
        "a b",
        "a" * 33,
        "a\n../x",
        "$(touch pwned)",
        "lean\n",
        "lean\r",
        "lean\n\n",
    ],  # fmt: skip
)
def test_hybrid_rejects_an_invalid_name_without_calling_the_cli(hybrid, name):
    result = hybrid.run("hybrid", name)

    assert result.returncode == 2
    assert hybrid.calls() == []
    assert not (hybrid.work / "pwned").exists()


def test_profile_name_check_runs_in_the_c_locale_whatever_the_callers_locale(tmp_path):
    # macOS regex ranges are code-point based, so no input can show the difference there; pin the
    # guard itself: with a DEBUG trap, LC_ALL must read C inside the check even when the caller sets
    # a UTF-8 locale (it is the guard against locale-collated [a-z] matching uppercase letters).
    seen = tmp_path / "seen"
    script = tmp_path / "probe.sh"
    script.write_text(
        f"set -T\n"
        f"eval \"$(sed -n '/^require_profile_name()/,/^}}/p' '{MANAGER}')\"\n"
        f'trap \'echo "${{LC_ALL-unset}}" >> "{seen}"\' DEBUG\n'
        f"require_profile_name lean\n"
    )

    subprocess.run(
        ["bash", str(script)], env={**os.environ, "LC_ALL": "en_US.UTF-8"}, check=True
    )

    inside = seen.read_text().split()
    assert "C" in inside


def test_hybrid_accepts_the_longest_valid_name(hybrid):
    hybrid.have_models()
    name = "a" + "b0._-" * 6 + "c"  # 32 characters

    result = hybrid.run("hybrid", name)

    assert len(name) == 32 and result.returncode == 0, result.stderr


@pytest.mark.parametrize("code", [1, 2])
def test_hybrid_stops_with_the_resolvers_exit_code(hybrid, code):
    result = hybrid.run("hybrid", "nope", FAKE_RESOLVE_RC=str(code))

    assert result.returncode == code
    assert "resolve failed" in result.stderr
    assert [c["sub"] for c in hybrid.py()] == ["resolve"]
    assert hybrid.curls() == []
    assert not any(c.startswith("launchctl") for c in hybrid.calls())


@pytest.mark.parametrize(
    "bad_line", ["surprise=1", "no equals sign", "fast_sha256", "", "PROFILE=x"]
)
def test_hybrid_rejects_malformed_resolver_output(hybrid, bad_line):
    hybrid.have_models()
    hybrid.write_resolve(bad_line + "\n" + hybrid.resolve_out.read_text())

    result = hybrid.run("hybrid")

    assert result.returncode != 0
    assert hybrid.curls() == []
    assert [c["sub"] for c in hybrid.py()] == ["resolve"]


def test_hybrid_rejects_a_duplicate_resolver_key(hybrid):
    hybrid.have_models()
    hybrid.write_resolve(
        hybrid.resolve_out.read_text() + "fast_sha256=" + OTHER_SHA + "\n"
    )

    result = hybrid.run("hybrid")

    assert result.returncode == 1
    assert "more than once" in result.stdout
    assert [c["sub"] for c in hybrid.py()] == ["resolve"]
    assert hybrid.curls() == []


def test_hybrid_logs_resolved_values_without_interpreting_backslash_escapes(hybrid):
    # Arrange: a profile name that would print an ANSI sequence and a BEL through `echo -e`
    hybrid.have_models()
    name = "a\\033[31mRED\\x07b\\ncd"
    hybrid.set_resolve(profile=name)

    # Act
    result = hybrid.run("hybrid")

    # Assert
    assert result.returncode == 0, result.stderr
    assert f"Profile: {name}" in result.stdout
    assert "\x07" not in result.stdout
    assert "\x1b[31m" not in result.stdout


def test_hybrid_rejects_resolver_output_with_a_missing_key(hybrid):
    hybrid.have_models()
    text = hybrid.resolve_out.read_text()
    hybrid.write_resolve(
        "".join(ln + "\n" for ln in text.splitlines() if "smart_url" not in ln)
    )

    result = hybrid.run("hybrid")

    assert result.returncode != 0
    assert [c["sub"] for c in hybrid.py()] == ["resolve"]


def test_hybrid_never_executes_resolver_values(hybrid):
    # Arrange: command substitutions in the profile, a model name and a model path
    hostile = hybrid.work / "models" / "$(touch PWNED_PATH).gguf"
    hybrid.set_resolve(
        profile="$(touch PWNED_PROFILE)",
        fast_model="`touch PWNED_MODEL`",
        fast_path=str(hostile),
    )
    hybrid.smart_path.parent.mkdir(parents=True)
    hybrid.smart_path.write_text("existing")

    # Act
    result = hybrid.run("hybrid")

    # Assert
    assert result.returncode == 0, result.stderr
    assert sorted(p.name for p in hybrid.work.iterdir()) == ["models"]
    assert hostile.read_text() == PAYLOAD
    assert hybrid.py("plist")[0]["args"][0] == "$(touch PWNED_PROFILE)"


def test_hybrid_downloads_a_missing_model_to_part_then_moves_it_after_the_hash_matches(
    hybrid,
):
    # Arrange: only the fast model is missing
    hybrid.have_models()
    hybrid.fast_path.unlink()

    # Act
    result = hybrid.run("hybrid")

    # Assert
    assert result.returncode == 0, result.stderr
    (curl,) = hybrid.curls()
    url = (
        "https://huggingface.co/ggml-org/Kev-0.8B-GGUF/resolve/" + "a" * 40 + "/k.gguf"
    )
    assert curl == f"curl {' '.join(CURL_FLAGS)} -o {hybrid.fast_path}.part {url}"
    assert hybrid.fast_path.read_text() == PAYLOAD
    assert not Path(f"{hybrid.fast_path}.part").exists()


def test_hybrid_downloads_both_tiers_when_both_are_missing(hybrid):
    result = hybrid.run("hybrid")

    assert result.returncode == 0, result.stderr
    assert len(hybrid.curls()) == 2
    assert hybrid.fast_path.read_text() == PAYLOAD
    assert hybrid.smart_path.read_text() == PAYLOAD


def test_hybrid_discards_a_download_whose_hash_does_not_match(hybrid):
    hybrid.have_models()
    hybrid.smart_path.unlink()
    hybrid.set_resolve(smart_sha=OTHER_SHA)

    result = hybrid.run("hybrid")

    assert result.returncode == 1
    assert not hybrid.smart_path.exists()
    assert not Path(f"{hybrid.smart_path}.part").exists()
    assert not any(c.startswith("launchctl") for c in hybrid.calls())
    assert [c["sub"] for c in hybrid.py()] == ["resolve"]


def test_hybrid_discards_a_failed_download(hybrid):
    result = hybrid.run("hybrid", CURL_RC="22")

    assert result.returncode == 1
    assert [p for p in hybrid.models.rglob("*") if p.is_file()] == []
    assert not any(c.startswith("launchctl") for c in hybrid.calls())
    assert [c["sub"] for c in hybrid.py()] == ["resolve"]


def test_hybrid_does_not_download_or_rehash_an_existing_model(hybrid):
    # Arrange: both files exist and the resolver's hashes would not match them
    hybrid.have_models()
    hybrid.set_resolve(fast_sha=OTHER_SHA, smart_sha=OTHER_SHA)

    # Act
    result = hybrid.run("hybrid")

    # Assert
    assert result.returncode == 0, result.stderr
    assert hybrid.curls() == []
    assert hybrid.fast_path.read_text() == "existing"


def test_hybrid_refuses_a_non_https_download_url(hybrid):
    hybrid.set_resolve(fast_url="--output=/tmp/x")

    result = hybrid.run("hybrid")

    assert result.returncode == 1
    assert hybrid.curls() == []


def test_hybrid_generates_each_tiers_plist_at_the_existing_label_log_and_port(hybrid):
    # Arrange
    hybrid.have_models()

    # Act
    result = hybrid.run("hybrid")

    # Assert
    assert result.returncode == 0, result.stderr
    fast, smart = hybrid.py("plist")
    home = hybrid.env["HOME"]
    for call, tier, port in ((fast, "fast", "8080"), (smart, "smart", "8081")):
        assert call["args"][0] == "lean"
        assert option(call, "--tier") == tier
        assert option(call, "--port") == port
        assert option(call, "--label") == f"com.jev.mlx_server_{tier}"
        assert option(call, "--log") == f"{home}/.jev/mlx_server_{tier}.log"
        assert (
            option(call, "--out")
            == f"{home}/Library/LaunchAgents/com.jev.mlx_server_{tier}.plist.new"
        )
        assert option(call, "--llama-server") == str(hybrid.llama)
        assert option(call, "--workdir") == str(hybrid.work)
        assert option(call, "--config") == f"{hybrid.work}/engine_profiles.toml"
        assert option(call, "--models-dir") == f"{hybrid.work}/models"


def test_hybrid_honours_custom_ports_in_plist_and_activate(hybrid):
    hybrid.have_models()

    result = hybrid.run("hybrid", JEV_FAST_PORT="9080", JEV_SMART_PORT="9081")

    assert result.returncode == 0, result.stderr
    assert [option(c, "--port") for c in hybrid.py("plist")] == ["9080", "9081"]
    (activate,) = hybrid.py("activate")
    assert option(activate, "--fast-port") == "9080"
    assert option(activate, "--smart-port") == "9081"


def test_hybrid_writes_both_plists_then_stops_old_agents_then_starts_new_ones_then_activates(
    hybrid,
):
    # Arrange: a leftover single-daemon plist and old tier plists
    hybrid.have_models()
    old = hybrid.agents / "com.jev.mlx_server.plist"
    old.write_text(PLIST.format(port=8080))
    (hybrid.agents / "com.jev.mlx_server_fast.plist").write_text(
        PLIST.format(port=8080)
    )
    (hybrid.agents / "com.jev.mlx_server_smart.plist").write_text(
        PLIST.format(port=8081)
    )

    # Act
    result = hybrid.run("hybrid")

    # Assert
    assert result.returncode == 0, result.stderr
    assert not old.exists()
    steps = [
        "launchctl bootout" if c.startswith("launchctl bootout") else
        "launchctl bootstrap" if c.startswith("launchctl bootstrap") else
        "py " + json.loads(c[3:])["sub"] if c.startswith("py ") else None
        for c in hybrid.calls()
    ]  # fmt: skip
    steps = [s for s in steps if s]
    assert steps == [
        "py resolve",
        "py plist", "py plist",
        "launchctl bootout", "launchctl bootout", "launchctl bootout",
        "launchctl bootstrap", "launchctl bootstrap",
        "py activate",
    ]  # fmt: skip
    activate = hybrid.py("activate")[0]
    assert activate["args"][:1] == ["lean"]


def test_hybrid_activate_failure_makes_the_manager_fail(hybrid):
    hybrid.have_models()

    result = hybrid.run("hybrid", FAKE_ACTIVATE_RC="1")

    assert result.returncode == 1
    assert "active profile" not in result.stdout
    assert "Hybrid Mode Active" not in result.stdout


def test_hybrid_activate_usage_error_propagates_its_exit_code(hybrid):
    hybrid.have_models()

    assert hybrid.run("hybrid", FAKE_ACTIVATE_RC="2").returncode == 2


def test_hybrid_plist_failure_makes_the_manager_fail(hybrid):
    hybrid.have_models()

    result = hybrid.run("hybrid", FAKE_PLIST_RC="2")

    assert result.returncode == 2
    assert not any(c.startswith("launchctl") for c in hybrid.calls())
    assert not list(hybrid.agents.glob("*.new"))


def seed_running_agents(hybrid, fast_port=8080, smart_port=8081):
    """Old live plists for both tiers plus a single-daemon plist; returns their bytes."""
    (hybrid.agents / "com.jev.mlx_server.plist").write_text(
        PLIST.format(port=fast_port)
    )
    for tier, port in (("fast", fast_port), ("smart", smart_port)):
        (hybrid.agents / f"com.jev.mlx_server_{tier}.plist").write_text(
            PLIST.format(port=port)
        )
    return {p.name: p.read_bytes() for p in hybrid.agents.iterdir()}


@pytest.mark.parametrize("failing_tier", ["fast", "smart"])
def test_hybrid_leaves_live_plists_and_agents_untouched_when_a_plist_fails(
    hybrid, failing_tier
):
    # Arrange: running agents (old plists) and a leftover single-daemon plist
    hybrid.have_models()
    before = seed_running_agents(hybrid)

    # Act
    result = hybrid.run("hybrid", FAKE_PLIST_FAIL_TIER=failing_tier, FAKE_PLIST_RC="2")

    # Assert
    assert result.returncode == 2
    assert [option(c, "--tier") for c in hybrid.py("plist")] == (
        ["fast"] if failing_tier == "fast" else ["fast", "smart"]
    )
    assert [c for c in hybrid.calls() if c.startswith(("launchctl", "pkill"))] == []
    assert {
        p.name: p.read_bytes() for p in hybrid.agents.iterdir()
    } == before  # no .new left
    assert [c["sub"] for c in hybrid.py()][:1] == ["resolve"]
    assert "activate" not in [c["sub"] for c in hybrid.py()]


def test_hybrid_kills_the_old_port_when_the_port_changes(hybrid):
    # Arrange: agents last started on 8080/8081; this run moves them to 9080/9081
    hybrid.have_models()
    seed_running_agents(hybrid)

    # Act
    result = hybrid.run("hybrid", JEV_FAST_PORT="9080", JEV_SMART_PORT="9081")

    # Assert: stop read the OLD live plists, then the new ones replaced them
    assert result.returncode == 0, result.stderr
    kills = " ".join(c for c in hybrid.calls() if c.startswith("pkill"))
    assert "--port 8080" in kills and "--port 8081" in kills
    assert "--port 9080" not in kills and "--port 9081" not in kills
    live = {
        tier: plistlib.loads(
            (hybrid.agents / f"com.jev.mlx_server_{tier}.plist").read_bytes()
        )
        for tier in ("fast", "smart")
    }
    assert live["fast"]["ProgramArguments"][6] == "9080"
    assert live["smart"]["ProgramArguments"][6] == "9081"
    assert not list(hybrid.agents.glob("*.new"))


def test_hybrid_moves_the_new_plists_in_after_the_stop_and_before_the_start(hybrid):
    # Arrange: record the live fast plist's port every time launchctl is called
    hybrid.have_models()
    seed_running_agents(hybrid)
    live = hybrid.agents / "com.jev.mlx_server_fast.plist"
    hybrid._shim(
        "launchctl",
        f'#!/bin/bash\necho "launchctl $1 port=$(grep -A1 -- --port {live} | tail -1 | tr -dc 0-9)" >> "{hybrid.log}"\n',
    )

    # Act
    result = hybrid.run("hybrid", JEV_FAST_PORT="9080")

    # Assert
    assert result.returncode == 0, result.stderr
    ports = [c.split("port=")[1] for c in hybrid.calls() if c.startswith("launchctl")]
    verbs = [c.split()[1] for c in hybrid.calls() if c.startswith("launchctl")]
    assert verbs == ["bootout"] * 3 + ["bootstrap"] * 2
    assert ports[:3] == ["8080"] * 3  # old plist still live while stopping
    assert ports[3:] == ["9080"] * 2  # new plist live when starting


def test_hybrid_reports_a_missing_default_python_with_its_path_and_the_fix(hybrid):
    result = hybrid.run("hybrid", JEV_PROFILES_PYTHON=None)

    assert result.returncode == 1
    assert f"{hybrid.work}/.venv/bin/python" in result.stdout
    assert "make install" in result.stdout
    assert hybrid.calls() == []


def test_hybrid_reports_a_bad_jev_profiles_python_by_its_path_without_the_install_hint(
    hybrid,
):
    bad = hybrid.work / "nope" / "python"

    result = hybrid.run("hybrid", JEV_PROFILES_PYTHON=str(bad))

    assert result.returncode == 1
    assert f"JEV_PROFILES_PYTHON points to {bad}" in result.stdout
    assert "make install" not in result.stdout
    assert hybrid.calls() == []


def test_hybrid_defaults_to_the_checkouts_venv_python(hybrid):
    # Arrange: JEV_PROFILES_PYTHON unset; the fake CLI stands in as .venv/bin/python
    hybrid.have_models()
    venv_python = hybrid.work / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    shutil.copy(hybrid.env["JEV_PROFILES_PYTHON"], venv_python)

    # Act
    result = hybrid.run("hybrid", JEV_PROFILES_PYTHON=None)

    # Assert
    assert result.returncode == 0, result.stderr
    assert [c["sub"] for c in hybrid.py()] == ["resolve", "plist", "plist", "activate"]
    assert hybrid.py("resolve")[0]["pythonpath"].split(":")[0] == f"{hybrid.work}/src"


def test_cli_calls_bypass_any_proxy_for_loopback(hybrid):
    hybrid.have_models()

    result = hybrid.run("hybrid", NO_PROXY=None, no_proxy=None)

    assert result.returncode == 0, result.stderr
    for call in hybrid.py():
        assert call["NO_PROXY"] == call["no_proxy"] == "127.0.0.1,localhost"


def test_cli_calls_keep_the_callers_no_proxy_entries(hybrid):
    hybrid.have_models()

    hybrid.run("hybrid", NO_PROXY="corp.example", no_proxy="lower.example")

    call = hybrid.py("activate")[0]
    assert call["NO_PROXY"] == "127.0.0.1,localhost,corp.example"
    assert call["no_proxy"] == "127.0.0.1,localhost,lower.example"


@pytest.mark.parametrize("signame", ["TERM", "INT"])
def test_interrupt_during_a_download_removes_the_part_file(hybrid, signame):
    # Arrange: the curl shim writes the .part file, then signals the manager
    expected = {"TERM": 143, "INT": 130}[signame]

    # Act
    result = hybrid.run("hybrid", CURL_SIGNAL=signame)

    # Assert
    assert result.returncode == expected
    assert [p for p in hybrid.models.rglob("*") if p.is_file()] == []
    assert not any(c.startswith("launchctl") for c in hybrid.calls())


def test_term_during_a_running_download_kills_curl_and_removes_the_part(
    hybrid, tmp_path
):
    # Arrange: the curl shim writes the .part file, records its pid, then sleeps for 30 s
    pidfile = tmp_path / "curl.pid"
    proc = subprocess.Popen(
        ["bash", str(MANAGER), "hybrid"],
        env={**hybrid.env, "CURL_HANG": str(pidfile)},
        cwd=hybrid.work,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=_default_sigint,
    )
    try:
        deadline = time.monotonic() + 15
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pidfile.exists(), "curl shim never started"
        curl_pid = int(pidfile.read_text())

        # Act
        started = time.monotonic()
        proc.terminate()
        returncode = proc.wait(timeout=10)
        elapsed = time.monotonic() - started
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.communicate()

    # Assert: prompt exit, no leftovers, no live curl
    assert returncode == 143
    assert elapsed < 5
    assert [p for p in hybrid.models.rglob("*") if p.is_file()] == []
    for _ in range(40):
        try:
            os.kill(curl_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(curl_pid, signal.SIGKILL)
        pytest.fail("the curl child was left running")


def test_allow_eval_only_flag_reaches_resolve_plist_and_activate(hybrid):
    hybrid.have_models()

    result = hybrid.run("hybrid", "accuracy", JEV_ALLOW_EVAL_ONLY="1")

    assert result.returncode == 0, result.stderr
    calls = hybrid.py()
    assert [c["sub"] for c in calls] == ["resolve", "plist", "plist", "activate"]
    assert all("--allow-eval-only" in c["args"] for c in calls)


@pytest.mark.parametrize("value", [None, "0", "", "true"])
def test_allow_eval_only_flag_is_absent_unless_the_variable_is_1(hybrid, value):
    hybrid.have_models()
    env = {} if value is None else {"JEV_ALLOW_EVAL_ONLY": value}

    result = hybrid.run("hybrid", **env)

    assert result.returncode == 0, result.stderr
    assert not any("--allow-eval-only" in c["args"] for c in hybrid.py())


@pytest.mark.parametrize("args", [(), ("0.5b",), ("7b",)])
def test_legacy_start_exits_2_and_points_to_make_start_profile(hybrid, args):
    result = hybrid.run("start", *args)

    assert result.returncode == 2
    assert "make start PROFILE=" in result.stdout + result.stderr
    assert hybrid.calls() == []


def test_plist_port_reads_a_plist_written_by_plistlib(hybrid):
    # Arrange: the layout engine_profiles.launchd_plist emits, one <string> per argument
    program = ["/opt/homebrew/bin/llama-server", "-m", "/m/x.gguf", "--host", "127.0.0.1",
              "--port", "8123", "--alias", "kev-0.8b", "--parallel", "4"]  # fmt: skip
    plist = hybrid.agents / "com.jev.mlx_server_fast.plist"
    plist.write_bytes(plistlib.dumps({"Label": "x", "ProgramArguments": program}))

    # Act
    result = hybrid.run("stop")

    # Assert
    assert result.returncode == 0, result.stderr
    kills = [c for c in hybrid.calls() if c.startswith("pkill")]
    assert len(kills) == 1 and "--port 8123" in kills[0]


def test_stop_after_hybrid_kills_the_generated_agents_ports(hybrid):
    hybrid.have_models()
    assert hybrid.run("hybrid").returncode == 0
    hybrid.log.write_text("")

    result = hybrid.run("stop")

    assert result.returncode == 0, result.stderr
    kills = [c for c in hybrid.calls() if c.startswith("pkill")]
    assert any("--port 8080" in k for k in kills) and any(
        "--port 8081" in k for k in kills
    )


# --- Makefile -----------------------------------------------------------------------------


@pytest.fixture
def make_dir(tmp_path):
    if shutil.which("make") is None:
        pytest.skip("make is not installed")
    work = tmp_path / "make_work"
    work.mkdir()
    shutil.copy(REPO / "Makefile", work / "Makefile")
    manager = work / "jev_mac_manager.sh"
    manager.write_text(f'#!/bin/bash\nprintf "<%s>" "$@" >> "{tmp_path}/argv"\n')
    manager.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k != "PROFILE"}

    def run(*args):
        subprocess.run(
            ["make", *args], cwd=work, env=env, capture_output=True, check=True
        )
        argv = tmp_path / "argv"
        return argv.read_text() if argv.exists() else ""

    return work, run


def test_make_start_passes_the_profile_through(make_dir):
    _, run = make_dir

    assert run("start", "PROFILE=lean") == "<hybrid><lean>"


def test_make_start_without_a_profile_passes_no_argument(make_dir):
    _, run = make_dir

    assert run("start") == "<hybrid>"


def test_make_start_with_an_empty_profile_passes_no_argument(make_dir):
    _, run = make_dir

    assert run("start", "PROFILE=") == "<hybrid>"


@pytest.mark.parametrize(
    "value",
    [
        "a;touch pwned",
        "$(touch pwned)",
        "$(shell touch pwned)",
        "$$(touch pwned)",
        "`touch pwned`",
        "a' ; touch pwned #",
        'a" b',
        "a,b)",
    ],
)
def test_make_start_cannot_inject_through_the_profile(make_dir, value):
    work, run = make_dir

    assert run("start", f"PROFILE={value}") == f"<hybrid><{value}>"
    assert not (work / "pwned").exists()


# --- syntax -------------------------------------------------------------------------------


def test_manager_has_valid_bash_syntax():
    subprocess.run(["bash", "-n", str(MANAGER)], check=True)


def test_manager_has_valid_syntax_for_the_system_bash_3_2():
    if not Path("/bin/bash").exists():
        pytest.skip("no /bin/bash")
    subprocess.run(["/bin/bash", "-n", str(MANAGER)], check=True)
