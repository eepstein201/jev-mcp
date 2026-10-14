"""Offline routing simulation, score dump and CLI of tests/run_evals.py (no inference)."""

import json
import random
import urllib.error
import urllib.request

import pytest

from jev_mcp import engine_profiles
from jev_mcp.engine_profiles import DEFAULT_GATE
from tests import run_evals
from tests.run_evals import (
    build_score_rows,
    gate_sweep,
    parse_gate_range,
    read_scores_jsonl,
    route_row,
    simulate_routing,
    write_scores_jsonl,
)


def row(fast, smart, label, **extra):
    return {"fast_score": fast, "smart_score": smart, "label": label, **extra}


# Hand-computed: confidence is max(p, 1 - p); the fast answer is kept unless
# confidence < gate, in which case the smart score decides.
ROWS = [
    row(
        0.95, 0.2, True
    ),  # conf 0.95: always stays fast, fast right (smart would be wrong)
    row(0.4, 0.9, True),  # conf 0.60: fast wrong, smart right
    row(0.1, 0.8, False),  # conf 0.90: stays fast, fast right (smart would be wrong)
    row(0.55, 0.3, False),  # conf 0.55: fast wrong, smart right
]


def test_gate_zero_never_escalates_so_the_fast_engine_decides_alone():
    out = simulate_routing(ROWS, 0.0)
    assert out == {
        "routed_accuracy": 0.5,
        "escalation_rate": 0.0,
        "n": 4,
        "unanswered": 0,
    }


def test_gate_one_escalates_every_unsaturated_row_so_the_smart_engine_decides():
    out = simulate_routing(ROWS, 1.0)
    # smart: 0.2 -> False (label True, wrong); 0.9 right; 0.8 -> True (label False, wrong); 0.3 right
    assert out == {
        "routed_accuracy": 0.5,
        "escalation_rate": 1.0,
        "n": 4,
        "unanswered": 0,
    }


def test_the_production_gate_escalates_only_the_low_confidence_rows():
    out = simulate_routing(ROWS, 0.85)
    # rows 2 and 4 (conf 0.60, 0.55) escalate and are right; rows 1 and 3 stay fast and are right
    assert out == {
        "routed_accuracy": 1.0,
        "escalation_rate": 0.5,
        "n": 4,
        "unanswered": 0,
    }


def test_default_gate_is_the_routing_providers_085():
    assert simulate_routing(ROWS, DEFAULT_GATE) == simulate_routing(ROWS, 0.85)


# --- the default gate follows the active profile -------------------------------------------

# A real (fixture) profile table: `gated` carries a gate that is neither DEFAULT_GATE
# nor on the default sweep's 0.05 grid, so a test can tell "read the profile" from
# "fell back to 0.85" and from "the gate landed on the grid by luck".
GATED_TABLE = """
[models."kev-0.8b"]
tier = "fast"
repo = "ggml-org/Kev-0.8B-GGUF"
revision = "e551e319d483ff57e1ff208924b349d397cffc1c"
filename = "Kev-0.8B-Q8_0.gguf"
sha256 = "27278f34eb3273bceea4c053dc50dd61a5161da21a718c4aacdf8fd5830771d0"
parallel = 4
ctx_per_slot = 16384
reasoning = "auto"
style = "plain"

[models."qwen2.5-7b"]
tier = "smart"
repo = "bartowski/Qwen2.5-7B-Instruct-GGUF"
revision = "8911e8a47f92bac19d6f5c64a2e2095bd2f7d031"
filename = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
sha256 = "65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423"
parallel = 2
ctx_per_slot = 16384
reasoning = "auto"
style = "sandbox"

[profiles.default]
fast = "kev-0.8b"
smart = "qwen2.5-7b"

[profiles.gated]
fast = "kev-0.8b"
smart = "qwen2.5-7b"
gate = 0.72
"""

# Confidence 0.75 sits between the two gates: kept fast at 0.72, escalated at 0.85.
GATE_ROWS = [row(0.75, 0.2, True), row(0.95, 0.2, True)]


@pytest.fixture
def active_gate_table(monkeypatch, tmp_path):
    """Point gate resolution at a real marker+TOML pair (offline: marker and TOML reads).

    point_at("gated") marks the 0.72-gate profile active; point_at(None) leaves the
    marker missing, so the table's gate-less `default` profile resolves. config_path
    swaps in a table that cannot be read at all."""
    config = tmp_path / "engine_profiles.toml"
    config.write_text(GATED_TABLE)
    marker = tmp_path / "active_profile"
    real_active = engine_profiles.active_profile

    def point_at(name, config_path=str(config)):
        if name is None:
            marker.unlink(missing_ok=True)
        else:
            marker.write_text(f"{name}\n")
        monkeypatch.setattr(
            engine_profiles,
            "active_profile",
            lambda *a, **k: real_active(
                config_path=config_path, marker_path=str(marker)
            ),
        )

    return point_at


def test_the_default_gate_is_the_active_profiles_gate(active_gate_table):
    active_gate_table("gated")

    assert run_evals.active_gate() == 0.72
    assert simulate_routing(GATE_ROWS) == simulate_routing(GATE_ROWS, 0.72)
    assert simulate_routing(GATE_ROWS) != simulate_routing(GATE_ROWS, 0.85)
    assert route_row(GATE_ROWS[0])["escalated"] is False


def test_the_gate_boundary_is_strict_at_the_profiles_gate(active_gate_table):
    active_gate_table("gated")

    at_gate = row(0.72, 0.9, True, fast_confidence=0.72)
    below = row(0.72, 0.9, True, fast_confidence=0.7199)
    assert route_row(at_gate)["escalated"] is False
    assert route_row(below)["escalated"] is True


def test_without_a_marker_the_default_gate_resolves(active_gate_table):
    # A missing marker names the table's `default` profile, which sets no gate.
    active_gate_table(None)

    assert run_evals.active_gate() == DEFAULT_GATE == 0.85
    assert simulate_routing(ROWS) == simulate_routing(ROWS, 0.85)


def test_when_no_profile_resolves_the_default_gate_answers(active_gate_table):
    active_gate_table(None, config_path="/nonexistent/engine_profiles.toml")

    assert run_evals.active_gate() == DEFAULT_GATE


def test_an_explicit_gate_still_overrides_the_active_profile(active_gate_table):
    active_gate_table("gated")

    at_085 = simulate_routing(GATE_ROWS, 0.85)
    assert at_085 != simulate_routing(GATE_ROWS)
    # only the conf-0.75 row escalates at 0.85; the conf-0.95 one never does
    assert at_085["escalation_rate"] == 0.5


def test_confidence_exactly_at_the_gate_is_not_escalated():
    # RoutingProvider escalates on `max_prob < 0.85`, strictly.
    rows = [row(0.85, 0.0, True)]
    assert simulate_routing(rows, 0.85)["escalation_rate"] == 0.0
    assert simulate_routing(rows, 0.8500001)["escalation_rate"] == 1.0


def test_a_false_leaning_score_has_the_same_confidence_as_its_mirror():
    # p=0.1 means P(false)=0.9, so it is just as confident as p=0.9.
    assert simulate_routing([row(0.1, 0.5, False)], 0.85)["escalation_rate"] == 0.0
    assert simulate_routing([row(0.9, 0.5, True)], 0.85)["escalation_rate"] == 0.0
    assert simulate_routing([row(0.2, 0.5, False)], 0.85)["escalation_rate"] == 1.0


def test_saturated_scores_stay_fast_even_at_gate_one():
    # max(1.0, 0.0) = 1.0 is not < 1.0, exactly as in RoutingProvider.
    out = simulate_routing([row(1.0, 0.0, True), row(0.0, 1.0, False)], 1.0)
    assert out == {
        "routed_accuracy": 1.0,
        "escalation_rate": 0.0,
        "n": 2,
        "unanswered": 0,
    }


def test_an_all_zero_fast_answer_always_escalates_whatever_the_gate():
    # RoutingProvider: `all(v == 0.0 for v in probs.values())` is the absolute gate.
    rows = [row(0.0, 0.9, True, fast_gated=True)]
    assert simulate_routing(rows, 0.0) == {
        "routed_accuracy": 1.0,
        "escalation_rate": 1.0,
        "n": 1,
        "unanswered": 0,
    }


def test_a_score_of_exactly_zero_without_the_gated_flag_is_a_confident_false():
    # Kev answers {true: 0.0, false: 1.0}; only an all-zero distribution escalates.
    rows = [row(0.0, 0.9, False)]
    assert simulate_routing(rows, 0.0) == {
        "routed_accuracy": 1.0,
        "escalation_rate": 0.0,
        "n": 1,
        "unanswered": 0,
    }


def test_states_over_4000_tokens_bypass_the_fast_engine():
    over = row(0.99, 0.1, False, state_tokens=4001)
    at_limit = row(0.99, 0.1, False, state_tokens=4000)
    assert simulate_routing([over], 0.0) == {
        "routed_accuracy": 1.0,
        "escalation_rate": 1.0,
        "n": 1,
        "unanswered": 0,
    }
    assert simulate_routing([at_limit], 0.0)["escalation_rate"] == 0.0


def test_an_empty_row_list_is_zero_not_a_division_error():
    assert simulate_routing([], 0.85) == {
        "routed_accuracy": 0.0,
        "escalation_rate": 0.0,
        "n": 0,
        "unanswered": 0,
    }


def test_sweep_returns_one_point_per_gate_in_order():
    sweep = gate_sweep(ROWS, [0.0, 0.85, 1.0])
    assert [p["gate"] for p in sweep] == [0.0, 0.85, 1.0]
    assert [p["escalation_rate"] for p in sweep] == [0.0, 0.5, 1.0]
    assert [p["routed_accuracy"] for p in sweep] == [0.5, 1.0, 0.5]
    assert all(p["n"] == 4 for p in sweep)


def test_simulation_does_not_mutate_its_rows():
    rows = [dict(r) for r in ROWS]
    simulate_routing(rows, 0.85)
    assert rows == ROWS


# --- score dump -------------------------------------------------------------


DATASET = [
    {
        "id": "a",
        "group_id": "g1",
        "kind": "orig",
        "question": "Q1?",
        "state": "state one",
        "expected": True,
    },
    {
        "id": "b",
        "group_id": "g1",
        "kind": "reverse",
        "question_override": "Q2?",
        "state": "x" * 40,
        "expected": False,
    },
]


def detail(score, gated=False, answered=True, confidence=None, present=None):
    return {
        "score": score,
        "gated": gated,
        "answered": answered,
        "present": answered if present is None else present,
        "confidence": max(score, 1 - score) if confidence is None else confidence,
    }


def test_build_score_rows_has_one_row_per_dataset_row_with_both_scores():
    rows = build_score_rows(
        DATASET,
        fast=[detail(0.9), detail(0.0, gated=True, confidence=0.0)],
        smart=[detail(0.8), detail(0.0, answered=False, confidence=None)],
        fast_model="kev-0.8b",
        smart_model="qwen.gguf",
        state_tokens=[3, 10],
        fast_model_source="provider",
        smart_model_source="props",
    )
    assert rows[0] == {
        "index": 0,
        "id": "a",
        "group_id": "g1",
        "kind": "orig",
        "question_key": "eval",
        "label": True,
        "fast_score": 0.9,
        "smart_score": 0.8,
        "fast_gated": False,
        "fast_answered": True,
        "fast_confidence": 0.9,
        "smart_answered": True,
        "smart_returned_key": True,
        "state_tokens": 3,
        "fast_model": "kev-0.8b",
        "smart_model": "qwen.gguf",
        "fast_model_source": "provider",
        "smart_model_source": "props",
    }
    assert rows[1]["index"] == 1
    assert rows[1]["label"] is False
    assert rows[1]["fast_gated"] is True
    assert rows[1]["smart_answered"] is False
    assert rows[1]["smart_returned_key"] is False


def test_score_dataset_detailed_records_answered_and_the_routers_confidence():
    outs = iter(
        [
            {"eval": {"probabilities": {"true": 0.3, "false": 0.5}}},
            {"eval": {"probabilities": {}}},
            {},
            {"eval": {"probabilities": {"true": 0.0, "false": 0.0}}},
        ]
    )

    class P:
        def evaluate_batch(self, state, questions):
            return next(outs)

    ds = [{"state": "s", "question": "Q?", "expected": True}] * 4
    got = run_evals.score_dataset_detailed(P(), ds)

    def d(score, gated, answered, present, confidence):
        return {
            "score": score,
            "gated": gated,
            "answered": answered,
            "present": present,
            "confidence": confidence,
        }

    assert got[0] == d(0.3, False, True, True, 0.5)
    assert got[1] == d(0.0, True, False, True, None)  # key present, empty map
    assert got[2] == d(0.0, True, False, False, None)  # no key at all
    assert got[3] == d(0.0, True, True, True, 0.0)  # all-zero map is an answer


def test_jsonl_round_trips_and_is_one_object_per_line(tmp_path):
    rows = [row(0.9, 0.1, True, index=0), row(0.2, 0.7, False, index=1)]
    path = tmp_path / "scores.jsonl"
    write_scores_jsonl(path, rows)
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["fast_score"] == 0.9
    assert read_scores_jsonl(path) == rows


def test_reading_skips_blank_lines(tmp_path):
    path = tmp_path / "scores.jsonl"
    path.write_text(json.dumps(row(0.9, 0.1, True)) + "\n\n")
    assert len(read_scores_jsonl(path)) == 1


# --- gate range parser ------------------------------------------------------


def test_gate_range_is_inclusive_and_free_of_float_drift():
    assert parse_gate_range("0.5:0.9:0.1") == [0.5, 0.6, 0.7, 0.8, 0.9]


def test_gate_range_of_a_single_point():
    assert parse_gate_range("0.85:0.85:0.05") == [0.85]


def test_gate_range_never_overshoots_stop():
    # round() would give [0.5, 0.8, 1.1]; the last gate must stay <= STOP
    assert parse_gate_range("0.5:1.0:0.3") == [0.5, 0.8]


@pytest.mark.parametrize(
    "bad",
    [
        "0.5",
        "0.5:0.9",
        "a:b:c",
        "0.9:0.5:0.1",
        "0.5:0.9:0",
        "0.5:0.9:-0.1",
        "0:inf:0.1",
        "0:1:inf",
        "nan:1:0.1",
        "-inf:1:0.1",
    ],
)
def test_malformed_gate_ranges_are_rejected(bad):
    with pytest.raises(ValueError):
        parse_gate_range(bad)


# --- CLI --------------------------------------------------------------------


@pytest.fixture
def scores_file(tmp_path):
    path = tmp_path / "scores.jsonl"
    write_scores_jsonl(path, ROWS)
    return path


def test_simulate_prints_a_table_with_one_line_per_gate(scores_file, capsys):
    run_evals.main(["--simulate", str(scores_file), "--gates", "0:1:0.5"])
    out = capsys.readouterr().out
    assert "gate" in out and "routed accuracy" in out and "escalation" in out
    lines = [ln for ln in out.splitlines() if ln.strip().startswith(("0.", "1."))]
    assert len(lines) == 3
    assert "50.0%" in lines[0] and "0.0%" in lines[0]
    assert "100.0%" in lines[-1]


def test_simulate_json_is_machine_readable(scores_file, capsys):
    run_evals.main(
        ["--simulate", str(scores_file), "--gates", "0.85:0.85:0.05", "--json"]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload == [
        {
            "gate": 0.85,
            "routed_accuracy": 1.0,
            "escalation_rate": 0.5,
            "n": 4,
            "unanswered": 0,
        }
    ]


def test_simulate_defaults_include_the_production_gate(
    scores_file, capsys, active_gate_table
):
    active_gate_table("gated")

    run_evals.main(["--simulate", str(scores_file), "--json"])
    gates = [p["gate"] for p in json.loads(capsys.readouterr().out)]
    assert 0.72 in gates  # off the 0.05 grid: only following the profile puts it there
    assert 0.85 in gates  # the standard grid itself is unchanged


def test_a_bad_gate_range_exits_with_a_usage_error(scores_file):
    with pytest.raises(SystemExit):
        run_evals.main(["--simulate", str(scores_file), "--gates", "nope"])


def test_simulating_an_empty_file_exits_with_a_message(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(SystemExit, match="no rows"):
        run_evals.main(["--simulate", str(empty)])


def test_the_legacy_scores_mode_still_works(tmp_path, capsys):
    dataset = json.loads(run_evals.DATASET_PATH.read_text())
    by_id = {r["id"]: (0.9 if r["expected"] else 0.1) for r in dataset}
    path = tmp_path / "by_id.json"
    path.write_text(json.dumps(by_id))
    run_evals.main(["--scores", str(path), "oracle"])
    assert "--- oracle ---" in capsys.readouterr().out


def test_legacy_scores_with_missing_ids_exit(tmp_path):
    path = tmp_path / "by_id.json"
    path.write_text("{}")
    with pytest.raises(SystemExit, match="missing"):
        run_evals.main(["--scores", str(path), "x"])


# --- live path with fake providers -----------------------------------------


class FakeProvider:
    """Answers by the row's state text; mimics both providers' result shape."""

    def __init__(self, model_id, table, gated=(), base_url=None):
        self.current_model_id = model_id
        self.max_tokens = 8192
        self.base_url = base_url
        self.table = table
        self.gated = set(gated)
        self.calls = 0

    def check_token_limit(self, state):
        return len(state) // 4

    def evaluate_batch(self, state, questions):
        self.calls += 1
        if state in self.gated:
            return {"eval": {"probabilities": {"true": 0.0, "false": 0.0}}}
        p = self.table[state]
        return {"eval": {"probabilities": {"true": p, "false": 1.0 - p}}}


@pytest.fixture
def tiny_dataset(monkeypatch, tmp_path):
    data = [
        {
            "id": "a",
            "group_id": "g",
            "kind": "orig",
            "question": "Q?",
            "state": "alpha",
            "expected": True,
        },
        {
            "id": "b",
            "group_id": "g",
            "kind": "reverse",
            "question": "Q?",
            "state": "beta",
            "expected": False,
        },
    ]
    path = tmp_path / "ds.json"
    path.write_text(json.dumps(data))
    monkeypatch.setattr(run_evals, "DATASET_PATH", path)
    return data


def install_fakes(monkeypatch, fast, smart, fast_engine=None):
    """Live runs build their providers through RoutingProvider(); patch what it constructs."""
    import jev_mcp.routing_provider as rp

    built = []

    def kev(base_url):
        built.append(("kev", base_url))
        return fast

    def daemon(base_url):
        built.append(("daemon", base_url))
        return fast if len(built) == 1 and fast_engine == "daemon" else smart

    monkeypatch.setattr(rp, "KevProvider", kev)
    monkeypatch.setattr(rp, "DaemonProvider", daemon)
    if fast_engine:
        monkeypatch.setenv("JEV_FAST_ENGINE", fast_engine)
    else:
        monkeypatch.delenv("JEV_FAST_ENGINE", raising=False)
    monkeypatch.setattr(run_evals, "WARMUP_PASSES", 1)
    return built


def test_live_run_dumps_one_row_per_dataset_row_with_served_model_ids(
    tiny_dataset, tmp_path, monkeypatch, capsys
):
    fast = FakeProvider("kev-0.8b", {"alpha": 0.9, "beta": 0.2}, gated=["beta"])
    smart = FakeProvider("qwen-served.gguf", {"alpha": 0.7, "beta": 0.1})
    install_fakes(monkeypatch, fast, smart)
    out = tmp_path / "scores.jsonl"

    run_evals.main(["--scores-out", str(out)])

    rows = read_scores_jsonl(out)
    assert [r["id"] for r in rows] == ["a", "b"]
    assert [r["fast_score"] for r in rows] == [0.9, 0.0]
    assert [r["fast_gated"] for r in rows] == [False, True]
    assert [r["smart_score"] for r in rows] == [0.7, 0.1]
    assert {r["fast_model"] for r in rows} == {"kev-0.8b"}
    assert {r["smart_model"] for r in rows} == {"qwen-served.gguf"}
    assert [r["label"] for r in rows] == [True, False]
    assert rows[0]["state_tokens"] == len("alpha") // 4
    # the report names engines by served id, not by a hard-coded string
    report = capsys.readouterr().out
    assert "qwen-served.gguf" in report and "Qwen-2.5-7B" not in report


def test_each_dataset_row_is_scored_once_per_warmup_plus_timed_pass(
    tiny_dataset, monkeypatch
):
    fast = FakeProvider("kev-0.8b", {"alpha": 0.9, "beta": 0.2})
    smart = FakeProvider("m", {"alpha": 0.7, "beta": 0.1})
    install_fakes(monkeypatch, fast, smart)
    run_evals.main([])
    assert fast.calls == 4 and smart.calls == 4  # 2 rows x (1 warm-up + 1 timed)


def test_an_unreported_model_id_falls_back_to_the_role_name(
    tiny_dataset, monkeypatch, capsys
):
    fast = FakeProvider("kev-0.8b", {"alpha": 0.9, "beta": 0.2})
    smart = FakeProvider("unknown_model", {"alpha": 0.7, "beta": 0.1})
    install_fakes(monkeypatch, fast, smart)
    run_evals.main([])
    assert run_evals.served_name(smart, "smart") == "smart"
    assert "=== Benchmarking smart ===" in capsys.readouterr().out


def test_seed_makes_the_global_random_stream_reproducible(tiny_dataset, monkeypatch):
    fast = FakeProvider("kev-0.8b", {"alpha": 0.9, "beta": 0.2})
    smart = FakeProvider("m", {"alpha": 0.7, "beta": 0.1})
    install_fakes(monkeypatch, fast, smart)
    random.seed(999)  # something else, so a no-op would be visible
    run_evals.main(["--seed", "7"])
    drawn = random.random()
    random.seed(7)
    # the fakes consume no randomness, so the next draw is the first of seed 7
    assert drawn == random.random()


def test_seed_is_accepted_together_with_simulate(scores_file, capsys):
    random.seed(999)
    run_evals.main(["--simulate", str(scores_file), "--seed", "7", "--json"])
    drawn = random.random()
    random.seed(7)
    assert drawn == random.random()
    assert json.loads(capsys.readouterr().out)[0]["n"] == 4


def test_all_zero_scores_warn_that_the_daemon_is_probably_down(
    tiny_dataset, monkeypatch, capsys
):
    fast = FakeProvider("kev-0.8b", {}, gated=["alpha", "beta"])
    smart = FakeProvider("m", {"alpha": 0.7, "beta": 0.1})
    install_fakes(monkeypatch, fast, smart)
    run_evals.main([])
    assert "daemon is probably not running" in capsys.readouterr().out


# --- unanswered engines, exact confidence -----------------------------------


def test_an_escalated_row_the_smart_engine_did_not_answer_keeps_the_fast_answer():
    # RoutingProvider: {**fast_results, **escalated} with no smart key leaves fast.
    r = row(
        0.6,
        0.0,
        True,
        fast_confidence=0.6,
        smart_answered=False,
        smart_returned_key=False,
    )
    assert route_row(r, 0.85) == {"escalated": True, "score": 0.6, "unanswered": False}
    assert simulate_routing([r], 0.85) == {
        "routed_accuracy": 1.0,
        "escalation_rate": 1.0,
        "n": 1,
        "unanswered": 0,
    }


def test_a_smart_key_with_an_empty_map_keeps_the_fast_answer():
    # An empty-probability smart entry is no answer: the router keeps the fast answer
    # (marked escalation_failed) instead of letting the empty entry replace it and raise.
    r = row(
        0.6,
        0.0,
        True,
        fast_confidence=0.6,
        smart_answered=False,
        smart_returned_key=True,
    )
    assert route_row(r, 0.85) == {"escalated": True, "score": 0.6, "unanswered": False}


def test_a_smart_zero_that_was_a_real_answer_is_not_mistaken_for_silence():
    r = row(0.6, 0.0, False, fast_confidence=0.6, smart_answered=True)
    assert route_row(r, 0.85)["score"] == 0.0
    assert simulate_routing([r], 0.85)["routed_accuracy"] == 1.0


def test_when_neither_engine_answered_the_router_raises_so_the_row_is_unanswered():
    r = row(0.0, 0.0, True, fast_gated=True, fast_answered=False, smart_answered=False)
    assert route_row(r, 0.85)["unanswered"] is True
    assert simulate_routing([r], 0.85) == {
        "routed_accuracy": 0.0,
        "escalation_rate": 1.0,
        "n": 1,
        "unanswered": 1,
    }


def test_an_oversized_state_with_a_silent_smart_engine_is_unanswered_not_fast():
    # The >4000 bypass returns _require_answers(smart only); fast is never consulted.
    r = row(
        0.99, 0.0, True, fast_confidence=0.99, state_tokens=4001, smart_answered=False
    )
    assert route_row(r, 0.85) == {"escalated": True, "score": None, "unanswered": True}


def test_an_all_zero_fast_answer_with_a_silent_smart_engine_raises_like_an_empty_one():
    r = row(0.0, 0.0, False, fast_gated=True, smart_answered=False)
    assert route_row(r, 0.85) == {"escalated": True, "score": None, "unanswered": True}


def test_the_recorded_confidence_is_used_over_one_minus_p():
    # Router saw {true: 0.1500000000000001, false: 0.85}: top probability 0.85, not escalated.
    # 1 - p is 0.8499999999999999, which would wrongly escalate.
    p = 0.1500000000000001
    exact = row(p, 0.9, False, fast_confidence=0.85)
    assert route_row(exact, 0.85)["escalated"] is False
    assert route_row(row(p, 0.9, False), 0.85)["escalated"] is True  # old-dump fallback


def test_old_dumps_without_the_new_fields_fall_back_and_are_flagged():
    old = [row(0.9, 0.1, True)]
    assert run_evals.dump_warnings(old)
    new = [row(0.9, 0.1, True, fast_confidence=0.9, smart_answered=True)]
    assert run_evals.dump_warnings(new) == []


def test_simulate_prints_the_old_dump_warning_to_stderr(scores_file, capsys):
    run_evals.main(["--simulate", str(scores_file), "--json"])
    err = capsys.readouterr().err
    assert "smart_answered" in err and "fast_confidence" in err


# --- differential: the real RoutingProvider vs the simulation ---------------

NOKEY = "no key"
F_ID, S_ID = "fast-id", "smart-id"


class StubEngine:
    def __init__(self, model_id, probs, tokens=0):
        self.current_model_id = model_id
        self.max_tokens = 8192
        self.probs = probs
        self.tokens = tokens

    def check_token_limit(self, state):
        return self.tokens

    def evaluate_batch(self, state, questions):
        if self.probs is NOKEY:
            return {}
        return {q.key: {"probabilities": dict(self.probs)} for q in questions}


def tf(true, false):
    return {"true": true, "false": false}


FAST_CASES = [
    tf(0.85, 0.15),  # exactly at the gate
    tf(0.8499, 0.1501),  # just below
    tf(0.8501, 0.1499),  # just above
    tf(0.15, 0.85),  # confident false, exactly at the gate
    tf(0.1500000000000001, 0.85),  # 1 - p is 1 ulp under the gate, top prob is not
    tf(0.0, 0.0),  # all-zero gate
    {},  # empty map
    NOKEY,  # engine returned no key
    tf(0.3, 0.5),  # daemon-class: does not sum to 1
    tf(1.0, 0.0),  # saturated
]
SMART_CASES = [tf(0.9, 0.1), tf(0.2, 0.8), tf(0.0, 0.0), NOKEY, {}]
TOKEN_CASES = [4000, 4001]


@pytest.mark.parametrize("tokens", TOKEN_CASES)
@pytest.mark.parametrize("smart_probs", SMART_CASES, ids=lambda c: str(c)[:22])
@pytest.mark.parametrize("fast_probs", FAST_CASES, ids=lambda c: str(c)[:22])
def test_simulation_agrees_with_the_real_router_cell_by_cell(
    monkeypatch, active_gate_table, fast_probs, smart_probs, tokens
):
    import jev_mcp.routing_provider as rp
    from jev_mcp.provider import EngineUnavailableError, NoulQuestion

    # A non-default gate, resolved the same way on both sides: the router reads it
    # through daemon_provider, the simulation through run_evals.active_gate().
    active_gate_table("gated")
    fast = StubEngine(F_ID, fast_probs, tokens)
    smart = StubEngine(S_ID, smart_probs)
    monkeypatch.setattr(rp, "KevProvider", lambda base_url: fast)
    monkeypatch.setattr(rp, "DaemonProvider", lambda base_url: smart)
    monkeypatch.delenv("JEV_FAST_ENGINE", raising=False)
    router = rp.RoutingProvider()

    q = NoulQuestion(key="eval", prompt="Q?")
    try:
        res = router.evaluate_batch("state", [q])
        actual = {
            "escalated": router.current_model_id != F_ID,
            "score": res["eval"]["probabilities"].get("true", 0.0),
            "unanswered": False,
        }
    except EngineUnavailableError:
        actual = {"escalated": True, "score": None, "unanswered": True}

    ds = [{"state": "state", "question": "Q?", "expected": True}]
    rows = build_score_rows(
        ds,
        run_evals.score_dataset_detailed(fast, ds),
        run_evals.score_dataset_detailed(smart, ds),
        F_ID,
        S_ID,
        [router.check_token_limit("state")],
    )
    simulated = route_row(rows[0])  # no explicit gate: resolves as the router did

    assert simulated == actual
    # the gate the simulation applied is the one the router really read: the profile's
    assert run_evals.active_gate() == 0.72


# --- dump reading failures --------------------------------------------------


def test_a_malformed_line_names_the_file_and_line(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text(json.dumps(row(0.9, 0.1, True)) + "\n{not json\n")
    with pytest.raises(ValueError, match=r"s\.jsonl:2"):
        read_scores_jsonl(path)


def test_a_row_missing_a_required_key_names_the_key(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text(json.dumps({"fast_score": 0.9, "label": True}) + "\n")
    with pytest.raises(ValueError, match="smart_score"):
        read_scores_jsonl(path)


def test_a_non_object_line_is_rejected(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text("[1, 2]\n")
    with pytest.raises(ValueError, match="object"):
        read_scores_jsonl(path)


def test_simulate_exits_cleanly_on_a_malformed_file(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text("{oops\n")
    with pytest.raises(SystemExit) as exc:
        run_evals.main(["--simulate", str(path)])
    assert "s.jsonl:1" in str(exc.value)


def test_simulate_exits_cleanly_on_a_missing_file(tmp_path):
    with pytest.raises(SystemExit) as exc:
        run_evals.main(["--simulate", str(tmp_path / "nope.jsonl")])
    assert "nope.jsonl" in str(exc.value) and exc.value.code not in (0, None)


# --- live path: RoutingProvider-built tiers, served model id from /props ----


class FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_live_run_builds_both_tiers_through_routing_provider(tiny_dataset, monkeypatch):
    fast = FakeProvider("f", {"alpha": 0.9, "beta": 0.2})
    smart = FakeProvider("s", {"alpha": 0.7, "beta": 0.1})
    built = install_fakes(monkeypatch, fast, smart)
    monkeypatch.setenv("JEV_FAST_PORT", "9100")
    monkeypatch.setenv("JEV_SMART_PORT", "9101")
    run_evals.main([])
    assert built[0] == ("kev", "http://127.0.0.1:9100/v1/systemone")
    assert built[1] == ("daemon", "http://127.0.0.1:9101/v1/chat/completions")


def test_live_run_honours_jev_fast_engine_daemon(tiny_dataset, monkeypatch):
    fast = FakeProvider("f", {"alpha": 0.9, "beta": 0.2})
    smart = FakeProvider("s", {"alpha": 0.7, "beta": 0.1})
    built = install_fakes(monkeypatch, fast, smart, fast_engine="daemon")
    run_evals.main([])
    assert [kind for kind, _ in built] == ["daemon", "daemon"]
    assert fast.calls == 4  # the configured fast tier was the one benchmarked


def test_served_model_id_comes_from_props_model_path_basename(
    tiny_dataset, tmp_path, monkeypatch
):
    fast = FakeProvider(
        "kev-0.8b",
        {"alpha": 0.9, "beta": 0.2},
        base_url="http://127.0.0.1:8080/v1/systemone",
    )
    smart = FakeProvider(
        "unknown_model",
        {"alpha": 0.7, "beta": 0.1},
        base_url="http://127.0.0.1:8081/v1/chat/completions",
    )
    install_fakes(monkeypatch, fast, smart)
    urls = []

    def urlopen(url, timeout=None):
        urls.append((url, timeout))
        name = "Kev-0.8B-Q8_0.gguf" if ":8080/" in url else "Qwen2.5-7B-Q4_K_M.gguf"
        return FakeResponse({"model_path": f"/some/dir/{name}"})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    out = tmp_path / "s.jsonl"
    run_evals.main(["--scores-out", str(out)])

    rows = read_scores_jsonl(out)
    assert {r["fast_model"] for r in rows} == {"Kev-0.8B-Q8_0.gguf"}
    assert {r["smart_model"] for r in rows} == {"Qwen2.5-7B-Q4_K_M.gguf"}
    assert {r["fast_model_source"] for r in rows} == {"props"}
    assert sorted(u for u, _ in urls) == [
        "http://127.0.0.1:8080/props",
        "http://127.0.0.1:8081/props",
    ]
    assert all(t is not None and t <= 5 for _, t in urls)


def test_props_failure_falls_back_to_the_provider_id_and_says_so(
    tiny_dataset, tmp_path, monkeypatch, capsys
):
    fast = FakeProvider(
        "kev-0.8b", {"alpha": 0.9, "beta": 0.2}, base_url="http://127.0.0.1:8080/v1/x"
    )
    smart = FakeProvider("unknown_model", {"alpha": 0.7, "beta": 0.1}, base_url=None)
    install_fakes(monkeypatch, fast, smart)

    def urlopen(url, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    out = tmp_path / "s.jsonl"
    run_evals.main(["--scores-out", str(out)])

    rows = read_scores_jsonl(out)
    assert {r["fast_model"] for r in rows} == {"kev-0.8b"}
    assert {r["fast_model_source"] for r in rows} == {"provider"}
    assert {r["smart_model"] for r in rows} == {"smart"}
    assert {r["smart_model_source"] for r in rows} == {"role"}
    assert "falling back" in capsys.readouterr().out


def test_props_is_only_asked_of_loopback_hosts(monkeypatch):
    called = []
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: called.append(a) or FakeResponse({})
    )
    remote = FakeProvider("m", {}, base_url="http://example.com:8080/v1/x")
    assert run_evals.fetch_served_model(remote, "fast") == ("m", "provider")
    assert called == []
