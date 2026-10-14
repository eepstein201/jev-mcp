"""Offline routing simulation, score dump and CLI of tests/run_evals.py (no inference)."""

import json
import random

import pytest

from tests import run_evals
from tests.run_evals import (
    ROUTING_CONFIDENCE_GATE,
    build_score_rows,
    gate_sweep,
    parse_gate_range,
    read_scores_jsonl,
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
    assert out == {"routed_accuracy": 0.5, "escalation_rate": 0.0, "n": 4}


def test_gate_one_escalates_every_unsaturated_row_so_the_smart_engine_decides():
    out = simulate_routing(ROWS, 1.0)
    # smart: 0.2 -> False (label True, wrong); 0.9 right; 0.8 -> True (label False, wrong); 0.3 right
    assert out == {"routed_accuracy": 0.5, "escalation_rate": 1.0, "n": 4}


def test_the_production_gate_escalates_only_the_low_confidence_rows():
    out = simulate_routing(ROWS, 0.85)
    # rows 2 and 4 (conf 0.60, 0.55) escalate and are right; rows 1 and 3 stay fast and are right
    assert out == {"routed_accuracy": 1.0, "escalation_rate": 0.5, "n": 4}


def test_default_gate_is_the_routing_providers_085():
    assert ROUTING_CONFIDENCE_GATE == 0.85
    assert simulate_routing(ROWS) == simulate_routing(ROWS, 0.85)


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
    assert out == {"routed_accuracy": 1.0, "escalation_rate": 0.0, "n": 2}


def test_an_all_zero_fast_answer_always_escalates_whatever_the_gate():
    # RoutingProvider: `all(v == 0.0 for v in probs.values())` is the absolute gate.
    rows = [row(0.0, 0.9, True, fast_gated=True)]
    assert simulate_routing(rows, 0.0) == {
        "routed_accuracy": 1.0,
        "escalation_rate": 1.0,
        "n": 1,
    }


def test_a_score_of_exactly_zero_without_the_gated_flag_is_a_confident_false():
    # Kev answers {true: 0.0, false: 1.0}; only an all-zero distribution escalates.
    rows = [row(0.0, 0.9, False)]
    assert simulate_routing(rows, 0.0) == {
        "routed_accuracy": 1.0,
        "escalation_rate": 0.0,
        "n": 1,
    }


def test_states_over_4000_tokens_bypass_the_fast_engine():
    over = row(0.99, 0.1, False, state_tokens=4001)
    at_limit = row(0.99, 0.1, False, state_tokens=4000)
    assert simulate_routing([over], 0.0) == {
        "routed_accuracy": 1.0,
        "escalation_rate": 1.0,
        "n": 1,
    }
    assert simulate_routing([at_limit], 0.0)["escalation_rate"] == 0.0


def test_an_empty_row_list_is_zero_not_a_division_error():
    assert simulate_routing([], 0.85) == {
        "routed_accuracy": 0.0,
        "escalation_rate": 0.0,
        "n": 0,
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


def test_build_score_rows_has_one_row_per_dataset_row_with_both_scores():
    detail = lambda s, g=False: {"score": s, "gated": g}  # noqa: E731
    rows = build_score_rows(
        DATASET,
        fast=[detail(0.9), detail(0.0, True)],
        smart=[detail(0.8), detail(0.2)],
        fast_model="kev-0.8b",
        smart_model="qwen.gguf",
        state_tokens=[3, 10],
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
        "state_tokens": 3,
        "fast_model": "kev-0.8b",
        "smart_model": "qwen.gguf",
    }
    assert rows[1]["index"] == 1
    assert rows[1]["label"] is False
    assert rows[1]["fast_gated"] is True
    assert rows[1]["smart_score"] == 0.2


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


@pytest.mark.parametrize(
    "bad", ["0.5", "0.5:0.9", "a:b:c", "0.9:0.5:0.1", "0.5:0.9:0", "0.5:0.9:-0.1"]
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
        {"gate": 0.85, "routed_accuracy": 1.0, "escalation_rate": 0.5, "n": 4}
    ]


def test_simulate_defaults_include_the_production_gate(scores_file, capsys):
    run_evals.main(["--simulate", str(scores_file), "--json"])
    gates = [p["gate"] for p in json.loads(capsys.readouterr().out)]
    assert ROUTING_CONFIDENCE_GATE in gates


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

    def __init__(self, model_id, table, gated=()):
        self.current_model_id = model_id
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


def install_fakes(monkeypatch, fast, smart):
    import jev_mcp.daemon_provider as dp
    import jev_mcp.kev_provider as kp

    monkeypatch.setattr(kp, "KevProvider", lambda url: fast)
    monkeypatch.setattr(dp, "DaemonProvider", lambda url: smart)
    monkeypatch.setattr(run_evals, "WARMUP_PASSES", 1)


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
    run_evals.main(["--seed", "7"])
    after_first = random.random()
    run_evals.main(["--seed", "7"])
    # main seeds, then (fakes) consume no randomness: the next draw equals a fresh seed's.
    random.seed(7)
    assert after_first == random.random()


def test_all_zero_scores_warn_that_the_daemon_is_probably_down(
    tiny_dataset, monkeypatch, capsys
):
    fast = FakeProvider("kev-0.8b", {}, gated=["alpha", "beta"])
    smart = FakeProvider("m", {"alpha": 0.7, "beta": 0.1})
    install_fakes(monkeypatch, fast, smart)
    run_evals.main([])
    assert "daemon is probably not running" in capsys.readouterr().out
