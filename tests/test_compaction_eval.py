"""The compaction evaluation harness, with the engine replaced by a scoring function."""

import json

import pytest

from tests.evals import compaction_eval as ev


def _line(**overrides):
    row = {
        "id": "r1",
        "goal_type": "open_items",
        "split": "tune",
        "goal": "what is open",
        "state": {"a": "alpha", "b": "beta", "c": "gamma"},
        "relevant": ["a"],
        "note": "",
    }
    return json.dumps({**row, **overrides})


def _rows(*lines, tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text("\n".join(lines) + "\n\n")
    return ev.load_dataset(path)


def test_load_dataset_reads_rows_and_chunks(tmp_path):
    (row,) = _rows(_line(), tmp_path=tmp_path)
    assert row.keys == ("a", "b", "c")
    assert row.chunks[0] == "a: alpha"
    assert row.relevant == frozenset({"a"})


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"goal_type": "poetry"}, "goal_type"),
        ({"split": "dev"}, "split"),
        ({"relevant": ["zzz"]}, "unknown keys"),
        ({"state": {"a": "x" * 2500}, "relevant": ["a"]}, "longer than one chunk"),
    ],
)
def test_load_dataset_rejects_bad_rows(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        _rows(_line(**overrides), tmp_path=tmp_path)


def test_load_dataset_rejects_missing_keys(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text(json.dumps({"id": "x"}) + "\n")
    with pytest.raises(ValueError, match="missing"):
        ev.load_dataset(path)


def test_shipped_dataset_loads():
    rows = ev.load_dataset(ev.DEFAULT_DATASET)
    assert {r.split for r in rows} == set(ev.SPLITS)
    assert {r.goal_type for r in rows} == set(ev.GOAL_TYPES)


def test_counts_metrics_and_empty_edges():
    counts = ev.Counts(relevant=2, relevant_kept=1, kept=2, total=4)
    assert counts.as_dict() == {
        "recall": 0.5,
        "precision": 0.5,
        "compression": 0.5,
        "chunks": 4,
    }
    empty = ev.Counts()
    assert (empty.recall, empty.precision, empty.compression) == (1.0, 1.0, 0.0)


def test_thresholds_span_005_to_070():
    assert ev.THRESHOLDS[0] == 0.05 and ev.THRESHOLDS[-1] == 0.7
    assert len(ev.THRESHOLDS) == 14


def test_choose_threshold_ties_go_to_the_higher_value(tmp_path):
    rows = _rows(_line(), tmp_path=tmp_path)
    scores = {"r1": [0.52, 0.31, 0.12]}
    # 0.35 to 0.50 all keep only the relevant 0.52 chunk; the tie takes the highest.
    assert ev.choose_threshold(rows, scores) == 0.5


def test_choose_threshold_falls_back_to_highest_recall(tmp_path):
    rows = _rows(_line(relevant=["a", "b"]), tmp_path=tmp_path)
    # Chunk b scores below every swept threshold, so recall cannot reach 0.95.
    scores = {"r1": [0.9, 0.01, 0.01]}
    assert ev.choose_threshold(rows, scores) == 0.7


def _scorer(chunk, prompt):
    return 0.9 if "alpha" in chunk else 0.1


def test_score_all_asks_the_generic_question_once_per_chunk(tmp_path):
    rows = _rows(_line(), tmp_path=tmp_path)
    prompts = []

    def score_fn(chunk, prompt):
        prompts.append(prompt)
        return _scorer(chunk, prompt)

    assert ev.score_all(rows, score_fn) == {"r1": [0.9, 0.1, 0.1]}
    assert len(prompts) == 3 and "what is open" in prompts[0]


def test_evaluate_freezes_the_tune_threshold_and_reports_holdout(tmp_path):
    rows = _rows(
        _line(),
        _line(id="r2", goal_type="find_fact", split="holdout", relevant=["a", "b"]),
        tmp_path=tmp_path,
    )
    report = ev.evaluate(rows, _scorer)
    # Only the tune row drives the choice; ties between kept sets go to the top of the grid.
    assert report["chosen_threshold"] == 0.7
    chosen = report["at_chosen"]
    assert chosen["all"]["tune"]["recall"] == 1.0
    assert chosen["find_fact"]["holdout"]["recall"] == 0.5
    assert chosen["all"]["all_rows"]["chunks"] == 6
    assert report["at_old_default"]["open_items"]["tune"]["compression"] == 0.667
    assert set(report["sweep"]) == {f"{t:.2f}" for t in ev.THRESHOLDS}


def test_format_report_lists_sweep_and_both_thresholds(tmp_path):
    rows = _rows(_line(), tmp_path=tmp_path)
    text = ev.format_report(ev.evaluate(rows, _scorer))
    assert "thr 0.05" in text and "at old default 0.30" in text
    assert "at chosen 0.70 (tune only)" in text


def test_live_score_fn_scores_through_the_app_provider(monkeypatch):
    from unittest.mock import MagicMock

    from jev_mcp import app

    provider = MagicMock()
    provider.fast_provider.evaluate_batch.return_value = {"chunk_0": {"noul": 0.4}}
    monkeypatch.setattr(app, "provider", provider)
    assert ev.live_score_fn()("text", "Q?") == 0.4


def test_main_prints_text_or_json(tmp_path, monkeypatch, capsys):
    path = tmp_path / "d.jsonl"
    path.write_text(_line() + "\n")
    monkeypatch.setattr(ev, "live_score_fn", lambda: _scorer)

    assert ev.main(["--dataset", str(path)]) == 0
    assert "pooled sweep" in capsys.readouterr().out
    assert ev.main(["--dataset", str(path), "--json"]) == 0
    assert "chosen_threshold" in json.loads(capsys.readouterr().out)
