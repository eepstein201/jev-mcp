"""The compaction evaluation harness, with the engine replaced by a scoring function."""

import json

import pytest

from jev_mcp.compaction import GENERIC_QUESTION, TYPED_QUESTIONS
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


def test_choose_threshold_takes_most_compression_at_full_recall(tmp_path):
    rows = _rows(_line(), tmp_path=tmp_path)
    scores = {"r1": [0.52, 0.31, 0.12]}
    # 0.35 to 0.50 all keep the relevant 0.52 chunk and drop the rest; a tie takes the lowest.
    assert ev.choose_threshold(rows, scores) == 0.35


def test_choose_threshold_falls_back_to_highest_recall(tmp_path):
    rows = _rows(_line(relevant=["a", "b"]), tmp_path=tmp_path)
    # Chunk b scores below every swept threshold, so recall cannot reach 0.95.
    scores = {"r1": [0.9, 0.05, 0.05]}
    assert ev.choose_threshold(rows, scores) == 0.1


def _scorer(chunk, prompt):
    return 0.9 if "alpha" in chunk else 0.1


def test_score_all_sends_each_chunk_question_pair_once(tmp_path):
    rows = _rows(_line(), tmp_path=tmp_path)
    calls = []

    def score_fn(chunk, prompt):
        calls.append((chunk, prompt))
        return _scorer(chunk, prompt)

    variants, predicted = ev.score_all(
        rows, score_fn, lambda goal: "open_items", TYPED_QUESTIONS
    )
    assert len(calls) == len(set(calls))
    assert predicted == {"r1": "open_items"}
    assert variants["baseline"]["r1"] == [0.9, 0.1, 0.1]
    assert set(variants) == {
        "baseline",
        "oracle_a",
        "oracle_b",
        "typed_oracle",
        "typed_classified",
    }


def test_score_all_uses_the_generic_question_when_the_goal_is_other(tmp_path):
    rows = _rows(_line(), tmp_path=tmp_path)
    prompts = []

    def score_fn(chunk, prompt):
        prompts.append(prompt)
        return 0.5

    ev.score_all(rows, score_fn, lambda goal: "other", TYPED_QUESTIONS)
    assert GENERIC_QUESTION.format(goal="what is open") in prompts


def test_evaluate_reports_every_variant_and_freezes_the_tune_threshold(tmp_path):
    rows = _rows(
        _line(),
        _line(id="r2", split="holdout", relevant=["a", "b"]),
        tmp_path=tmp_path,
    )
    report = ev.evaluate(rows, _scorer, lambda goal: "open_items")
    baseline = report["baseline"]["open_items"]
    # Tune picks 0.15, the lowest threshold that already drops both irrelevant chunks.
    assert baseline["threshold"] == 0.15
    assert baseline["tune"]["recall"] == 1.0
    assert baseline["holdout"]["recall"] == 0.5
    assert report["baseline@default"]["all"]["tune"]["compression"] == 0.667
    assert report["typed_classified@shipped"]["open_items"]["tune"]["recall"] == 1.0
    assert report["classifier"]["correct"] == 2
    assert report["classifier"]["other"] == 0


def test_format_report_lists_variants_and_classifier(tmp_path):
    rows = _rows(_line(), tmp_path=tmp_path)
    text = ev.format_report(ev.evaluate(rows, _scorer, lambda goal: "other"))
    assert "baseline@default" in text and "thr 0.15" in text
    assert "classifier  0/1 correct, 1 other" in text


def test_live_functions_score_and_classify_through_the_app_provider(monkeypatch):
    from unittest.mock import MagicMock

    from jev_mcp import app

    provider = MagicMock()
    provider.fast_provider.evaluate_batch.side_effect = [
        {"chunk_0": {"noul": 0.4}},
        {"goal_type": {"choice": "find_fact"}},
    ]
    monkeypatch.setattr(app, "provider", provider)
    score_fn, classify_fn = ev.live_functions()
    assert score_fn("text", "Q?") == 0.4
    assert classify_fn("what port") == "find_fact"


def test_main_prints_text_or_json(tmp_path, monkeypatch, capsys):
    path = tmp_path / "d.jsonl"
    path.write_text(_line() + "\n")
    monkeypatch.setattr(ev, "live_functions", lambda: (_scorer, lambda goal: "other"))

    assert ev.main(["--dataset", str(path)]) == 0
    assert "baseline" in capsys.readouterr().out
    assert ev.main(["--dataset", str(path), "--json"]) == 0
    assert "classifier" in json.loads(capsys.readouterr().out)


def test_each_shipped_question_is_one_of_the_two_candidates():
    for goal_type, shipped in TYPED_QUESTIONS.items():
        assert shipped in ev.CANDIDATES[goal_type].values()
