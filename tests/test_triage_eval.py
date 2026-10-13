"""The triage eval loads labelled rows, scores them, and breaks routing down by label."""

import json

import pytest

from tests.evals import triage_eval
from tests.evals.triage_eval import (
    format_report,
    load_dataset,
    routing_by_label,
    score_row,
)

ROW = {
    "subject": "Your code is 123456",
    "sender": "no-reply@auth.example",
    "body": "Use 123456 to sign in.",
    "expected_routing": "archive_and_label",
}


def _scored(label: str | None, got: str, expected: str = "archive_and_label") -> dict:
    row = {
        "subject": f"{label} email",
        "expected": expected,
        "got": got,
        "requires_action": 0.1,
        "is_important": 0.1,
    }
    return {**row, "label": label} if label else row


def _fake_triage(routing: str):
    def evaluate(subject, sender, body, labels):
        return {
            "suggested_routing": routing,
            "requires_action_score": 0.2,
            "is_important_score": 0.1,
        }

    return evaluate


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_routing_is_counted_per_label():
    scored = [
        _scored("promotions", "archive_and_label"),
        _scored("promotions", "hitl_slack_block"),
        _scored("verify_code", "escalate_to_slack"),
    ]

    assert routing_by_label(scored) == {
        "promotions": {"archive_and_label": 1, "hitl_slack_block": 1},
        "verify_code": {"escalate_to_slack": 1},
    }


def test_rows_without_a_label_are_left_out_of_the_breakdown():
    assert routing_by_label([_scored(None, "archive_and_label")]) == {}


def test_the_report_lists_each_label_with_its_routing_mix():
    report = format_report(
        [
            _scored("verify_code", "escalate_to_slack"),
            _scored("forum", "archive_and_label"),
        ]
    )

    assert "by label" in report
    assert "verify_code" in report
    assert "escalate_to_slack=1" in report


def test_the_report_has_no_breakdown_when_nothing_is_labelled():
    assert "by label" not in format_report([_scored(None, "archive_and_label")])


def test_a_row_missing_a_required_key_is_refused(tmp_path):
    path = _write(tmp_path / "rows.jsonl", [{"subject": "no body or sender"}])

    with pytest.raises(ValueError, match="missing"):
        load_dataset(path)


def test_a_row_with_an_unknown_routing_is_refused(tmp_path):
    path = _write(
        tmp_path / "rows.jsonl", [{**ROW, "expected_routing": "delete_everything"}]
    )

    with pytest.raises(ValueError, match="expected_routing"):
        load_dataset(path)


def test_a_scored_row_carries_its_label_when_it_has_one(monkeypatch):
    monkeypatch.setattr(
        triage_eval, "evaluate_email_with_mlx", _fake_triage("hitl_slack_block")
    )

    scored = score_row({**ROW, "label": "verify_code"}, ["Notifications"])

    assert scored["label"] == "verify_code"
    assert scored["got"] == "hitl_slack_block"
    assert scored["expected"] == "archive_and_label"
    assert "label" not in score_row(ROW, ["Notifications"])


def test_the_report_prints_the_agreement_and_label_breakdown(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        triage_eval, "evaluate_email_with_mlx", _fake_triage("archive_and_label")
    )
    path = _write(tmp_path / "rows.jsonl", [{**ROW, "label": "verify_code"}])

    code = triage_eval.main(["--dataset", str(path)])

    output = capsys.readouterr().out
    assert code == 0
    assert "agreement  1/1  100%" in output
    assert "verify_code: archive_and_label=1" in output


def test_json_output_has_the_agreement_and_every_row(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        triage_eval, "evaluate_email_with_mlx", _fake_triage("escalate_to_slack")
    )
    path = _write(tmp_path / "rows.jsonl", [ROW])

    triage_eval.main(["--dataset", str(path), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["agreement"] == {"runs": 1, "passes": 0, "rate": 0.0}
    assert payload["rows"][0]["got"] == "escalate_to_slack"
