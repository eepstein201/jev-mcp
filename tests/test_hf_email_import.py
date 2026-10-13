"""The HF email-classifier importer builds a stratified triage eval set outside the repo."""

import json
import sys
import types
from pathlib import Path

import pytest

from tests.evals import hf_email_import
from tests.evals.hf_email_import import (
    CATEGORIES,
    EXPECTED_ROUTING,
    select_rows,
    to_triage_row,
)
from tests.evals.triage_eval import load_dataset


def _hf_row(category: str, number: int, text: str | None = None) -> dict:
    body = f"{category} body {number}"
    return {
        "id": f"{category}_{number}",
        "subject": f"{category} subject {number}",
        "body": body,
        "text": text or f"{category} subject {number} {body}",
        "category": category,
        "category_id": CATEGORIES.index(category),
    }


def _hf_rows(per_category: int) -> list[dict]:
    return [_hf_row(c, n) for c in CATEGORIES for n in range(per_category)]


def test_a_row_becomes_a_triage_row_labelled_with_its_category():
    row = to_triage_row(_hf_row("promotions", 7))

    assert row["subject"] == "promotions subject 7"
    assert row["body"] == "promotions body 7"
    assert row["label"] == "promotions"
    assert row["expected_routing"] == EXPECTED_ROUTING == "archive_and_label"
    assert row["sender"]


def test_an_unknown_category_is_refused():
    with pytest.raises(ValueError, match="category"):
        to_triage_row({**_hf_row("forum", 1), "category": "invoices"})


def test_selection_takes_the_same_number_from_every_category():
    selected = select_rows(_hf_rows(10), per_class=3, seed=0)

    labels = [row["label"] for row in selected]
    assert len(selected) == 3 * len(CATEGORIES)
    assert {label: labels.count(label) for label in CATEGORIES} == {
        label: 3 for label in CATEGORIES
    }


def test_selection_is_repeatable_for_a_seed():
    rows = _hf_rows(10)

    assert select_rows(rows, per_class=3, seed=5) == select_rows(
        rows, per_class=3, seed=5
    )


def test_selection_drops_excluded_and_repeated_texts():
    rows = [
        *(_hf_row("forum", n) for n in range(4)),
        _hf_row("forum", 98, text="seen in train"),
        _hf_row("forum", 99, text="repeated"),
        _hf_row("forum", 100, text="repeated"),
    ]
    others = [_hf_row(c, n) for c in CATEGORIES if c != "forum" for n in range(4)]

    selected = select_rows(
        [*rows, *others],
        per_class=4,
        seed=0,
        exclude_texts=frozenset({"seen in train"}),
    )

    subjects = [row["subject"] for row in selected if row["label"] == "forum"]
    assert len(subjects) == 4
    assert "forum subject 98" not in subjects
    assert len({"forum subject 99", "forum subject 100"} & set(subjects)) <= 1


def test_selection_fails_when_a_category_is_too_small():
    with pytest.raises(ValueError, match="spam"):
        select_rows(
            [r for r in _hf_rows(5) if r["category"] != "spam" or r["id"] < "spam_3"],
            per_class=4,
            seed=0,
        )


def test_the_importer_writes_a_private_dataset_the_triage_eval_can_read(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        hf_email_import,
        "load_split",
        lambda split: _hf_rows(6 if split == "test" else 1),
    )
    out = tmp_path / "hf.jsonl"

    code = hf_email_import.main(["--per-class", "2", "--out", str(out)])

    assert code == 0
    rows = load_dataset(out)
    assert len(rows) == 2 * len(CATEGORIES)
    assert all(
        json.loads(line)["expected_routing"] == EXPECTED_ROUTING
        for line in out.read_text().splitlines()
    )


def test_a_missing_datasets_package_gets_an_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)

    with pytest.raises(RuntimeError, match="datasets"):
        hf_email_import.load_split("test")


def test_load_split_returns_the_rows_of_the_requested_split(monkeypatch):
    requested = []

    def fake_load_dataset(dataset_id, split):
        requested.append((dataset_id, split))
        return iter([_hf_row("forum", 1)])

    monkeypatch.setitem(
        sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset)
    )

    assert hf_email_import.load_split("test") == [_hf_row("forum", 1)]
    assert requested == [(hf_email_import.DATASET_ID, "test")]


def test_the_default_output_is_outside_the_repository():
    repo_root = Path(__file__).resolve().parent.parent

    assert not hf_email_import.DEFAULT_OUT.resolve().is_relative_to(repo_root)
