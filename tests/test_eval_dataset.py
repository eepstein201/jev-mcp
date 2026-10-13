"""The private dataset writer keeps labelled real email out of the repository."""

import json
import stat
from pathlib import Path

import pytest

from tests.evals.dataset import write_private_dataset

REPO_ROOT = Path(__file__).parent.parent

ROW = {
    "subject": "Dentist appointment moved to Friday",
    "sender": "office@clinic.example",
    "body": "Your Thursday cleaning is moved to Friday at 9am. Reply to confirm.",
    "expected_routing": "escalate_to_slack",
}


def test_a_dataset_path_inside_the_repository_is_refused(tmp_path):
    with pytest.raises(ValueError, match="repository"):
        write_private_dataset([ROW], REPO_ROOT / "tests" / "evals" / "real.jsonl")


def test_the_dataset_is_readable_by_its_owner_only(tmp_path):
    path = write_private_dataset([ROW], tmp_path / "triage_eval.jsonl")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text().splitlines()[0]) == ROW
